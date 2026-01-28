import os
import sys
import json
import logging
import asyncio
import shutil
import base64
import mimetypes
import re  # [New]
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client
from logging.handlers import RotatingFileHandler
from mcp import ClientSession, StdioServerParameters
from geopy.geocoders import Nominatim # [New]
from geopy.exc import GeocoderTimedOut, GeocoderServiceError # [New]

from model import Model  # 自定义模型封装类

# ================= 配置区域 =================
# 代理配置
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

# 路径配置
IMAGE_BASE_ROOT = Path(r"E:\Idea2Image-Bench\Benchmark\Idea2Image-Bench") # 图像数据的根目录
INPUT_JSONL_PATH = r"E:\Idea2Image-Bench\Benchmark\Idea2Image-Bench\checklist\2_Weather.jsonl" # 输入文件路径
OUTPUT_ROOT_PATH = Path(r"E:\Idea2Image-Bench\Idea2Image-Agent\agent-6\output") # 结果输出根目录

# 设置工作目录为当前脚本所在路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 全局变量
temp_dir_path = None
logger = None

# ================= 日志与工具函数 =================

class JsonFormatter(logging.Formatter):
    def format(self, record):
        convs_out = []
        if isinstance(record.args, tuple) and len(record.args) > 0 and isinstance(record.args[0], list):
             for conv in record.args[0]:
                try:
                    if 'tool_calls' in conv:
                        for tool_call in conv['tool_calls']:
                            if isinstance(tool_call['function']['arguments'], str):
                                tool_call['function']['arguments'] = json.loads(tool_call['function']['arguments'])
                except:
                    pass
                finally:
                    convs_out.append(conv)
        
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "conversations": convs_out
        }
        return json.dumps(log_record, ensure_ascii=False, indent=4)

def init_globel_params():
    global temp_dir_path, logger
    # 定义固定的中转站，位于当前目录下的 tmp/buffer
    temp_dir_path = Path('./tmp/buffer1').absolute()
    
    # 每次启动前清理整个中转站，保证环境干净
    if temp_dir_path.exists():
        try:
            shutil.rmtree(temp_dir_path)
        except Exception as e:
            print(f"⚠️ 清理缓存目录失败 (可能是文件占用): {e}")

    temp_dir_path.mkdir(parents=True, exist_ok=True)

    # 预先创建子目录，供工具使用
    (temp_dir_path / 'draw_output').mkdir(parents=True, exist_ok=True)
    (temp_dir_path / 'search').mkdir(parents=True, exist_ok=True)
    (temp_dir_path / 'search_text').mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("text_logger")
    logger.setLevel(logging.INFO)
    # 使用 RotatingFileHandler 防止单个日志过大
    handler = RotatingFileHandler(temp_dir_path / "app.log", maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return temp_dir_path, logger

def clear_buffer_files(dir_path: Path):
    """只清空文件，不删文件夹"""
    if not dir_path.exists(): return
    for item in dir_path.iterdir():
        if item.name == "app.log": continue # 保留日志文件句柄
        try:
            if item.is_file(): item.unlink()
            elif item.is_dir(): shutil.rmtree(item)
        except: pass

def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        return None # 不抛出异常，只是返回None表示无法编码

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError:
        return None

# ================= MCP 客户端类 =================

class MCPClient:
    def __init__(self, model_name: str, provider: str, base_url: str = None, api_key: str = None, 
                 global_env: dict = {}, servers_info: dict = {}, temp_dir_path: Path = None, 
                 need_context: bool = True, system_prompt: str = None):
        
        self.model_name = model_name
        self.servers_info = servers_info
        self.global_env = global_env
        self.sessions = {}
        self.tool_mapping = {}
        self.exit_stack = AsyncExitStack()
        self.client = Model(
            model_name=model_name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt
        )
        self.temp_dir_path = temp_dir_path
        self.need_context = need_context
        self.context = []

    async def initialize_sessions(self):
        for server_name, server_info in self.servers_info.items():
            # 强制将工具的输出指向我们的固定中转站
            if server_name == 'ImageGen':
                server_info['args'][-1] = str(self.temp_dir_path / 'draw_output')
            if 'RAG' in server_name or 'Image_RAG' in server_name:
                server_info['args'][-1] = str(self.temp_dir_path / 'search')
            if 'Text_RAG' in server_name:
                 server_info['args'][-1] = str(self.temp_dir_path / 'search_text')
                
            # 合并环境变量
            current_env = os.environ.copy()
            current_env.update(self.global_env)
            if server_info.get('env'):
                current_env.update(server_info['env'])

            server_params = StdioServerParameters(
                command=server_info['command'],
                args=server_info['args'],
                env=current_env,
            )

            write, read = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(write, read))
            await session.initialize()

            self.sessions[server_name] = session
            self.tool_mapping[server_name] = session # 简化映射逻辑，或者保持原有的遍历 logic

            # 获取支持的工具并建立映射 (兼容原有逻辑)
            response = await session.list_tools()
            for tool in response.tools:
                prefixed_name = f"{server_name}_{tool.name}"
                self.tool_mapping[prefixed_name] = (session, tool.name)
            
            # print(f"已连接: {server_name}")

    async def cleanup(self):
        await self.exit_stack.aclose()

    # [New] 添加地理编码预处理函数
    def _preprocess_coordinates(self, text: str) -> str:
        """
        识别文本中的地理坐标并替换为地名（精确到城市/区县）。
        支持格式示例：
        - 39.9917N, 116.3906E
        - 39°59'30.1"N 116°23'26.2"E
        - 39.9917, 116.3906
        """
        geolocator = Nominatim(user_agent="mcp_geo_bot", timeout=10)

        pattern = re.compile(
            r'(?:'
            # 模式1: 纬度(含N/S) + 分隔符 + 经度(含E/W)
            r'(?:[-+]?\d{1,3}(?:\.\d+)?(?:[°d]\s*\d{1,2}(?:\.\d+)?)?(?:[\'′]\s*\d{1,2}(?:\.\d+)?)?[″"]?\s*[NS])'
            r'\s*[,，\s]\s*'
            r'(?:[-+]?\d{1,3}(?:\.\d+)?(?:[°d]\s*\d{1,2}(?:\.\d+)?)?(?:[\'′]\s*\d{1,2}(?:\.\d+)?)?[″"]?\s*[EW])'
            r')|(?='
            # 模式2: 纯数字对
            r'[-+]?\d+\.\d+\s*[,，]\s*[-+]?\d+\.\d+'
            r')'
            r'(?:[-+]?\d+\.\d+\s*[,，]\s*[-+]?\d+\.\d+)',
            re.IGNORECASE | re.VERBOSE
        )

        def replacer(match):
            coord_str = match.group(0)
            try:
                location = geolocator.reverse(coord_str, language='en')
                
                if location and location.raw.get('address'):
                    addr = location.raw['address']
                    
                    state = addr.get('state')
                    country = addr.get('country')
                    
                    parts = [p for p in [state, country] if p]
                    
                    seen = set()
                    cleaned_parts = []
                    for p in parts:
                        if p not in seen:
                            cleaned_parts.append(p)
                            seen.add(p)
                            
                    place_name = ", ".join(cleaned_parts)
                    
                    # print(f"🌍 [Geo Detect] 坐标 '{coord_str}' 已替换为 -> '{place_name}'")
                    return f"{place_name}"
            
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"⚠️ [Geo Warning] API 请求超时: {e}")
            except Exception as e:
                print(f"⚠️ [Geo Error] 解析出错: {e}")
            
            return coord_str

        return pattern.sub(replacer, text)

    async def process_query(self, query: str, image_path: Optional[str] = None):
        """
        处理单条用户查询，支持图像路径注入
        """
        # 1. 每次处理前清空上下文 (批量模式下每个任务是独立的)
        self.context = [] 
        
        # [New] 在处理前先进行地理坐标解析 (使用 asyncio.to_thread 避免阻塞)
        # print("🔍 正在检查地理位置信息...")
        processed_query = await asyncio.to_thread(self._preprocess_coordinates, query)
        user_content = processed_query
        
        if image_path and ", " in image_path:
            # 兼容多张图片路径，只选取第一张图像
            image_path = image_path.split(", ")[0]
        
        # 2. 处理图片路径注入
        if image_path:
            clean_path = str(image_path).strip().strip('"').strip("'")
            if os.path.exists(clean_path):
                # 注入 System Note，指导 Agent 使用 intent_analyzer 读取图片
                user_content += f"\n\n[System Note: The user has provided an input image located at: {clean_path}. If you call 'intent_analyzer', you MUST pass this path string to its 'image_path' argument.]"
            else:
                print(f"⚠️ 图片路径不存在: {clean_path}")

        messages = self.context
        messages.append({"role": "user", "content": user_content})

        # 3. 收集工具定义
        available_tools = []
        for server_id, session in self.sessions.items():
            try:
                response = await session.list_tools()
                for tool in response.tools:
                    prefixed_name = f"{server_id}_{tool.name}"
                    available_tools.append({
                        "type": "function",
                        "function": {
                            "name": prefixed_name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    })
            except: pass 

        # 4. 调用 LLM
        try:
            response = await self.client.chat(messages=messages, tools=available_tools)
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return

        message = response[0]
        if message.content:
            messages.append({"role": "assistant", "content": message.content})

        # 5. 循环工具调用
        loop_count = 0
        MAX_LOOPS = 15 # 防止死循环

        while message.tool_calls and loop_count < MAX_LOOPS:
            loop_count += 1
            for tool_call in message.tool_calls:
                prefixed_name = tool_call.function.name
                
                content_str = "Error: Tool not found"
                if prefixed_name in self.tool_mapping:
                    session, original_tool_name = self.tool_mapping[prefixed_name]
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        # print(f"🛠️ Tool: {prefixed_name}")
                        result = await session.call_tool(original_tool_name, tool_args)
                        
                        # 兼容不同类型的 Content 返回
                        if hasattr(result, 'content') and isinstance(result.content, list):
                            content_str = "".join([item.text for item in result.content if hasattr(item, 'text')])
                        else:
                            content_str = str(result)
                            
                    except Exception as e:
                        content_str = f"Error executing tool: {str(e)}"
                        print(f"❌ Tool Error ({prefixed_name}): {e}")

                # 记录工具结果到历史
                messages.extend([
                    {
                        "role": "assistant", 
                        "tool_calls": [{
                            "id": tool_call.id, 
                            "type": "function", 
                            "function": {"name": prefixed_name, "arguments": tool_call.function.arguments}
                        }]
                    },
                    {"role": "tool", "tool_call_id": tool_call.id, "content": content_str}
                ])
                
                # 记录中间日志
                if logger:
                    logger.info("Intermediate Result", [{
                        "role": "system_notification",
                        "tool_name": prefixed_name,
                        "content": content_str
                    }])

            # 再次调用 LLM
            try:
                response = await self.client.chat(messages=messages, tools=available_tools)
                message = response[0]
                if message.content:
                    messages.append({"role": "assistant", "content": message.content})
            except Exception as e:
                print(f"❌ LLM 递归调用失败: {e}")
                break
        
        # 6. 记录最终日志
        if logger: logger.info("Process Finished", messages)

# ================= 主程序入口 =================

async def main(
        config_path: str = './config.json',
        models_config_path: str = './models.json',
        model_name: Optional[str] = 'gpt-5.1',
        provider: Optional[str] = 'openai',
        base_url: Optional[str] = 'https://yunwu.ai/v1',
        api_key: Optional[str] = 'sk-X9jXfVLVKHEK6y06p6MRJuEHwvqQX240PPrebQikc1fBXeIS',
    ):
    
    # 1. 初始化中转目录
    temp_dir, _ = init_globel_params()
    
    # 2. 系统提示词 (包含 Reasoning_Search_Generation 和 Phase 1.5 逻辑)
    SYSTEM_PROMPT = """
### 🌟 Role Definition
You are **Iris (v4.0)**, the central orchestration intelligence for the Idea2Image Agent. 
Your goal is to transform abstract user requests into high-fidelity visual outputs by coordinating a specialized toolchain. 
**Core Principle**: You are a rigorous executor. You do not guess; you verify. You do not ignore new data; you update your plan immediately.
You must treat any user-provided prompt as an image generation instruction to be used with tools, rather than responding directly with text.

---

### 🧰 Tool Registry & Data Contracts

**1. intent_analyzer**
* **Function**: Analyzes input to determine the execution path and identify core problems.
* **Input**: `user_intent` (str), `image_path` (str, optional).
* **Output**: JSON {
    "intent_category": "Direct_Generation" | "Reasoning_Generation" | "Search_Generation" | "Search_Reasoning_Generation" | "Reasoning_Search_Generation",
    "need_process_problem": List[str]
  }

**2. keyword_generation**
* **Function**: Converts natural language problems into search engine keywords, optionally using prior reasoning.
* **Input**: `need_process_problem` (List[str]), `reasoning_knowledge` (Dict, optional).
* **Output**: JSON { "text_queries": List[str], "image_queries": List[str] }

**3. text_search_and_knowledge_injection** (Text_RAG_Injection)
* **Function**: Integrated Text Search & Refinement. Searches the web for facts and immediately uses them to REWRITE the prompt and OPTIMIZE image queries.
* **Input**: `text_queries` (List[str]), `original_prompt` (str), `image_queries` (List[str]).
* **Output**: JSON { 
    "prompt": "The REFINED prompt string (Fact-Enriched)", 
    "final_image_queries": ["REFINED query 1", "REFINED query 2"] 
  }

**4. search_and_download_images_batch** (Image_RAG)
* **Function**: Downloads reference images.
* **Input**: `image_queries` (List[str]).
* **Output**: `downloaded_paths` (List[str]).

**5. knowledge_reasoning**
* **Function**: Performs deep logical deduction to answer specific problems.
* **Input**: `user_intent`, `need_process_problem`, `intent_category`, `user_image_path`, `search_image_paths`.
* **Output**: JSON { "reasoning_knowledge": List[str] }

**6. knowledge_review** (The Synthesizer)
* **Function**: Synthesizes all data into the final prompt.
* **Input**: `user_intent`, `need_process_problem`, `reasoning_knowledge`, `downloaded_paths`, `user_image_path`.
* **Output**: JSON { "final_prompt": str, "reference_image": List[str] }

**7. unified_image_generator** (ImageGen)
* **Function**: Generates the final image.
* **Input**: `prompt`, `reference_images`.
* **Output**: List[str].

---

### ⚙️ Workflow Protocol (Strict Execution Chain)

You must execute the following logic step-by-step.

#### Phase 1: Analysis (The Router)
1.  **Action**: Call `intent_analyzer(user_intent, image_path)`.
2.  **Observation**: Identify `intent_category` and `need_process_problem`.
3.  **Branching**:
    * IF `Direct_Generation`: Go to **Phase 5**.
    * IF `Reasoning_Generation`: Go to **Phase 4**.
    * IF `Reasoning_Search_Generation`: Go to **Phase 2**.
    * IF `Search_Generation` OR `Search_Reasoning_Generation`: Go to **Phase 3**.

#### Phase 2: Pre-Search Reasoning (Only for Reasoning_Search_Generation)
1.  **Action**: Call `knowledge_reasoning`.
    * **Inputs**: `user_intent`, `need_process_problem`, `intent_category`, `user_image_path` (Search inputs are None).
    * **Get**: `reasoning_knowledge`.
3.  **Transition**: Go to **Phase 3**.

#### Phase 3: Retrieval (The RAG Loop)
1.  **Action**: Call `keyword_generation`.
    * **Inputs**: 
        * `need_process_problem`: (From Phase 1)
        * `reasoning_knowledge`: you MUST input it explicitly from Phase 2 (if available).
    * **Get**: `text_queries`, `image_queries` (Draft Version).

2.  **Logic Check & Search Execution**:
    * **CASE A: Text Search IS Required (`text_queries` is NOT empty)**
        * **Action**: Call `text_search_and_knowledge_injection(text_queries, user_intent, image_queries)`.
        * **⚠️ DATA OVERRIDE PROTOCOL (CRITICAL)**: 
            * The tool returns a refined `prompt` and `final_image_queries`.
            * **YOU MUST DISCARD** the old `image_queries` and the old `user_intent`.
            * **YOU MUST USE** the new values for all subsequent steps.
            * *Reasoning Trace*: "I have received the fact-enriched prompt and optimized queries. I will use them as the new truth."
        * **Update Context**: 
            * Let `current_image_queries` = `final_image_queries` (from tool output).
            * Let `current_user_intent` = `prompt` (from tool output).
    * **CASE B: No Text Search (`text_queries` is empty)**
        * Let `current_image_queries` = `image_queries` (from Step 1).
        * Let `current_user_intent` = `user_intent` (Original input).

3.  **Action**: Call `search_and_download_images_batch(image_queries=current_image_queries)`.
    * **Get**: `downloaded_paths`.

4.  **Branching Return**:
    * IF `Search_Reasoning_Generation`: Go to **Phase 4**.
    * IF `Search_Generation` OR `Reasoning_Search_Generation`: Go to **Phase 5**. (For `Reasoning_Search`, reasoning was already done in Phase 1.5).

#### Phase 4: Reasoning (The Logic Engine)
1.  **Action**: Call `knowledge_reasoning`.
    * **Inputs**:
        * `user_intent`: **MUST use `current_user_intent`** (The refined prompt if Phase 3 updated it).
        * `search_image_paths`: Use `downloaded_paths` from Phase 3 (if available).
        * Pass other args as defined in registry.
    * **Get**: `reasoning_knowledge` (Update reasoning knowledge).
2.  **Transition**: Go to **Phase 5**.

#### Phase 5: Synthesis (The Review)
1.  **Action**: Call `knowledge_review`.
    * **Inputs**:
        * `original_request`: **MUST use `current_user_intent`** (The refined prompt if Phase 3 updated it).
        * `need_process_problem`: From Phase 1.
        * `reasoning_knowledge`: **CRITICAL CHECK**:
            * IF `Reasoning_Search_Generation`: Use `reasoning_knowledge` from **Phase 2**.
            * IF `Search_Reasoning_Generation` or `Reasoning_Generation`: Use `reasoning_knowledge` from **Phase 4**.
        * `downloaded_image_paths`: From Phase 3 (or empty List).
        * `user_image_path`: The user's uploaded image path.
    * **Get**: `final_prompt`, `reference_image`.

#### Phase 6: Execution (The Artist)
1.  **Action**: Call `unified_image_generator`.
    * **Inputs**:
        * `prompt`: `final_prompt` from Phase 5.
        * `reference_images`: `reference_image` from Phase 5.
    * **End**: Output the final result.

---

### 🚨 Critical Rules

1.  **Context Update**: When `text_search_and_knowledge_injection` returns, you **MUST** map its output (`final_image_queries`, `prompt`) to `current_image_queries` and `current_user_intent`. **Do not blindly reuse the old variables.**
2.  **Variable Persistence**: If you generate `reasoning_knowledge` in Phase 2, you must carry it safely through Phase 3 and deliver it to Phase 5. Do not drop it.
3.  **No Hallucination**: If a tool returns an empty list `[]`, pass an empty list `[]`. Do not invent file paths.
4.  **Silent Operation**: Do not output your internal thought process to the user unless specifically asked. Just execute the tool calls.
"""
    # 3. 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    global_env = {}
    if os.path.exists(models_config_path):
        with open(models_config_path, 'r') as f:
            models_data = json.load(f)
            global_env = models_data.get('global_env', {})
            if models_data.get('models'):
                m = models_data['models'][0]
                model_name = m.get('model', model_name)
                api_key = m.get('apiKey', api_key)
                base_url = m.get('apiBase', base_url)

    # 4. 初始化客户端
    client = MCPClient(
        model_name=model_name,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        servers_info=config['mcpServers'],
        temp_dir_path=temp_dir,
        need_context=False, # 批量模式不需要上下文
        system_prompt=SYSTEM_PROMPT,
        global_env=global_env
    )

    # 5. 读取 JSONL 文件
    tasks = []
    if os.path.exists(INPUT_JSONL_PATH):
        with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        tasks.append(json.loads(line))
                    except: pass
    else:
        print(f"❌ 输入文件不存在: {INPUT_JSONL_PATH}")
        return

    # 设置输出目录
    input_filename = os.path.basename(INPUT_JSONL_PATH).split('.')[0]
    final_output_root = OUTPUT_ROOT_PATH / input_filename
    final_output_root.mkdir(parents=True, exist_ok=True)

    try:
        print("正在初始化服务...")
        await client.initialize_sessions()
        print(f"🚀 开始批量处理 {len(tasks)} 个任务...")
        print(f"📂 输出目录: {final_output_root}")

        # 定义中转目录快捷引用
        buf_draw = temp_dir / 'draw_output'
        buf_img = temp_dir / 'search'
        buf_txt = temp_dir / 'search_text'

        # 6. 批量循环
        for item in tqdm(tasks, desc="Generating", unit="task"):
            try:
                # 提取任务信息
                prompt = item.get("Prompt")
                idx = item.get("ID", "unknown")
                type_name = item.get("Type", "unknown")
                rel_img_path = item.get("Image_Name")

                if not prompt: continue
                
                # --- 拼接绝对图片路径 ---
                abs_img_path = None
                if rel_img_path:
                    abs_img_path = str(IMAGE_BASE_ROOT / rel_img_path)
                    if not os.path.exists(abs_img_path):
                        tqdm.write(f"⚠️ 警告: 图片文件未找到: {abs_img_path}")
                        # 即使找不到图片，也继续执行，只是 image_path 为 None

                # --- 检查是否已存在结果 (跳过逻辑) ---
                task_output_dir = final_output_root / f"{type_name}_{idx}"
                if (task_output_dir / "guided.png").exists() or (task_output_dir / "textgen.png").exists():
                    tqdm.write(f"⏭️  ID {idx} 已存在结果，跳过。")
                    continue
                
                # A. 清空中转站 (只删文件，不删目录)
                # 先清空日志文件内容
                with open(temp_dir / "app.log", "w", encoding="utf-8") as f:
                    f.truncate(0)
                    
                clear_buffer_files(buf_draw)
                clear_buffer_files(buf_img)
                clear_buffer_files(buf_txt)

                # B. 创建任务目标文件夹
                task_output_dir.mkdir(parents=True, exist_ok=True)

                # C. 执行任务 (传入 prompt 和 绝对路径的 image_path)
                await client.process_query(prompt, image_path=abs_img_path)

                # D. 搬运结果
                # 1. 搬运生成的图片
                if buf_draw.exists():
                    for f in buf_draw.iterdir():
                        shutil.move(str(f), str(task_output_dir / f.name))
                
                # 2. 搬运图片搜索结果
                if buf_img.exists() and any(buf_img.iterdir()):
                    (task_output_dir / "image_search").mkdir(exist_ok=True)
                    for f in buf_img.iterdir():
                        shutil.move(str(f), str(task_output_dir / "image_search" / f.name))

                # 3. 搬运文本搜索结果
                if buf_txt.exists() and any(buf_txt.iterdir()):
                    (task_output_dir / "text_search").mkdir(exist_ok=True)
                    for f in buf_txt.iterdir():
                        shutil.move(str(f), str(task_output_dir / "text_search" / f.name))

                # 4. 保存 metadata (包含原始 JSON 信息)
                with open(task_output_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
                    
                # 5. 搬运日志 (刷新 Handler 防止丢失)
                if logger:
                    for h in logger.handlers: h.flush()
                shutil.copy(str(temp_dir / "app.log"), str(task_output_dir / "app.log"))

            except Exception as e:
                tqdm.write(f"❌ ID {idx} 处理出错: {e}")
                import traceback
                traceback.print_exc()

    finally:
        await client.cleanup()
        print(f"\n✅ 全部完成。")

if __name__ == "__main__":
    asyncio.run(main())