import os
import sys
import json
import logging
import asyncio
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


import base64
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client  # MCP 通信模块
from logging.handlers import RotatingFileHandler
from mcp import ClientSession, StdioServerParameters
import yaml

from model import Model  # 自定义模型封装类

os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

# 设置工作目录为当前脚本所在路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 全局临时目录路径与日志器对象
temp_dir_path = None
logger = None



# 定义一个自定义的日志格式化器，用于将对话记录格式化为 JSON 结构
class JsonFormatter(logging.Formatter):
    def format(self, record):
        convs_out = []
        for conv in record.args[0]:
            try:
                for tool_call in conv['tool_calls']:
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

# 初始化临时目录与日志记录器
def init_globel_params():
    global temp_dir_path, logger
    temp_dir_path = Path('./tmp/{}'.format(datetime.now().strftime('%y-%m-%d_%H-%M'))).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("text_logger")
    logger.setLevel(logging.INFO)
    # 强制指定 utf-8 编码防止特殊字符报错
    handler = RotatingFileHandler(temp_dir_path / "app.log", encoding='utf-8')  
    handler.setFormatter(JsonFormatter()) 
    logger.addHandler(handler)
    return temp_dir_path, logger


def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")

# MCP 主客户端类
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

    # 遍历配置文件中的工具服务器
    async def initialize_sessions(self):
        for server_name, server_info in self.servers_info.items():
            # 设置每个工具对应的输出目录
            # if server_name == 'ImageGen':
            #     server_info['args'][-1] = str(self.temp_dir_path / 'draw_output')
            # if server_name == 'RAG':
            #     server_info['args'][-1] = str(self.temp_dir_path / 'search')
                
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

            # 创建 stdio 客户端通道
            write, read = await self.exit_stack.enter_async_context(stdio_client(server_params))
            # 创建会话并初始化
            session = await self.exit_stack.enter_async_context(ClientSession(write, read))
            await session.initialize()

            # 注册会话
            self.sessions[server_name] = session

            # 获取支持的工具并建立映射
            response = await session.list_tools()
            for tool in response.tools:
                prefixed_name = f"{server_name}_{tool.name}" 
                self.tool_mapping[prefixed_name] = (session, tool.name)
            print(f"\n已连接到服务器 {server_name}，支持以下工具:", [tool.name for tool in response.tools])

            self.context = []

    # 关闭所有连接与资源
    async def cleanup(self):
        await self.exit_stack.aclose()

    def clear_history(self):
        self.context = []
        
    def _preprocess_coordinates(self, text: str) -> str:
        """
        识别文本中的地理坐标并替换为地名（精确到城市/区县）。
        支持格式示例：
        - 39.9917N, 116.3906E
        - 39°59'30.1"N 116°23'26.2"E
        - 39.9917, 116.3906
        """
        # 初始化 geolocator (注意：建议指定 user_agent)
        # 这里的代理设置通常会自动读取 os.environ 中的 http_proxy
        geolocator = Nominatim(user_agent="mcp_geo_bot", timeout=10)

        # 核心正则：匹配两种主要模式
        # 1. 带方向后缀模式 (DMS 或 Decimal + N/S/E/W): e.g., 39.9N, 116.9E 或 39°...N
        # 2. 纯数字模式 (Decimal pair): e.g., 39.9917, 116.3906 (中间必须有逗号，防止误伤其他数字)
        pattern = re.compile(
            r'(?:'
            # 模式1: 纬度(含N/S) + 分隔符 + 经度(含E/W)
            r'(?:[-+]?\d{1,3}(?:\.\d+)?(?:[°d]\s*\d{1,2}(?:\.\d+)?)?(?:[\'′]\s*\d{1,2}(?:\.\d+)?)?[″"]?\s*[NS])'
            r'\s*[,，\s]\s*'
            r'(?:[-+]?\d{1,3}(?:\.\d+)?(?:[°d]\s*\d{1,2}(?:\.\d+)?)?(?:[\'′]\s*\d{1,2}(?:\.\d+)?)?[″"]?\s*[EW])'
            r')|(?='
            # 模式2: 纯数字对 (利用前瞻断言确保是坐标对格式)
            r'[-+]?\d+\.\d+\s*[,，]\s*[-+]?\d+\.\d+'
            r')'
            r'(?:[-+]?\d+\.\d+\s*[,，]\s*[-+]?\d+\.\d+)',
            re.IGNORECASE | re.VERBOSE
        )

        def replacer(match):
            coord_str = match.group(0)
            try:
                # 修改点 1: 将语言设置为英文 ('en')
                # 这样美国的坐标会返回 "Los Angeles"，中国的会返回 "Beijing" (或拼音)
                # 这种格式最适合作为 Prompt 输入给绘图或推理模型
                location = geolocator.reverse(coord_str, language='en')
                
                if location and location.raw.get('address'):
                    addr = location.raw['address']
                    
                    # 修改点 2: 动态判断是否需要中文 (可选)
                    # 如果你希望中国境内的坐标显示汉字，而非拼音，可以加一个判断：
                    # if addr.get('country_code') == 'cn':
                    #     location = geolocator.reverse(coord_str, language='zh-CN')
                    #     addr = location.raw['address']

                    # 修改点 3: 结构化提取，避免重复内容
                    # 优先获取具体的“市/镇/区”
                    # city = (addr.get('city') or 
                    #         addr.get('town') or 
                    #         addr.get('village') or 
                    #         addr.get('county') or 
                    #         addr.get('hamlet'))
                    # city = addr.get('city')
                    state = addr.get('state')
                    country = addr.get('country')
                    
                    # 过滤掉空值，并用逗号连接
                    # 结果示例: "Los Angeles, California, United States"
                    parts = [p for p in [state, country] if p]
                    
                    # 去重逻辑 (防止出现 "New York, New York" 这种由于城市和州同名的情况)
                    seen = set()
                    cleaned_parts = []
                    for p in parts:
                        if p not in seen:
                            cleaned_parts.append(p)
                            seen.add(p)
                            
                    place_name = ", ".join(cleaned_parts)
                    
                    print(f"🌍 [Geo Detect] 坐标 '{coord_str}' 已替换为 -> '{place_name}'")
                    # 返回格式：地名 (保留原坐标)
                    return f"{place_name}"
            
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"⚠️ [Geo Warning] API 请求超时: {e}")
            except Exception as e:
                print(f"⚠️ [Geo Error] 解析出错: {e}")
            
            # 失败则返回原字符串
            return coord_str

        # 执行替换
        return pattern.sub(replacer, text)

    # 处理单条用户查询（已移除显式记忆模块）
    async def process_query(self, query: str, image_path: Optional[str] = None) -> str:
        """
        处理用户请求，支持文本和可选的图片路径。
        """
        
        # >>>>> 新增代码开始 >>>>>
        # 在处理任何逻辑前，先进行坐标替换
        # 注意：geopy 是同步阻塞请求，如果对高并发极其敏感，建议放入 executor 运行；
        # 但对于 CLI 聊天应用，直接调用通常没问题。
        print("🔍 正在检查地理位置信息...")
        processed_query = await asyncio.to_thread(self._preprocess_coordinates, query)
        # <<<<< 新增代码结束 <<<<<

        user_content = processed_query  # 将 query 改为 processed_query
        
        # user_content = query
        
        # 处理图片路径注入
        if image_path:
            clean_path = image_path.strip().strip('"').strip("'")
            
            # # 转换为base64编码
            # with open(clean_path, 'rb') as f:
            #     image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            if os.path.exists(clean_path):
                user_content += f"\n\n[System Note: The user has provided an input image located at: {clean_path}. If you call 'intent_analyzer', you MUST pass this path string to its 'image_path' argument.]"
            else:
                return f"❌ 错误: 找不到图片路径: {clean_path}"

        # 构建消息历史
        if self.need_context:
            messages = self.context
            messages.append({"role": "user", "content": user_content})
        else:
            messages = [{"role": "user", "content": user_content}]

        # 1. 收集工具
        available_tools = []
        for server_id, session in self.sessions.items():
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

        # 2. 初次调用 LLM
        response = await self.client.chat(messages=messages, tools=available_tools)
        final_text = []
        message = response[0]
        print(message)
        final_text.append(message.content or "")
        
        if message.content:
            messages.append({"role": "assistant", "content": message.content})

        # 3. 循环处理工具调用
        while message.tool_calls:
            for tool_call in message.tool_calls:
                prefixed_name = tool_call.function.name
                if prefixed_name in self.tool_mapping:
                    session, original_tool_name = self.tool_mapping[prefixed_name]
                    
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        # print(f"🔧 Tool Call: {original_tool_name} | Args: {tool_args}")
                        result = await session.call_tool(original_tool_name, tool_args)
                    except Exception as e:
                        result = type('obj', (object,), {'content': [type('obj', (object,), {'text': f"Error: {str(e)}"})]})
                        print(f"❌ Tool Error: {e}")

                    # 提取结果文本
                    content_str = ""
                    if hasattr(result, 'content') and isinstance(result.content, list):
                        for item in result.content:
                            if hasattr(item, 'text'):
                                content_str += item.text
                    else:
                        content_str = str(result)
                    
                    # 写入日志 (符合 JsonFormatter 格式)
                    logger.info("Intermediate Result", [{
                        "role": "system_notification",
                        "tool_name": original_tool_name,
                        "content": content_str
                    }])

                    # print(f"🛠️ 调用工具: {prefixed_name}\n📊 参数: {tool_args}\n📝 结果: {content_str[:200]}... (略)\n")

                    final_text.append(f"🛠️ 调用工具: {prefixed_name}\n📊 参数: {tool_args}\n📝 结果: {content_str[:200]}... (略)\n")

                    # 更新对话历史
                    messages.extend([
                        {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {"name": prefixed_name, "arguments": json.dumps(tool_args)},
                            }],
                            "content": ""
                        },
                        {"role": "tool", "tool_call_id": tool_call.id, "content": content_str},
                    ])
                else:
                    final_text.append(f"❌ 工具 {prefixed_name} 未找到\n")

            # 再次调用 LLM
            response = await self.client.chat(messages=messages, tools=available_tools)
            message = response[0]
            print(message)
            if message.content:
                final_text.append(message.content)
                messages.append({"role": "assistant", "content": message.content})

        if self.need_context:
            self.context = messages

        out = "\n".join(final_text)
        logger.info("Chat content", messages)
        return out

    async def chat_loop(self):
        print("\n🚀 MCP 多模态客户端已启动")
        print("--------------------------------------------------")
        print("提示: 输入 'quit' 退出, 'new chat' 清空历史。")
        print("--------------------------------------------------")
        
        while True:
            try:
                query = input("\n🗣️  Prompt (文本指令): ").strip()
                if not query: continue
                if query.lower() == "quit": break
                if query.lower() == "new chat":
                    self.clear_history()
                    print("🧹 历史记录已清空")
                    continue
                
                image_path = input("🖼️  Image Path (图片路径, 回车跳过): ").strip()
                if image_path == "":
                    image_path = None
                else:
                    image_path = image_path.strip('"').strip("'")
                
                print("\n⏳ 思考中...")
                response = await self.process_query(query, image_path)
                print("\n🤖 Assistant Response:\n" + response)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"\n❌ 发生严重错误: {str(e)}")


# 主程序入口
async def main(
        config_path: str = './config.json',
        models_config_path: str = './models.json',
        model_name: Optional[str] = 'gpt-5.1',
        provider: Optional[str] = 'openai',
        base_url: Optional[str] = 'https://yunwu.ai/v1',
        api_key: Optional[str] = 'sk-X9jXfVLVKHEK6y06p6MRJuEHwvqQX240PPrebQikc1fBXeIS',
        need_context: bool = True
    ):
    init_globel_params()
    with open(f"./prompts/agent.yaml", "r", encoding="utf-8") as file:
        SYSTEM_PROMPT = yaml.safe_load(file).get("system_prompt")
        print(SYSTEM_PROMPT)

    with open(config_path, 'r') as f:
        config = json.load(f)
        
    global_env = {}
    model_api_key = api_key
    model_base_url = base_url
    
    if os.path.exists(models_config_path):
        with open(models_config_path, 'r') as f:
            models_data = json.load(f)
            global_env = models_data.get('global_env', {})
            
            if models_data.get('models'):
                target_model = models_data['models'][0]
                model_api_key = target_model.get('apiKey', model_api_key)
                model_base_url = target_model.get('apiBase', model_base_url)

    servers_info = config['mcpServers']
    client = MCPClient(model_name=model_name, 
                    provider=provider,
                    servers_info=servers_info,
                    global_env=global_env,
                    temp_dir_path=temp_dir_path,
                    base_url=base_url, 
                    api_key=api_key, 
                    need_context=need_context,
                    system_prompt=SYSTEM_PROMPT)
    try:
        await client.initialize_sessions()
        await client.chat_loop()
    except Exception as e:
        if hasattr(e, 'error'):
            print(f"MCP Error Code: {e.error.code}")
            print(f"MCP Error Message: {e.error.message}")
            if hasattr(e.error, 'data'):
                print(f"MCP Error Data: {e.error.data}")
        raise e
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())