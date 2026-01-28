import os
import yaml
import argparse
import base64
import json
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
import mimetypes

with open(f"./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if config.get("proxy_on", False):
    os.environ["http_proxy"] = config.get("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ["https_proxy"] = config.get("HTTPS_PROXY", "http://127.0.0.1:7890")

with open(f"./prompts/knowledge_reasoning.yaml", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = yaml.safe_load(file).get("system_prompt")

mcp = FastMCP("Knowledge Reasoning Engine")

def encode_image(image_path: str):
    if not image_path or not os.path.exists(image_path):
        return None
    
    # --- 修改开始: 增强格式识别 ---
    file_ext = os.path.splitext(image_path)[1].lower()
    
    # 优先根据后缀判断，确保 webp/png/gif 被正确识别
    known_mimes = {
        '.webp': 'image/webp',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff'
    }
    
    if file_ext in known_mimes:
        mime_type = known_mimes[file_ext]
    else:
        mime_type, _ = mimetypes.guess_type(image_path)
    
    if mime_type is None:
        mime_type = 'image/jpeg' # 默认兜底
    # --- 修改结束 ---
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

@mcp.tool(description="Perform deep reasoning or knowledge synthesis to answer specific problems.")
def knowledge_reasoning(
    user_intent: str,
    need_process_problem: List[str],
    intent_category: str,
    user_image_path: Optional[str] = None,
    downloaded_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Args:
        user_intent: The original user prompt (Fact-Enriched).
        need_process_problem: List of specific questions to answer.
        intent_category: name of generation type,
        user_image_path: Path to the user's uploaded image (if any).
        downloaded_paths: List of paths to downloaded reference images (if Search_Reasoning).
    output:
        reasoning_knowledge: A JSON object containing the reasoning results.
    """

    client = OpenAI(base_url=config.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"), api_key=config.get("OPENAI_API_KEY", "sk-X9jXfVLVKHEK6y06p6MRJuEHwvqQX240PPrebQikc1fBXeIS"))

    # 1. 构建 Prompt 上下文
    # 我们通过 Text Content 清晰地标注不同来源的信息
    text_context = f"""
    === Task Configuration ===
    Intent Category: {intent_category}
    
    === User Input (Fact-Enriched) ===
    Request: {user_intent}
    
    === Problems to Solve (Target Output) ===
    {json.dumps(need_process_problem, indent=2)}
    """
    
    # 注意：之前的 retrieved_knowledge 拼接逻辑已被移除，
    # 因为事实信息现在假设已经包含在 user_intent 中。

    # 2. 构建多模态消息体
    user_content = [{"type": "text", "text": text_context}]

    # A. 添加用户原始图像 (Primary Image)
    if user_image_path:
        base64_img = encode_image(user_image_path)
        if base64_img:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": base64_img,
                    "detail": "high" # 确保推理任务使用高分辨率
                }
            })
            # 添加标注告诉模型这是用户的图
            user_content.append({"type": "text", "text": "[System Note: The image above is the User Input Image.]"})
            
    # print(user_content)

    # B. 添加搜索结果图像 (Reference Images)
    # 仅在 Search_Reasoning 模式且有路径时添加
    if intent_category == "Search_Reasoning_Generation" and downloaded_paths:
        valid_search_images = 0
        for idx, img_path in enumerate(downloaded_paths):
            # 限制参考图数量，防止 Token 爆炸 (例如最多取前3-5张)
            if valid_search_images >= 5: 
                break
                
            base64_ref = encode_image(img_path)
            if base64_ref:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_ref}
                })
                valid_search_images += 1
        
        if valid_search_images > 0:
            user_content.append({"type": "text", "text": "[System Note: The images above are Retrieved Reference Images from search.]"})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    try:
        response = client.chat.completions.create(
            model=config.get("OPENAI_MODEL_NAME", "gpt-5.1"),
            messages=messages,
            temperature=0.0, # 推理任务需要较低的温度以保证准确性
        )
        content = response.choices[0].message.content.strip()
        
        # 清理 Markdown
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        return json.loads(content)

    except Exception as e:
        # 错误处理：返回空字典或包含错误信息的字典
        return {
            "reasoning_knowledge": [],
            "error": f"Reasoning failed: {str(e)}"
        }

if __name__ == "__main__":
    mcp.run()
    # 示例调用 (注释掉)
    # user_intent = "Generate a view of the Tower Bridge at coordinates 51.5055° N, 0.0754° W."
    # need_process_problem = ["Confirm the visual appearance of the location mentioned in the request."]
    # intent_category = "Search_Reasoning_Generation"
    # print(knowledge_reasoning(user_intent=user_intent, need_process_problem=need_process_problem, intent_category=intent_category))