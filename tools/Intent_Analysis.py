import os
import yaml
import base64
import json
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

with open(f"./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if config.get("proxy_on", False):
    os.environ["http_proxy"] = config.get("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ["https_proxy"] = config.get("HTTPS_PROXY", "http://127.0.0.1:7890")

with open(f"./prompts/intent_analysis.yaml", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = yaml.safe_load(file).get("system_prompt")

mcp = FastMCP("Intent Analyzer")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@mcp.tool(description="Analyze user intent from text and optional image input.")
def intent_analyzer(
    user_intent: str, 
    user_image_path: Optional[str] = None
    ) -> Dict[str, Any]:
    """
    Args:
        user_intent: The text input from the user describing their intent.
        user_image_path: Optional path to an image file provided by the user.
    """
    client = OpenAI(base_url=config.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"), api_key=config.get("OPENAI_API_KEY", "sk-X9jXfVLVKHEK6y06p6MRJuEHwvqQX240PPrebQikc1fBXeIS"))
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    user_content = []
    
    if user_intent:
        user_content.append({"type": "text", "text": user_intent})
    
    if user_image_path and os.path.exists(user_image_path):
        try:
            base64_image = encode_image(user_image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}
            
    messages.append({"role": "user", "content": user_content})
    try:
        response = client.chat.completions.create(
            model=config.get("OPENAI_MODEL_NAME", "gpt-5.1"),
            messages=messages,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        # 清理可能存在的 markdown 代码块标记
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        return json.loads(content)
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run()
    # print(intent_analyzer(user_intent="Generate an image of the process where potassium permanganate granules fall into water and dissolve.", user_image_path=r"E:\github-project\Idea2Image\code\cases\9_Science-and-Logic_1\textgen.png"))