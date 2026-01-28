import os
import yaml
import argparse
import json
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

with open(f"./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if config.get("proxy_on", False):
    os.environ["http_proxy"] = config.get("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ["https_proxy"] = config.get("HTTPS_PROXY", "http://127.0.0.1:7890")

with open(f"./prompts/keyword_generation.yaml", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = yaml.safe_load(file).get("system_prompt")

mcp = FastMCP("Keyword Generation")


@mcp.tool(description="Convert identified information gaps and reasoning results into optimized search keywords.")
def keyword_generation(
    need_process_problem: List[str],
    reasoning_knowledge: List[str] = []
) -> Dict[str, List[str]]:
    """
    Args:
        need_process_problem: A list of strings, where each string is a specific question or problem.
        reasoning_knowledge: (Optional) A list containing prior reasoning results (Problem -> Answer).
    """
    
    # 1. 快速检查：如果输入列表为空，直接返回空结果
    if not need_process_problem:
        return {
            "text_queries": [],
            "image_queries": []
        }

    client = OpenAI(base_url=config.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"), api_key=config.get("OPENAI_API_KEY", "sk-X9jXfVLVKHEK6y06p6MRJuEHwvqQX240PPrebQikc1fBXeIS"))

    # 2. 构建符合新提示词逻辑的输入上下文
    input_data = {
        "need_process_problem": need_process_problem
    }
    
    # 如果有推理知识，也加入到输入中
    if reasoning_knowledge:
        input_data["reasoning_knowledge"] = reasoning_knowledge

    input_payload = json.dumps(input_data, indent=2, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_payload}
    ]

    try:
        response = client.chat.completions.create(
            model=config.get("OPENAI_MODEL_NAME", "gpt-5.1"),
            messages=messages,
            temperature=0.0 # 保持 0.0 以确保逻辑执行的确定性
        )
        content = response.choices[0].message.content.strip()
        
        # 3. 清理 Markdown 格式
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        return json.loads(content)

    except Exception as e:
        # 4. Fallback 逻辑
        print(f"Error in keyword generation: {e}")
        return {
            "text_queries": [], 
            "image_queries": [],
            "error": str(e)
        }

if __name__ == "__main__":
    mcp.run()
    # user_request = "Generate an image of Rumi, Mira, and Zoey, members of HUNTR/X from 'Kpop Demon Hunters', performing while standing on the moon."
    # need_process_problem = ["Who are the characters Rumi, Mira, and Zoey from 'Kpop Demon Hunters' and what are their canonical visual appearances?", "What is the official or widely accepted visual design and style of the group HUNTR/X in 'Kpop Demon Hunters'?", "Are there any copyright or usage restrictions associated with generating images of characters from 'Kpop Demon Hunters'?", 'What does the surface and background of the moon look like in a realistic depiction for use as the performance stage?']
    # print(keyword_generation(need_process_problem=need_process_problem))