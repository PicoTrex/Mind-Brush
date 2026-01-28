import os
import yaml
import argparse
import json
import requests
from typing import List, Dict, Any
from pathlib import Path
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

with open(f"./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if config.get("proxy_on", False):
    os.environ["http_proxy"] = config.get("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ["https_proxy"] = config.get("HTTPS_PROXY", "http://127.0.0.1:7890")

with open(f"./prompts/knowledge_reasoning.yaml", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = yaml.safe_load(file).get("system_prompt")

# 3. 路径与客户端初始化
TEMP_DIR = Path(config.get("temp_dir", "./temp").get("text_rag", "./temp/text_rag")).absolute()
TEMP_DIR.mkdir(parents=True, exist_ok=True)
SERPER_API_KEY = config.get("SERPER_API_KEY", "bdf9b167a18e2e9071e4eed39f257aa28d8ad10c")

mcp = FastMCP("Knowledge Enhanced Search")

# ================= 模块 1: 文本搜索与爬取 (原 Text_RAG Logic) =================

def fetch_serper_text_urls(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """获取搜索结果 URL"""
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": max(num_results, 5)}) # 多抓一点备用
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data.get("organic", [])
        return [{"title": item.get("title", ""), "link": item.get("link", "")} for item in items[:num_results]]
    except Exception as e:
        print(f"Error fetching Serper results: {e}")
        return []

def scrape_webpage_content(url: str) -> str:
    """爬取并清洗网页正文"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for script in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "svg", "button"]):
            script.extract()
            
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # 截断以节省 Token
        return clean_text[:3000] + "..." if len(clean_text) > 3000 else clean_text
    except Exception as e:
        return f"[Error scraping content: {str(e)}]"

def run_search_workflow(text_queries: List[str]) -> str:
    """执行完整的搜索+爬取流程，返回合并后的文本报告"""
    final_report = []
    max_urls_per_query = 2 

    print(f"🔍 [Integrated Tool] Executing Text Search for: {text_queries}")

    for query in text_queries:
        search_results = fetch_serper_text_urls(query, num_results=max_urls_per_query)
        query_section = f"\n====== Search Query: {query} ======\n"
        
        if not search_results:
            query_section += "No results found.\n"
        else:
            for idx, res in enumerate(search_results, 1):
                url = res['link']
                content = scrape_webpage_content(url)
                query_section += f"\n--- Result {idx}: {res['title']} ---\nSource: {url}\nContent:\n{content}\n" + "-"*20 + "\n"
        
        final_report.append(query_section)

    full_knowledge = "\n".join(final_report)
    
    # 保存原始搜索日志 (可选，方便调试)
    try:
        with open(TEMP_DIR / "search_raw_log.txt", "w", encoding="utf-8") as f:
            f.write(full_knowledge)
    except: pass
    
    return full_knowledge

def run_injection_logic(user_intent: str, retrieved_knowledge: str, image_queries: List[str]) -> Dict[str, Any]:
    """调用 LLM 进行知识注入"""
    client = OpenAI(base_url=config.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"), api_key=config.get("OPENAI_API_KEY", "sk-X9jXfVLVKHEK6y06p6MRJuEHwvqQX240PPrebQikc1fBXeIS"))
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""
            Original Prompt: {user_intent}
            Retrieved Knowledge: {retrieved_knowledge}
            Image Queries: {json.dumps(image_queries)}
            """}
    ]
    
    try:
        response = client.chat.completions.create(
            model=config.get("OPENAI_MODEL_NAME", "gpt-5.1"),
            messages=messages,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        
        # 清理 Markdown
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        return json.loads(content)
    except Exception as e:
        print(f"❌ Injection Failed: {e}")
        # Fallback: 发生错误时返回原始值
        return {
            "prompt": user_intent,
            "final_image_queries": image_queries,
            "error": str(e)
        }

# ================= MCP Tool 定义 (合并入口) =================

@mcp.tool(description="Performs text search, scrapes content, and immediately uses that knowledge to refine the prompt and optimize image queries.")
def text_search_and_knowledge_injection(
    text_queries: List[str], 
    user_intent: str, 
    image_queries: List[str] = []
) -> str:
    """
    Integration Tool: Text Search -> Knowledge Injection.
    
    Args:
        text_queries: List of keywords for Google Search.
        user_intent: The user's original request.
        image_queries: The draft image queries identified by Intent Analysis.
        
    Returns:
        JSON String containing:
        {
            "prompt": "Refined prompt with facts",
            "final_image_queries": ["Optimized list"]
        }
    """
    
    # 1. 执行搜索与爬取 (The "Eyes")
    if text_queries:
        retrieved_knowledge = run_search_workflow(text_queries)
    else:
        retrieved_knowledge = "No text queries provided. Proceeding with optimization only."
        return json.dumps({
            "prompt": user_intent,
            "final_image_queries": image_queries,
        }, ensure_ascii=False)

    # 2. 执行思考与重写 (The "Brain")
    result_json = run_injection_logic(user_intent, retrieved_knowledge, image_queries)
    
    # 3. 返回 JSON 字符串 (供 Client 直接解析)
    return json.dumps(result_json, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()
    # Debug:
    # res = text_search_and_knowledge_injection(
    #     text_queries=["2025 NBA Finals teams", "2025 NBA Finals Game 7 score"], 
    #     user_intent="Scoreboard of 2025 NBA Finals Game 7", 
    #     image_queries=["NBA Finals Scoreboard design"]
    # )
    # print(res)