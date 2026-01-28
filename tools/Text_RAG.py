"""
Text RAG Tool
=============
MCP tool for performing text search, scraping content,
and using knowledge to refine prompts and optimize image queries.
"""

import os
import sys

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Import shared utilities
from tools.base import (
    parse_json_response,
    load_config,
    load_prompt,
    setup_proxy_from_config,
    setup_stdio_encoding,
    _log_info,
    _log_error,
)

# ==============================================================================
# Configuration
# ==============================================================================

setup_stdio_encoding()
config = load_config("./config.yaml")
setup_proxy_from_config(config)

SYSTEM_PROMPT = load_prompt("text_rag_injection")

# Get temp directory - use session-specific if available
session_dir = os.environ.get("MINDBRUSH_SESSION_DIR")
if session_dir:
    TEMP_DIR = Path(session_dir) / "temp" / "text_rag"
else:
    temp_dir_config = config.get("temp_dir", {})
    if isinstance(temp_dir_config, dict):
        TEMP_DIR = Path(temp_dir_config.get("text_rag", "./temp/text_rag")).absolute()
    else:
        TEMP_DIR = Path("./temp/text_rag").absolute()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Serper API configuration
SERPER_API_KEY = config.get("SERPER_API_KEY", "")

# Initialize MCP server
mcp = FastMCP("Knowledge Enhanced Search")


# ==============================================================================
# Text Search Functions
# ==============================================================================

def fetch_serper_text_urls(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Fetch search result URLs using Serper API.
    
    Args:
        query: Search query string
        num_results: Number of results to retrieve
        
    Returns:
        List of dicts with 'title' and 'link' keys
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": max(num_results, 5)})
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("organic", [])
        return [
            {"title": item.get("title", ""), "link": item.get("link", "")}
            for item in items[:num_results]
        ]
        
    except Exception as e:
        _log_error(f"Error fetching Serper results: {e}")
        return []


def scrape_webpage_content(url: str, max_length: int = 3000) -> str:
    """
    Scrape and clean webpage content.
    
    Args:
        url: Webpage URL to scrape
        max_length: Maximum content length to return
        
    Returns:
        Cleaned text content
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", 
                            "iframe", "noscript", "svg", "button"]):
            element.extract()
        
        # Extract and clean text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        # Truncate to save tokens
        if len(clean_text) > max_length:
            return clean_text[:max_length] + "..."
        return clean_text
        
    except Exception as e:
        return f"[Error scraping content: {str(e)}]"


def run_search_workflow(text_queries: List[str]) -> str:
    """
    Execute complete search and scrape workflow.
    
    Args:
        text_queries: List of search queries
        
    Returns:
        Merged text report from all searches
    """
    final_report = []
    max_urls_per_query = 2

    _log_info(f"Executing Text Search for: {text_queries}")

    for query in text_queries:
        search_results = fetch_serper_text_urls(query, num_results=max_urls_per_query)
        query_section = f"\n====== Search Query: {query} ======\n"
        
        if not search_results:
            query_section += "No results found.\n"
        else:
            for idx, res in enumerate(search_results, 1):
                url = res["link"]
                content = scrape_webpage_content(url)
                query_section += f"\n--- Result {idx}: {res['title']} ---\n"
                query_section += f"Source: {url}\n"
                query_section += f"Content:\n{content}\n"
                query_section += "-" * 20 + "\n"
        
        final_report.append(query_section)

    full_knowledge = "\n".join(final_report)
    
    # Save raw log for debugging
    try:
        log_file = TEMP_DIR / "search_raw_log.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(full_knowledge)
    except Exception:
        pass
    
    return full_knowledge


def run_injection_logic(
    user_intent: str,
    retrieved_knowledge: str,
    image_queries: List[str]
) -> Dict[str, Any]:
    """
    Use LLM to inject knowledge into prompt.
    
    Args:
        user_intent: Original user request
        retrieved_knowledge: Scraped knowledge text
        image_queries: Draft image queries
        
    Returns:
        Dict with refined prompt and final image queries
    """
    client = OpenAI(
        base_url=config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=config.get("OPENAI_API_KEY", ""),
    )
    
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
            model=config.get("OPENAI_MODEL_NAME", "gpt-4"),
            messages=messages,
            temperature=0.0,
        )
        
        content = response.choices[0].message.content.strip()
        return parse_json_response(content)
        
    except Exception as e:
        _log_error(f"Injection Failed: {e}")
        return {
            "prompt": user_intent,
            "final_image_queries": image_queries,
            "error": str(e)
        }


# ==============================================================================
# MCP Tool Definition
# ==============================================================================

@mcp.tool(description="Performs text search, scrapes content, and uses knowledge to refine prompt and image queries.")
def text_search_and_knowledge_injection(
    text_queries: List[str],
    user_intent: str,
    image_queries: List[str] = []
) -> str:
    """
    Integration tool: Text Search -> Knowledge Injection.
    
    Args:
        text_queries: List of keywords for Google Search.
        user_intent: The user's original request.
        image_queries: Draft image queries from Intent Analysis.
        
    Returns:
        JSON string containing:
            - prompt: Refined prompt with facts
            - final_image_queries: Optimized list
    """
    # Execute search and scrape
    if text_queries:
        retrieved_knowledge = run_search_workflow(text_queries)
    else:
        # No queries, return original intent
        return json.dumps({
            "prompt": user_intent,
            "final_image_queries": image_queries,
        }, ensure_ascii=False)

    # Execute LLM injection
    result_json = run_injection_logic(user_intent, retrieved_knowledge, image_queries)
    
    return json.dumps(result_json, ensure_ascii=False)


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()

    # ==============================================================================
    # Test Cases (Uncomment to test locally)
    # ==============================================================================
    # text_queries = [
    #     "Tower Bridge London architecture history",
    #     "Tower Bridge at night illumination"
    # ]
    # user_intent = "Generate a view of the Tower Bridge at coordinates 51.5055° N, 0.0754° W."
    # image_queries = ["Tower Bridge London night view", "Tower Bridge architectural details"]
    # 
    # result = text_search_and_knowledge_injection(
    #     text_queries=text_queries,
    #     user_intent=user_intent,
    #     image_queries=image_queries
    # )
    # print("RAG Result:", result)