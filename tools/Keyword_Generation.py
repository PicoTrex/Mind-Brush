"""
Keyword Generation Tool
=======================
MCP tool for converting identified information gaps and reasoning results
into optimized search keywords for text and image retrieval.
"""

import os
import sys

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, List

from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Import shared utilities
from tools.base import (
    parse_json_response,
    load_config,
    load_prompt,
    setup_proxy_from_config,
    setup_stdio_encoding,
)

# ==============================================================================
# Configuration
# ==============================================================================

setup_stdio_encoding()
config = load_config("./config.yaml")
setup_proxy_from_config(config)

SYSTEM_PROMPT = load_prompt("keyword_generation")

mcp = FastMCP("Keyword Generation")


# ==============================================================================
# MCP Tool Definition
# ==============================================================================

@mcp.tool(description="Convert identified information gaps and reasoning results into optimized search keywords.")
def keyword_generation(
    need_process_problem: List[str],
    reasoning_knowledge: List[str] = []
) -> Dict[str, List[str]]:
    """
    Generate optimized search keywords from problems and reasoning results.
    
    Args:
        need_process_problem: A list of strings, each a specific question or problem.
        reasoning_knowledge: Optional list containing prior reasoning results.
        
    Returns:
        Dict containing:
            - text_queries: Keywords for text/web search
            - image_queries: Keywords for image search
    """
    # Quick check: empty input returns empty results
    if not need_process_problem:
        return {
            "text_queries": [],
            "image_queries": []
        }

    client = OpenAI(
        base_url=config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=config.get("OPENAI_API_KEY", ""),
    )

    # Build input payload
    input_data = {"need_process_problem": need_process_problem}
    
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
            temperature=0.0,
        )
        
        content = response.choices[0].message.content.strip()
        return parse_json_response(content)

    except Exception as e:
        # Fallback with error info
        return {
            "text_queries": [],
            "image_queries": [],
            "error": str(e)
        }


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
    
    # ==============================================================================
    # Test Cases (Uncomment to test locally)
    # ==============================================================================
    # user_request = "Generate an image of Rumi, Mira, and Zoey, members of HUNTR/X from 'Kpop Demon Hunters', performing while standing on the moon."
    # need_process_problem = [
    #     "Who are the characters Rumi, Mira, and Zoey from 'Kpop Demon Hunters' and what are their canonical visual appearances?",
    #     "What is the official or widely accepted visual design and style of the group HUNTR/X in 'Kpop Demon Hunters'?",
    #     "Are there any copyright or usage restrictions associated with generating images of characters from 'Kpop Demon Hunters'?",
    #     "What does the surface and background of the moon look like in a realistic depiction for use as the performance stage?"
    # ]
    # print(keyword_generation(need_process_problem=need_process_problem))