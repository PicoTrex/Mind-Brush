"""
Knowledge Reasoning Tool
========================
MCP tool for performing deep reasoning or knowledge synthesis
to answer specific problems identified during intent analysis.
"""

import os
import sys

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Import shared utilities
from tools.base import (
    encode_image,
    parse_json_response,
    load_config,
    load_prompt,
    setup_proxy_from_config,
    setup_stdio_encoding,
    _log_info,
)

# ==============================================================================
# Configuration
# ==============================================================================

setup_stdio_encoding()
config = load_config("./config.yaml")
setup_proxy_from_config(config)

SYSTEM_PROMPT = load_prompt("knowledge_reasoning")

mcp = FastMCP("Knowledge Reasoning Engine")


# ==============================================================================
# MCP Tool Definition
# ==============================================================================

@mcp.tool(description="Perform deep reasoning or knowledge synthesis to answer specific problems.")
def knowledge_reasoning(
    user_intent: str,
    need_process_problem: List[str],
    intent_category: str,
    user_image_path: Optional[str] = None,
    downloaded_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform reasoning on user intent with visual and textual context.
    
    Args:
        user_intent: The original user prompt (fact-enriched).
        need_process_problem: List of specific questions to answer.
        intent_category: Name of generation type determining workflow.
        user_image_path: Path to the user's uploaded image (if any).
        downloaded_paths: List of paths to downloaded reference images.
        
    Returns:
        Dict containing:
            - reasoning_knowledge: List of reasoning results
    """
    client = OpenAI(
        base_url=config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=config.get("OPENAI_API_KEY", ""),
    )

    # Build text context
    text_context = f"""
    === Task Configuration ===
    Intent Category: {intent_category}
    
    === User Input (Fact-Enriched) ===
    Request: {user_intent}
    
    === Problems to Solve (Target Output) ===
    {json.dumps(need_process_problem, indent=2)}
    """

    # Build multimodal message content
    user_content = [{"type": "text", "text": text_context}]

    # Add user's primary image
    if user_image_path and os.path.exists(user_image_path):
        base64_img = encode_image(user_image_path)
        if base64_img:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": base64_img,
                    "detail": "high"  # High resolution for reasoning tasks
                }
            })
            user_content.append({
                "type": "text",
                "text": "[System Note: The image above is the User Input Image.]"
            })

    # Add reference images (for Search_Reasoning mode)
    if intent_category == "Search_Reasoning_Generation" and downloaded_paths:
        valid_count = 0
        max_ref_images = 5  # Limit to prevent token explosion
        
        for img_path in downloaded_paths:
            if valid_count >= max_ref_images:
                break
                
            if img_path and os.path.exists(img_path):
                base64_ref = encode_image(img_path)
                if base64_ref:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_ref}
                    })
                    valid_count += 1
        
        if valid_count > 0:
            user_content.append({
                "type": "text",
                "text": "[System Note: The images above are Retrieved Reference Images from search.]"
            })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    try:
        response = client.chat.completions.create(
            model=config.get("OPENAI_MODEL_NAME", "gpt-5.1"),
            messages=messages,
            temperature=0.0,  # Low temperature for accurate reasoning
        )
        
        content = response.choices[0].message.content.strip()
        return parse_json_response(content)

    except Exception as e:
        return {
            "reasoning_knowledge": [],
            "error": f"Reasoning failed: {str(e)}"
        }


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
    
    # ==============================================================================
    # Test Cases (Uncomment to test locally)
    # ==============================================================================
    # user_intent = "Generate a view of the Tower Bridge at coordinates 51.5055° N, 0.0754° W."
    # need_process_problem = ["Confirm the visual appearance of the location mentioned in the request."]
    # intent_category = "Search_Reasoning_Generation"
    # print(knowledge_reasoning(
    #     user_intent=user_intent,
    #     need_process_problem=need_process_problem,
    #     intent_category=intent_category
    # ))