"""
Intent Analysis Tool
====================
MCP tool for analyzing user intent from text and optional image input.
Determines the processing strategy for downstream tools.
"""

import os
import sys

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional

from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Import shared utilities (works both as module and standalone)
from tools.base import (
    encode_image,
    parse_json_response,
    load_config,
    load_prompt,
    setup_proxy_from_config,
    setup_stdio_encoding,
)

# ==============================================================================
# Configuration
# ==============================================================================

# Setup encoding for MCP stdio communication
setup_stdio_encoding()

# Load configuration
config = load_config("./config.yaml")
setup_proxy_from_config(config)

# Load system prompt
SYSTEM_PROMPT = load_prompt("intent_analysis")

# Initialize MCP server
mcp = FastMCP("Intent Analyzer")


# ==============================================================================
# MCP Tool Definition
# ==============================================================================

@mcp.tool(description="Analyze user intent from text and optional image input.")
def intent_analyzer(
    user_intent: str,
    user_image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze user intent and determine processing strategy.
    
    Args:
        user_intent: The text input from the user describing their intent.
        user_image_path: Optional path to an image file provided by the user.
        
    Returns:
        Dict containing:
            - need_process_problem: List of specific problems requiring processing
            - intent_category: Category determining downstream workflow:
                - Direct_Generation
                - Reasoning_Generation
                - Search_Generation
                - Reasoning_Search_Generation
                - Search_Reasoning_Generation
    """
    client = OpenAI(
        base_url=config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=config.get("OPENAI_API_KEY", ""),
    )
    
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_content = []
    
    # Add text content
    if user_intent:
        user_content.append({"type": "text", "text": user_intent})
    
    # Add image content if provided
    if user_image_path and os.path.exists(user_image_path):
        try:
            base64_image = encode_image(user_image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": base64_image}
            })
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}
    
    messages.append({"role": "user", "content": user_content})
    
    try:
        response = client.chat.completions.create(
            model=config.get("OPENAI_MODEL_NAME", "gpt-5.1"),
            messages=messages,
            temperature=0.0,
        )
        
        content = response.choices[0].message.content.strip()
        return parse_json_response(content)
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
    
    # ==============================================================================
    # Test Cases (Uncomment to test locally)
    # ==============================================================================
    # print(intent_analyzer(
    #     user_intent="Generate an image of the process where potassium permanganate granules fall into water and dissolve.",
    #     user_image_path=r"E:\github-project\Idea2Image\code\cases\9_Science-and-Logic_1\textgen.png"
    # ))