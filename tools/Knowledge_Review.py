"""
Knowledge Review Tool
=====================
MCP tool for synthesizing reasoning, inputs, and visual assets
into a final optimized image generation prompt.
"""

import os
import sys

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List, Dict, Any

from PIL import Image
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Import shared utilities
from tools.base import (
    encode_image,
    merge_images_smart,
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

SYSTEM_PROMPT = load_prompt("knowledge_reasoning")

mcp = FastMCP("Knowledge Review")


# ==============================================================================
# MCP Tool Definition
# ==============================================================================

@mcp.tool(description="Synthesize reasoning, inputs, and visual assets into a final image prompt.")
def knowledge_review(
    prompt: str,
    need_process_problem: List[str] = [],
    reasoning_knowledge: List[str] = [],
    downloaded_paths: List[str] = [],
    input_image_path: str = ""
) -> Dict[str, Any]:
    """
    Review and synthesize all inputs into an optimized prompt for image generation.
    
    Args:
        prompt: The core user instruction (immutable intent).
        need_process_problem: Initial problems from intent analysis.
        reasoning_knowledge: Reasoning results from Knowledge Reasoning tool.
        downloaded_paths: Style/vibe reference images from Image RAG.
        input_image_path: Visual ground truth (user's input image).
        
    Returns:
        Dict containing:
            - final_prompt: Optimized prompt following structure hierarchy
            - reference_image: List of image paths for downstream processing
    """
    client = OpenAI(
        base_url=config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=config.get("OPENAI_API_KEY", ""),
    )

    # 1. Prepare Text Context (Inventory)
    text_context = f"""
    === 📋 Input Inventory ===
    1. Primary Prompt: "{prompt}"
    
    === 🧩 Reasoning & Logic ===
    - Identified Problems: {json.dumps(need_process_problem, indent=2)}
    - Reasoning Results: {json.dumps(reasoning_knowledge, indent=2)}
    
    === 🖼️ Visual Assets Inventory ===
    - User Input Image: {"Present" if input_image_path else "None"}
    - Reference Images: {len(downloaded_paths)} images (Provided as a merged grid below)
    """

    # 2. Build Multimodal Messages
    user_content = [{"type": "text", "text": text_context}]

    # 3. Attach User Input Image (Ground Truth)
    if input_image_path and os.path.exists(input_image_path):
        base64_user_img = encode_image(input_image_path)
        if base64_user_img:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": base64_user_img,
                    "detail": "high"
                }
            })
            user_content.append({
                "type": "text",
                "text": "[System Note]: The image above is the 'User Input Image'. Use it as the primary structural ground truth."
            })

    # 4. Attach Downloaded References (Merged Grid)
    if downloaded_paths:
        try:
            _log_info(f"Merging {len(downloaded_paths)} references into grid...")
            merged_grid = merge_images_smart(downloaded_paths)
            base64_grid = encode_image(merged_grid)  # PIL Image supported
            
            if base64_grid:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_grid,
                        "detail": "auto"  # Auto for grid to save tokens
                    }
                })
                user_content.append({
                    "type": "text",
                    "text": "[System Note]: The image above is a merged grid of 'Reference Images'. Use these for style, texture, and atmosphere."
                })
        except Exception as e:
            _log_error(f"Merge failed: {e}")
            user_content.append({
                "type": "text",
                "text": f"[System Note]: Failed to load reference images: {e}"
            })

    # 5. Construct Payload
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    # 6. Prepare Final Reference List
    final_refs = []
    if input_image_path and os.path.exists(input_image_path):
        final_refs.append(input_image_path)
    if downloaded_paths:
        final_refs.extend(downloaded_paths)

    try:
        # 7. Call LLM
        response = client.chat.completions.create(
            model=config.get("OPENAI_MODEL_NAME", "gpt-5.1"),
            messages=messages,
            temperature=0.3,  # Low temperature for consistency
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        result_json = parse_json_response(content)
        
        # 8. Force update reference list (Logic Safety)
        result_json["reference_image"] = final_refs
        
        return result_json

    except Exception as e:
        _log_error(f"Knowledge Review Error: {e}")
        
        # Fallback Logic
        fallback_prompt = f"Generate an image of {prompt}"
        if reasoning_knowledge:
            solutions = ", ".join([f"{v}" for v in reasoning_knowledge])
            fallback_prompt += f". Visual details: {solutions}"
        
        return {
            "final_prompt": fallback_prompt,
            "reference_image": final_refs
        }


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
    
    # ==============================================================================
    # Test Cases (Uncomment to test locally)
    # ==============================================================================
    # prompt = "This is a brief, rhythmic poem by Gwendolyn Brooks about the defiant and doomed lives of a group of pool players who 'sing sin'. Based on this description, reason and generate the corresponding poem title and visualize the artistic conception."
    # need_process_problem = [
    #     "What is the title of the brief, rhythmic poem by the poet Gwendolyn Brooks about defiant and doomed pool players who \"sing sin\"?",
    #     "What is the visual artistic conception suggested by Gwendolyn Brooks's poem \"We Real Cool\"?"
    # ]
    # reasoning_knowledge = {
    #     "What is the title of the brief, rhythmic poem by the poet Gwendolyn Brooks about defiant and doomed pool players who \"sing sin\"?": "We Real Cool",
    #     "What is the visual artistic conception suggested by Gwendolyn Brooks's poem \"We Real Cool\"?": "An urban pool hall called the Golden Shovel late at night..."
    # }
    # downloaded_paths = [
    #     r"E:\github-project\Idea2Image\code\cases\9_Science-and-Logic_1\textgen.png",
    #     r"E:\github-project\Idea2Image\code\cases\8_MathVerse_11\guided.png",
    # ]
    # print(knowledge_review(
    #     prompt=prompt,
    #     need_process_problem=need_process_problem,
    #     downloaded_paths=downloaded_paths
    # ))