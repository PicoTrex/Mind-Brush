import os
import yaml
import argparse
import json
import base64
import mimetypes
import math
import io
from typing import List, Dict, Any, Optional
from PIL import Image
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

with open(f"./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if config.get("proxy_on", False):
    os.environ["http_proxy"] = config.get("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ["https_proxy"] = config.get("HTTPS_PROXY", "http://127.0.0.1:7890")

with open(f"./prompts/knowledge_reasoning.yaml", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = yaml.safe_load(file).get("system_prompt")

mcp = FastMCP("Knowledge Review")

# ----------------------------------------------------------------------
# Helper Functions (Image Processing)
# ----------------------------------------------------------------------

def encode_image_from_path(image_path: str):
    """Encodes a local image file to a base64 string."""
    if not image_path or not os.path.exists(image_path):
        return None
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/jpeg'
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def encode_pil_image(image: Image.Image):
    """Encodes a PIL Image object to a base64 string (JPEG)."""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding PIL image: {e}")
        return None

def merge_images_smart(image_paths: List[str], max_side: int = 2048) -> Image.Image:
    """
    智能网格拼接：将多张图片拼接成一个近似正方形的网格，
    保持每张图片的原始比例，不拉伸变形。
    """
    if not image_paths:
        raise ValueError("No images provided for merging.")
        
    images = []
    for path in image_paths:
        try:
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                images.append(img)
            else:
                print(f"Warning: Image path not found {path}, skipping.")
        except Exception as e:
            print(f"Warning: Could not open {path}, skipping. Error: {e}")

    if not images:
        raise ValueError("No valid images found to merge.")

    count = len(images)
    
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    cell_max_w = 768
    cell_max_h = 768
    
    grid_width = cols * cell_max_w
    grid_height = rows * cell_max_h
    
    combined_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255)) 

    for index, img in enumerate(images):
        row_idx = index // cols
        col_idx = index % cols
        
        x_offset = col_idx * cell_max_w
        y_offset = row_idx * cell_max_h
        
        img_aspect = img.width / img.height
        cell_aspect = cell_max_w / cell_max_h
        
        if img_aspect > cell_aspect:
            new_w = cell_max_w
            new_h = int(cell_max_w / img_aspect)
        else:
            new_h = cell_max_h
            new_w = int(cell_max_h * img_aspect)
            
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        paste_x = x_offset + (cell_max_w - new_w) // 2
        paste_y = y_offset + (cell_max_h - new_h) // 2
        
        combined_image.paste(resized_img, (paste_x, paste_y))

    if combined_image.width > max_side or combined_image.height > max_side:
        combined_image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    return combined_image

# ----------------------------------------------------------------------
# MCP Tool Definition
# ----------------------------------------------------------------------
@mcp.tool(description="Synthesize reasoning, inputs, and visual assets into a final image prompt.")
def knowledge_review(
    prompt: str,
    need_process_problem: List[str] = [],
    reasoning_knowledge: List[str] = [],
    downloaded_paths: List[str] = [],
    input_image_path: str = "" 
) -> Dict[str, Any]:
    '''
    Args:
        prompt (str): The core user instruction (The immutable intent).
        need_process_problem (List[str], optional): initial problem from intent analysis tools. Defaults to [].
        reasoning_knowledge (List[str], optional): Reasoning Result from Knowledge Reasoning tools. Defaults to [].
        downloaded_paths (List[str], optional): Style/vibe references from Image RAG. Defaults to [].
        input_image_path (str, optional): Visual ground truth. Defaults to "".
    Returns:
        Dict[str, Any]: The final instruction following the Structure Hierarchy.
    '''

    client = OpenAI(base_url=config.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"), api_key=config.get("OPENAI_API_KEY", "sk-X9jXfVLVKHEK6y06p6MRJuEHwvqQX240PPrebQikc1fBXeIS"))

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

    # 3. Attach User Input Image (Independent - Ground Truth)
    if input_image_path and os.path.exists(input_image_path):
        base64_user_img = encode_image_from_path(input_image_path)
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
            print(f"[KnowledgeReview] Merging {len(downloaded_paths)} references into grid...")
            merged_grid = merge_images_smart(downloaded_paths)
            base64_grid = encode_pil_image(merged_grid)
            
            if base64_grid:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_grid,
                        "detail": "auto" # Grid uses auto detail to save tokens
                    }
                })
                user_content.append({
                    "type": "text",
                    "text": "[System Note]: The image above is a merged grid of 'Reference Images'. Use these for style, texture, and atmosphere."
                })
        except Exception as e:
            print(f"[KnowledgeReview] Merge failed: {e}")
            user_content.append({
                "type": "text",
                "text": f"[System Note]: Failed to load reference images: {e}"
            })

    # 5. Construct Payload
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    # 6. Prepare Final Reference List (for downstream tool)
    # Combine User Image + Downloaded Images into one list
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
            temperature=0.3, # Low temperature for consistency
            max_tokens=1000
        )
        content = response.choices[0].message.content.strip()
        
        # 8. Clean JSON
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        result_json = json.loads(content)
        
        # 9. Force update reference list (Logic Safety)
        result_json["reference_image"] = final_refs
        
        return result_json

    except Exception as e:
        print(f"[KnowledgeReview] Error: {e}")
        
        # Fallback Logic
        fallback_prompt = f"Generate an image of {prompt}"
        # Inject reasoning results into fallback prompt if available
        if reasoning_knowledge:
            solutions = ", ".join([f"{v}" for v in reasoning_knowledge])
            fallback_prompt += f". Visual details: {solutions}"
            
        return {
            "final_prompt": fallback_prompt,
            "reference_image": final_refs
        }

if __name__ == "__main__":
    mcp.run()
    # prompt = "This is a brief, rhythmic poem by Gwendolyn Brooks about the defiant and doomed lives of a group of pool players who 'sing sin'. Based on this description, reason and generate the corresponding poem title and visualize the artistic conception."
    # need_process_problem = [
    #                             "What is the title of the brief, rhythmic poem by the poet Gwendolyn Brooks about defiant and doomed pool players who \"sing sin\"?",
    #                             "What is the visual artistic conception suggested by Gwendolyn Brooks's poem \"We Real Cool\"?"
    #                         ]
    # reasoning_knowledge = {"What is the title of the brief, rhythmic poem by the poet Gwendolyn Brooks about defiant and doomed pool players who \"sing sin\"?": "We Real Cool", "What is the visual artistic conception suggested by Gwendolyn Brooks's poem \"We Real Cool\"?": "An urban pool hall called the Golden Shovel late at night, dimly lit and slightly run‑down, with a small group of young Black men clustered around a green felt pool table beneath a harsh overhead lamp. Their loose stances, tilted caps, cigarettes, and casual slouches project swagger, defiance, and collective bravado, while heavy surrounding darkness, sharp shadows on the worn wooden floor, and curling cigarette smoke hint at danger and an ominous future. Outside grimy front windows, a yellowish neon Golden Shovel sign glows against a quiet city street, its reflections on wet pavement and empty sidewalks underscoring their isolation and transgressive, short‑lived sense of freedom, with details like distant police lights or a very late clock time suggesting that beyond the circle of light around the table lies inescapable doom."}
    # downloaded_paths = [
    #                             r"E:\github-project\Idea2Image\code\cases\9_Science-and-Logic_1\textgen.png",
    #                             r"E:\github-project\Idea2Image\code\cases\8_MathVerse_11\guided.png",
    #                         ]
    # print(knowledge_review(prompt=prompt, need_process_problem=need_process_problem, downloaded_paths=downloaded_paths))