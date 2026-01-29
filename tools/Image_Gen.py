"""
Image Generation Tool
=====================
MCP tool for generating images based on text prompts
and optional reference images using image models.
"""

import os
import sys

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import base64
from pathlib import Path
from typing import List

import requests
from PIL import Image
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Import shared utilities
from tools.base import (
    merge_images_smart,
    load_config,
    setup_proxy_from_config,
    setup_stdio_encoding,
    _log_info,
    _log_error,
    _log_success,
)

# ==============================================================================
# Configuration
# ==============================================================================

setup_stdio_encoding()
config = load_config("./config.yaml")
setup_proxy_from_config(config)

# Get temp directory - use session-specific if available
session_dir = os.environ.get("MINDBRUSH_SESSION_DIR")
if session_dir:
    TEMP_DIR = Path(session_dir) / "temp" / "image_gen"
else:
    temp_dir_config = config.get("temp_dir", {})
    if isinstance(temp_dir_config, dict):
        TEMP_DIR = Path(temp_dir_config.get("image_gen", "./temp/image_gen")).absolute()
    else:
        TEMP_DIR = Path("./temp/image_gen").absolute()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize MCP server
mcp = FastMCP(
    name="Unified Image Generator",
    instructions="This MCP provides both text-to-image and image-guided generation using image models."
)


# ==============================================================================
# Helper Functions
# ==============================================================================

def _download_image_from_url(url: str, file_path: str) -> bool:
    """
    Download an image from URL and save to file.
    
    Args:
        url: Image URL to download
        file_path: Local path to save the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        _log_error(f"Failed to download image from URL: {e}")
        return False


def _save_base64_image(b64_data: str, file_path: str) -> bool:
    """
    Decode and save a base64 image.
    
    Args:
        b64_data: Base64 encoded image data
        file_path: Local path to save the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        img_bytes = base64.b64decode(b64_data)
        with open(file_path, "wb") as f:
            f.write(img_bytes)
        return True
    except Exception as e:
        _log_error(f"Failed to decode Base64: {e}")
        return False


# ==============================================================================
# MCP Tool Definition
# ==============================================================================

@mcp.tool(
    name="unified_image_generator",
    description="Generate image(s) based on prompt only or prompt + multiple reference images."
)
def unified_image_generator(
    prompt: str,
    reference_images: List[str] = []
) -> List[str]:
    """
    Generate images using text prompt with optional reference images.
    
    Args:
        prompt: Text description of the image to generate.
        reference_images: Optional list of reference image paths for guided generation.
        
    Returns:
        List of paths to generated images, or error messages.
    """
    size = "1024x1024"
    saved_paths: List[str] = []
    
    client = OpenAI(
        base_url=config.get("IMAGE_BASE_URL", "https://yunwu.ai/v1"),
        api_key=config.get("IMAGE_API_KEY", ""),
    )

    # ------------------------
    # Image-Guided Generation
    # ------------------------
    if reference_images:
        try:
            # Merge reference images into grid
            combined_img = merge_images_smart(reference_images)
            
            # Convert to bytes for API
            img_buffer = io.BytesIO()
            combined_img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            
            result = client.images.edit(
                model=config.get("IMAGE_EDIT_MODEL_NAME", "qwen-image-edit-plus-2025-12-15"),
                image=img_bytes,
                prompt=prompt,
                size=size,
                n=1,
            )
            
            combined_img.close()
            img_buffer.close()
            
            # Process response
            for index, img_data in enumerate(result.data, start=1):
                filename = f"guided_{index}.png"
                file_path = str(TEMP_DIR / filename)
                
                if getattr(img_data, "url", None):
                    _log_info(f"Downloading image from URL...")
                    if _download_image_from_url(img_data.url, file_path):
                        saved_paths.append(file_path)
                        _log_success(f"Image saved to: {file_path}")
                    else:
                        return [f"Error: Failed to download image"]
                        
                elif getattr(img_data, "b64_json", None):
                    if _save_base64_image(img_data.b64_json, file_path):
                        saved_paths.append(file_path)
                        _log_success(f"Image saved from Base64 to: {file_path}")
                    else:
                        return [f"Error: Failed to decode Base64"]
                else:
                    _log_error("No valid image data found in response")
                    return ["Error: No valid image data found"]
                    
        except Exception as e:
            return [f"Error: {str(e)}"]
            
    # ------------------------
    # Text-Only Generation
    # ------------------------
    else:
        try:
            result = client.images.generate(
                model=config.get("IMAGE_GEN_MODEL_NAME", "qwen-image-plus"),
                prompt=prompt,
                n=1,
                quality="medium",
                size=size,
            )
            
        except Exception as e:
            return [f"Error: {str(e)}"]

        # Process response
        for index, img_data in enumerate(result.data, start=1):
            filename = f"textgen_{index}.png"
            file_path = str(TEMP_DIR / filename)

            if getattr(img_data, "url", None):
                _log_info(f"Downloading image from URL...")
                if _download_image_from_url(img_data.url, file_path):
                    saved_paths.append(file_path)
                    _log_success(f"Image saved to: {file_path}")
                else:
                    return [f"Error: Failed to download image"]

            elif getattr(img_data, "b64_json", None):
                if _save_base64_image(img_data.b64_json, file_path):
                    saved_paths.append(file_path)
                    _log_success(f"Image saved from Base64 to: {file_path}")
                else:
                    return [f"Error: Failed to decode Base64"]
            else:
                _log_error("No valid image data found in response")
                return ["Error: No valid image data found"]

    return saved_paths


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()

    # ==============================================================================
    # Test Cases (Uncomment to test locally)
    # ==============================================================================
    # Test 1: Text-only generation
    # result1 = unified_image_generator(
    #     prompt="A serene mountain lake at sunset with snow-capped peaks reflected in the water"
    # )
    # print("Text-only generation:", result1)
    
    # # Test 2: Image-guided generation
    # result2 = unified_image_generator(
    #     prompt="Transform this into a watercolor painting style",
    #     reference_images=[
    #         r"E:\github-project\Idea2Image\code\cases\9_Science-and-Logic_1\textgen.png",
    #         r"E:\github-project\Idea2Image\code\cases\8_MathVerse_11\guided.png",
    #     ]
    # )
    # print("Image-guided generation:", result2)