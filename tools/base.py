"""
MindBrush Tools - Shared Utilities
==================================
This module provides shared utility functions used across all MCP tools.

Contains:
- Image encoding/decoding utilities
- Image processing (merge, resize)
- JSON response cleaning
- Proxy and configuration helpers

Note: This module is designed to work both when imported as a package
and when tools are run as standalone MCP servers.
"""

import os
import io
import sys
import math
import base64
import json
import mimetypes
import warnings
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

from PIL import Image


# ==============================================================================
# Image Encoding Utilities
# ==============================================================================

def encode_image(image_source: Union[str, bytes, Image.Image]) -> str:
    """
    Universal image encoder supporting file paths, bytes, and PIL Images.
    
    Returns a Data URI string suitable for LLM vision API inputs.
    
    Args:
        image_source: One of:
            - str: File path to an image
            - bytes: Raw image bytes
            - PIL.Image.Image: PIL Image object
            
    Returns:
        Data URI string (e.g., "data:image/jpeg;base64,...")
        
    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If input type is not supported
        
    Example:
        >>> uri = encode_image("/path/to/image.jpg")
        >>> uri = encode_image(pil_image)
        >>> uri = encode_image(image_bytes)
    """
    mime_type = "image/jpeg"
    image_data: Optional[bytes] = None

    try:
        # Case 1: File path (str)
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"File not found: {image_source}")
            
            # Determine MIME type from extension
            guessed_type, _ = mimetypes.guess_type(image_source)
            if guessed_type:
                mime_type = guessed_type
            else:
                # Fallback: Detect from extension
                ext = os.path.splitext(image_source)[1].lower()
                mime_type = _get_mime_from_extension(ext)
            
            with open(image_source, "rb") as f:
                image_data = f.read()
        
        # Case 2: PIL Image object
        elif isinstance(image_source, Image.Image):
            buffer = io.BytesIO()
            # Preserve format if available, otherwise default to JPEG
            fmt = image_source.format if image_source.format else "JPEG"
            
            # Handle RGBA images (PNG with alpha)
            if image_source.mode == "RGBA" and fmt.upper() == "JPEG":
                # Convert RGBA to RGB for JPEG
                image_source = image_source.convert("RGB")
            
            image_source.save(buffer, format=fmt)
            image_data = buffer.getvalue()
            mime_type = f"image/{fmt.lower()}"
        
        # Case 3: Raw bytes
        elif isinstance(image_source, bytes):
            image_data = image_source
            # Default to JPEG for raw bytes
            mime_type = "image/jpeg"

        else:
            raise ValueError(
                f"Unsupported input type: {type(image_source)}. "
                "Expected str (file path), PIL.Image.Image, or bytes."
            )

        # Encode to base64
        if image_data is None:
            raise ValueError("Failed to extract image data")
            
        encoded_string = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"

    except Exception as e:
        warnings.warn(f"Failed to encode image: {str(e)}")
        raise


def _get_mime_from_extension(ext: str) -> str:
    """Get MIME type from file extension."""
    mime_map = {
        ".webp": "image/webp",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mime_map.get(ext.lower(), "image/jpeg")


# ==============================================================================
# Image Processing Utilities
# ==============================================================================

def merge_images_smart(
    image_paths: List[str],
    max_side: int = 2048,
    cell_max_w: int = 768,
    cell_max_h: int = 768,
    background_color: tuple = (255, 255, 255)
) -> Image.Image:
    """
    Smart grid merging: Combine multiple images into an approximately square grid.
    
    Preserves aspect ratios of individual images without stretching.
    Automatically calculates optimal grid dimensions.
    
    Args:
        image_paths: List of image file paths to merge
        max_side: Maximum dimension of final merged image
        cell_max_w: Maximum width of each grid cell
        cell_max_h: Maximum height of each grid cell
        background_color: RGB tuple for background color
        
    Returns:
        PIL.Image.Image: Merged grid image
        
    Raises:
        ValueError: If no images provided or no valid images found
        
    Example:
        >>> merged = merge_images_smart(["/path/img1.jpg", "/path/img2.jpg"])
        >>> merged.save("grid.png")
    """
    if not image_paths:
        raise ValueError("No images provided for merging.")
    
    # Load all valid images
    images: List[Image.Image] = []
    for path in image_paths:
        try:
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                images.append(img)
            else:
                _log_warning(f"Image path not found: {path}, skipping.")
        except Exception as e:
            _log_warning(f"Could not open {path}, skipping. Error: {e}")

    if not images:
        raise ValueError("No valid images found to merge.")

    count = len(images)
    
    # Calculate grid dimensions (approximately square)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    # Create canvas
    grid_width = cols * cell_max_w
    grid_height = rows * cell_max_h
    combined_image = Image.new("RGB", (grid_width, grid_height), background_color)

    # Paste each image into its cell
    for index, img in enumerate(images):
        row_idx = index // cols
        col_idx = index % cols
        
        x_offset = col_idx * cell_max_w
        y_offset = row_idx * cell_max_h
        
        # Calculate resize dimensions preserving aspect ratio
        img_aspect = img.width / img.height
        cell_aspect = cell_max_w / cell_max_h
        
        if img_aspect > cell_aspect:
            # Image is wider than cell
            new_w = cell_max_w
            new_h = int(cell_max_w / img_aspect)
        else:
            # Image is taller than cell
            new_h = cell_max_h
            new_w = int(cell_max_h * img_aspect)
        
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center in cell
        paste_x = x_offset + (cell_max_w - new_w) // 2
        paste_y = y_offset + (cell_max_h - new_h) // 2
        
        combined_image.paste(resized_img, (paste_x, paste_y))

    # Resize if exceeds max_side
    if combined_image.width > max_side or combined_image.height > max_side:
        combined_image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    return combined_image


# ==============================================================================
# JSON Utilities
# ==============================================================================

def clean_json_markdown(content: str) -> str:
    """
    Clean markdown code block formatting from JSON responses.
    
    LLMs often wrap JSON responses in ```json ... ``` blocks.
    This function removes such formatting to get clean JSON.
    
    Args:
        content: Raw response content that may contain markdown
        
    Returns:
        Cleaned content ready for JSON parsing
        
    Example:
        >>> clean_json_markdown('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
    """
    content = content.strip()
    
    # Pattern 1: ```json ... ```
    if content.startswith("```json"):
        content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
    
    # Pattern 2: ``` ... ```
    if content.startswith("```"):
        content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
    
    return content


def parse_json_response(content: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Parse JSON from LLM response with cleaning and error handling.
    
    Args:
        content: Raw response content
        default: Default value if parsing fails (None raises exception)
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If parsing fails and no default provided
    """
    try:
        cleaned = clean_json_markdown(content)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        if default is not None:
            return default
        raise e


# ==============================================================================
# Configuration Helpers
# ==============================================================================

def get_project_root() -> Path:
    """
    Get the project root directory.
    Works whether running as standalone script or as module.
    """
    # Try to find project root by looking for config.yaml
    current = Path(__file__).resolve().parent
    
    # If we're in tools/, go up one level
    if current.name == "tools":
        current = current.parent
    
    # Check if this is the project root
    if (current / "config.yaml").exists() or (current / "app.py").exists():
        return current
    
    # Fallback to cwd
    return Path.cwd()


def setup_proxy_from_config(config: Dict[str, Any]) -> None:
    """
    Set up proxy environment variables from configuration.
    
    Args:
        config: Configuration dictionary with proxy settings
    """
    if config.get("proxy_on", False):
        os.environ["http_proxy"] = config.get("HTTP_PROXY", "http://127.0.0.1:7890")
        os.environ["https_proxy"] = config.get("HTTPS_PROXY", "http://127.0.0.1:7890")


def setup_stdio_encoding() -> None:
    """
    Configure stdout/stderr for UTF-8 encoding.
    
    Required on Windows to prevent encoding errors in MCP stdio communication.
    """
    if sys.platform.startswith("win"):
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
    else:
        # Unix-like systems - only wrap if not already wrapped
        if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


# ==============================================================================
# Logging Helpers
# ==============================================================================

def _log_warning(message: str) -> None:
    """Log a warning message to stderr (MCP-safe)."""
    sys.stderr.write(f"⚠️ {message}\n")
    sys.stderr.flush()


def _log_info(message: str) -> None:
    """Log an info message to stderr (MCP-safe)."""
    sys.stderr.write(f"ℹ️ {message}\n")
    sys.stderr.flush()


def _log_error(message: str) -> None:
    """Log an error message to stderr (MCP-safe)."""
    sys.stderr.write(f"❌ {message}\n")
    sys.stderr.flush()


def _log_success(message: str) -> None:
    """Log a success message to stderr (MCP-safe)."""
    sys.stderr.write(f"✅ {message}\n")
    sys.stderr.flush()


# ==============================================================================
# Configuration Loading (Works for standalone tools)
# ==============================================================================

def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Works whether running from project root or from tools directory.
        
    Args:
        config_path: Path to configuration file (relative to project root)
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    # Try the given path first
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    # Try from project root
    project_root = get_project_root()
    full_path = project_root / config_path.lstrip("./")
    
    if full_path.exists():
        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    _log_warning(f"Config file not found: {config_path}")
    return {}


def load_prompt(prompt_name: str, prompts_dir: str = "./prompts") -> str:
    """
    Load a system prompt from YAML file.
    
    Works whether running from project root or from tools directory.
    
    Args:
        prompt_name: Name of the prompt (without extension)
        prompts_dir: Directory containing prompt files
        
    Returns:
        System prompt string
    """
    import yaml
    
    # Try direct path first
    prompt_path = Path(prompts_dir) / f"{prompt_name}.yaml"
    
    if not prompt_path.exists():
        # Try from project root
        project_root = get_project_root()
        prompt_path = project_root / prompts_dir.lstrip("./") / f"{prompt_name}.yaml"
    
    if not prompt_path.exists():
        _log_warning(f"Prompt file not found: {prompt_name}")
        return ""
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data.get("system_prompt", "")