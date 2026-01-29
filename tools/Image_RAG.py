"""
Image RAG Tool
==============
MCP tool for searching and downloading images from the web
using the Serper API for visual reference retrieval.
"""

import os
import sys

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import List
from io import BytesIO

import requests
from PIL import Image
from fastmcp import FastMCP

# Import shared utilities
from tools.base import (
    load_config,
    setup_proxy_from_config,
    setup_stdio_encoding,
    _log_info,
    _log_error,
    _log_success,
    _log_warning,
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
    TEMP_DIR = Path(session_dir) / "temp" / "image_rag"
else:
    temp_dir_config = config.get("temp_dir", {})
    if isinstance(temp_dir_config, dict):
        TEMP_DIR = Path(temp_dir_config.get("image_rag", "./temp/image_rag")).absolute()
    else:
        TEMP_DIR = Path("./temp/image_rag").absolute()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Serper API configuration
SERPER_API_KEY = config.get("SERPER_API_KEY", "")

# Initialize MCP server
mcp = FastMCP(
    name="Image Searcher",
    instructions="MCP tool for searching and downloading images from the web using the Serper API for visual reference retrieval."
)

# ==============================================================================
# Helper Functions
# ==============================================================================

def fetch_serper_image_links(query: str, num_images: int = 5) -> List[str]:
    """
    Fetch image links using Serper API.
    
    Args:
        query: Search query string
        num_images: Number of images to retrieve
        
    Returns:
        List of image URLs
    """
    url = "https://google.serper.dev/images"
    payload = json.dumps({
        "q": query,
        "num": num_images
    })
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract image URLs from response
        items = data.get("images", [])
        links = []
        for item in items:
            image_url = item.get("imageUrl")
            if image_url:
                links.append(image_url)
        
        # Log fetched URLs for debugging
        log_file = TEMP_DIR / "fetched_image_urls.txt"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                for link in links:
                    f.write(f"{query} --> {link}\n")
        except Exception:
            pass  # Non-critical logging
        
        return links
        
    except Exception as e:
        _log_error(f"Error fetching Serper images: {e}")
        return []


def download_image_with_pil(url: str, filename_prefix: str) -> str:
    """
    Download image with PIL validation.
    
    Args:
        url: Image URL to download
        filename_prefix: Prefix for saved filename
        
    Returns:
        Local file path if successful, empty string otherwise
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()
        
        # Validate with PIL
        image = Image.open(BytesIO(response.content))
        
        # Determine extension from format
        ext = (image.format or "jpg").lower()
        if ext == "jpeg":
            ext = "jpg"
        
        filename = f"{filename_prefix}.{ext}"
        file_path = TEMP_DIR / filename
        image.save(file_path)
        
        return str(file_path)
        
    except Exception as e:
        # Return empty string to indicate failure
        return ""


def _sanitize_filename(query: str) -> str:
    """Create a safe filename from query string."""
    safe_chars = [c for c in query if c.isalnum() or c in (" ", "_")]
    return "".join(safe_chars).strip().replace(" ", "_")[:50]


# ==============================================================================
# MCP Tool Definition
# ==============================================================================

@mcp.tool()
def search_and_download_images_batch(image_queries: List[str]) -> List[str]:
    """
    Batch search and download images for multiple queries.
    
    Args:
        image_queries: List of search keywords
        
    Returns:
        List of local paths to successfully downloaded images
    """
    downloaded_paths: List[str] = []
    max_attempts = 5  # Try up to 5 images per query

    for query in image_queries:
        image_links = fetch_serper_image_links(query, num_images=max_attempts)
        success = False

        for idx, url in enumerate(image_links[:max_attempts]):
            # Generate safe filename
            safe_query = _sanitize_filename(query)
            prefix = f"{safe_query}_{idx + 1}"
            
            # Attempt download
            file_path = download_image_with_pil(url, prefix)
            
            if file_path:
                downloaded_paths.append(file_path)
                success = True
                break  # One successful download per query
        
        if not success:
            _log_warning(f"Failed to download image for: {query}")
    
    return downloaded_paths


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()

    # ==============================================================================
    # Test Cases (Uncomment to test locally)
    # ==============================================================================
    # query = ["Cyberpunk cityscape at night", "Traditional Japanese temple"]
    # result = search_and_download_images_batch_test(query)
    # print("Downloaded images:", result)