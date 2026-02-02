"""
Image RAG Tool
==============
MCP tool for searching and downloading images from the web
using Google Custom Search API or Serper API for visual reference retrieval.

Supports automatic API selection based on configuration.
"""

import os
import sys

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import List, Optional, Literal
from io import BytesIO
from enum import Enum

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

# API Keys
GOOGLE_API_KEY = config.get("GOOGLE_API_KEY", "")
GOOGLE_SEARCH_ENGINE_ID = config.get("GOOGLE_SEARCH_ENGINE_ID", "")
SERPER_API_KEY = config.get("SERPER_API_KEY", "")

# Image search configuration
image_search_config = config.get("image_search", {})
PROVIDER = image_search_config.get("provider", "auto")
PRIORITY = image_search_config.get("priority", ["google", "serper"])
NUM_IMAGES = image_search_config.get("num_images", 5)

# Initialize MCP server
mcp = FastMCP(
    name="Image Searcher",
    instructions="MCP tool for searching and downloading images from the web using Google Custom Search or Serper API."
)


# ==============================================================================
# API Provider Enum
# ==============================================================================

class ImageSearchProvider(str, Enum):
    """Supported image search API providers."""
    GOOGLE = "google"
    SERPER = "serper"
    AUTO = "auto"


# ==============================================================================
# Helper Functions
# ==============================================================================

def fetch_google_image_links(query: str, num_images: int = 5) -> List[str]:
    """
    Fetch image links using Google Custom Search API.
    
    Args:
        query: Search query string
        num_images: Number of images to retrieve (max 10 per request)
        
    Returns:
        List of image URLs
        
    Raises:
        requests.HTTPError: If API request fails
    """
    if not GOOGLE_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        _log_error("Google API key or Search Engine ID not configured")
        return []
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "searchType": "image",
        "num": min(num_images, 10),  # Google API max is 10
        "safe": "off"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get("items", [])

        links = []
        for item in items:
            image_url = item.get("link")
            if image_url:
                links.append(image_url)
        
        # Log fetched URLs for debugging
        log_file = TEMP_DIR / "google_fetched_image_urls.txt"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                for link in links:
                    f.write(f"{query} --> {link}\n")
        except Exception:
            pass  # Non-critical logging
        
        return links
        
    except Exception as e:
        _log_error(f"Error fetching Google images: {e}")
        return []


def fetch_serper_image_links(query: str, num_images: int = 5) -> List[str]:
    """
    Fetch image links using Serper API.
    
    Args:
        query: Search query string
        num_images: Number of images to retrieve
        
    Returns:
        List of image URLs
        
    Raises:
        requests.HTTPError: If API request fails
    """
    if not SERPER_API_KEY:
        _log_error("Serper API key not configured")
        return []
    
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
        log_file = TEMP_DIR / "serper_fetched_image_urls.txt"
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


def _get_available_provider() -> Optional[ImageSearchProvider]:
    """
    Determine which image search provider to use based on API key availability.
    
    Returns:
        Available provider enum, or None if no providers are available
    """
    # Check configured provider
    if PROVIDER == "google" and GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
        return ImageSearchProvider.GOOGLE
    elif PROVIDER == "serper" and SERPER_API_KEY:
        return ImageSearchProvider.SERPER
    elif PROVIDER == "auto":
        # Auto-detect based on priority
        for provider_name in PRIORITY:
            if provider_name == "google" and GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
                return ImageSearchProvider.GOOGLE
            elif provider_name == "serper" and SERPER_API_KEY:
                return ImageSearchProvider.SERPER
    
    # Fallback: try any available
    if SERPER_API_KEY:
        return ImageSearchProvider.SERPER
    elif GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
        return ImageSearchProvider.GOOGLE
    
    return None


def fetch_image_links(query: str, num_images: int = 5) -> List[str]:
    """
    Unified function to fetch image links using the configured provider.
    
    Automatically selects the appropriate API based on configuration:
    - Uses explicit provider if set in config
    - Auto-detects based on priority order
    - Falls back to any available provider
    
    Args:
        query: Search query string
        num_images: Number of images to retrieve
        
    Returns:
        List of image URLs from the selected provider
        
    Example:
        >>> links = fetch_image_links("cyberpunk city", num_images=3)
        >>> print(f"Found {len(links)} images")
    """
    provider = _get_available_provider()
    
    if provider is None:
        _log_error("No image search API configured. Please set GOOGLE_API_KEY or SERPER_API_KEY in config.yaml")
        return []
    
    _log_info(f"Using {provider.value} API for image search")
    
    if provider == ImageSearchProvider.GOOGLE:
        return fetch_google_image_links(query, num_images)
    elif provider == ImageSearchProvider.SERPER:
        return fetch_serper_image_links(query, num_images)
    else:
        _log_error(f"Unknown provider: {provider}")
        return []


def download_image_with_pil(url: str, filename_prefix: str) -> str:
    """
    Download and validate image using PIL.
    
    Args:
        url: Image URL to download
        filename_prefix: Prefix for saved filename
        
    Returns:
        Local file path if successful, empty string otherwise
        
    Note:
        Returns empty string on any error to allow graceful degradation.
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
        # Log warning but don't raise - allows trying next image
        _log_warning(f"Failed to download image from {url[:50]}...: {e}")
        return ""


def _sanitize_filename(query: str) -> str:
    """
    Create a filesystem-safe filename from query string.
    
    Args:
        query: Raw search query
        
    Returns:
        Sanitized filename (alphanumeric + underscores, max 50 chars)
    """
    safe_chars = [c for c in query if c.isalnum() or c in (" ", "_")]
    return "".join(safe_chars).strip().replace(" ", "_")[:50]


# ==============================================================================
# MCP Tool Definition
# ==============================================================================

@mcp.tool()
def search_and_download_images_batch(image_queries: List[str]) -> List[str]:
    """
    Batch search and download images for multiple queries.
    
    Uses the configured image search provider (Google or Serper) to find
    and download reference images for the given queries.
    
    Args:
        image_queries: List of search keywords
        
    Returns:
        List of local paths to successfully downloaded images
        
    Example:
        >>> queries = ["cyberpunk city", "traditional temple"]
        >>> paths = search_and_download_images_batch(queries)
        >>> print(f"Downloaded {len(paths)} images")
    """
    downloaded_paths: List[str] = []
    max_attempts = NUM_IMAGES  # Try up to configured number of images per query

    for query in image_queries:
        # Fetch image links using unified function
        image_links = fetch_image_links(query, num_images=max_attempts)
        success = False

        for idx, url in enumerate(image_links[:max_attempts]):
            # Generate safe filename
            safe_query = _sanitize_filename(query)
            prefix = f"{safe_query}_{idx + 1}"
            
            # Attempt download
            file_path = download_image_with_pil(url, prefix)
            
            if file_path:
                downloaded_paths.append(file_path)
                _log_success(f"Downloaded: {file_path}")
                success = True
                break  # One successful download per query
        
        if not success:
            _log_warning(f"Failed to download any images for: {query}")
    
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
    # result = search_and_download_images_batch(query)
    # print("Downloaded images:", result)