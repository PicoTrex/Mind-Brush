from fastmcp import FastMCP
import yaml
import os
import requests
import argparse
import json
from pathlib import Path
from typing import List
from PIL import Image
from io import BytesIO
import sys

with open(f"./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if config.get("proxy_on", False):
    os.environ["http_proxy"] = config.get("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ["https_proxy"] = config.get("HTTPS_PROXY", "http://127.0.0.1:7890")

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ✅ MCP 初始化
mcp = FastMCP("My Image Search")

TEMP_DIR = Path(config.get("temp_dir", "./temp").get("image_rag", "./temp/image_rag")).absolute()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Serper API 配置
SERPER_API_KEY = config.get("SERPER_API_KEY", "bdf9b167a18e2e9071e4eed39f257aa28d8ad10c")

def fetch_serper_image_links(query: str, num_images: int = 5) -> List[str]:
    """
    使用 Serper API 获取图像链接
    """
    url = "https://google.serper.dev/images"
    payload = json.dumps({
        "q": query,
        "num": num_images
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Serper 图片链接在 images -> imageUrl
        items = data.get("images", [])
        links = []
        for item in items:
            image_url = item.get("imageUrl")
            if image_url:
                links.append(image_url)
                
        # 记录 URL 获取日志
        with open(TEMP_DIR / "google_fetched_image_urls.txt", "w", encoding="utf-8") as f:
            for link in links:
                f.write(f"{query} --> {link}\n")
                
        return links
    except Exception as e:
        print(f"Error fetching Serper images: {e}")
        return []

def download_image_with_pil(url: str, filename_prefix: str) -> str:
    """
    下载图像，使用 PIL 验证并保存到本地
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()
        
        # 使用 PIL 打开数据流，验证这是否是真正的图片
        image = Image.open(BytesIO(response.content))

        # 确定扩展名
        ext = image.format.lower() if image.format else "jpg"
        if ext == "jpeg": ext = "jpg"
        
        filename = f"{filename_prefix}.{ext}"
        file_path = TEMP_DIR / filename
        image.save(file_path)
        return str(file_path)
    except Exception as e:
        # 失败时返回 None，让上层逻辑尝试下一张
        return None

@mcp.tool()
def search_and_download_images_batch(image_queries: List[str]) -> List[str]:
    """
    MCP 工具：批量搜索并下载图像

    Args:
        image_queries (List[str]): 搜索关键词列表
    Returns:
        downloaded_paths(List[str]): 本地成功保存的图片路径
    """
    downloaded_paths = []
    max_attempts = 5 # 每个 query 尝试下载前 5 张图

    for query in image_queries:
        # print(f"🔍 Processing Image Search: {query}")
        image_links = fetch_serper_image_links(query, num_images=max_attempts)
        success = False

        for idx, url in enumerate(image_links[:max_attempts]):
            # 生成安全的文件名
            safe_query = "".join([c for c in query if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
            prefix = f"{safe_query}_{idx+1}"
            
            # 尝试下载
            file_path = download_image_with_pil(url, prefix)
            
            if file_path:
                downloaded_paths.append(file_path)
                success = True
                break  # 策略：只要这一张成功了，就不下载后面的了
        
        if not success:
            print(f"❌ Failed to download image for: {query}")
            
    return downloaded_paths


if __name__ == "__main__":
    mcp.run()
    # print(search_and_download_logic(['Pop Mart Tom and Jerry Forbidden Compass Series Tom in Lantern']))