import os
import math
import sys
import io
import base64
import yaml
import mimetypes
from typing import List

from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from PIL import Image
from pathlib import Path
import requests

# 强制标准输出使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

with open(f"./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if config.get("proxy_on", False):
    os.environ["http_proxy"] = config.get("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ["https_proxy"] = config.get("HTTPS_PROXY", "http://127.0.0.1:7890")

TEMP_DIR = Path(config.get("temp_dir", "./temp").get("image_gen", "./temp/image_gen")).absolute()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

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
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            # 修改：使用 stderr 打印日志，防止破坏 MCP 协议
            sys.stderr.write(f"Warning: Could not open {path}, skipping. Error: {e}\n")

    if not images:
        raise ValueError("No valid images found to merge.")

    count = len(images)
    
    # 1. 计算网格的行数和列数
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    # 2. 确定单个网格单元的目标尺寸
    cell_max_w = 768
    cell_max_h = 768
    
    # 3. 创建大画布
    grid_width = cols * cell_max_w
    grid_height = rows * cell_max_h
    
    combined_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255)) # 白色背景

    # 4. 遍历图片并粘贴
    for index, img in enumerate(images):
        row_idx = index // cols
        col_idx = index % cols
        
        x_offset = col_idx * cell_max_w
        y_offset = row_idx * cell_max_h
        
        # 核心逻辑：保持比例缩放 (Contain)
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

    # 5. 最后整体缩放
    if combined_image.width > max_side or combined_image.height > max_side:
        combined_image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    return combined_image

# # ---用于 Base64 编码 ---
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")

# 初始化 MCP 服务
mcp = FastMCP(
    name="Unified Image Generator",
    instructions="This MCP provides both text-to-image and image-guided generation using OpenAI's image models."
)

@mcp.tool(
    name="unified_image_generator",
    description="Generate image(s) based on prompt only or prompt + multiple reference images."
)
def unified_image_generator(
    prompt: str,
    reference_images: List[str] = []
) -> List[str]: # <--- 修改 1: 返回类型必须与实际返回值 saved_paths (list) 一致
    
    size = "1024x1024"

    client = OpenAI(
        base_url=config.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"),
        api_key=config.get("OPENAI_API_KEY", "sk-X9jXfVLVKHEK6y06p6MRJuEHwvqQX240PPrebQikc1fBXeIS"),
    )

    saved_paths: List[str] = []

    # ------------------------
    # 图像引导生成（有图像）
    # ------------------------
    if reference_images:
        try:
            combined_img = merge_images_smart(reference_images)

            img_buffer = io.BytesIO()
            combined_img.save(img_buffer, format='PNG')
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
            
            for index, img_data in enumerate(result.data, start=1):
                filename = f"guided.png"
                file_path = os.path.join(TEMP_DIR, filename)
                
                if getattr(img_data, 'url', None):
                    image_url = img_data.url
                    # 修改：使用 stderr
                    sys.stderr.write(f"⬇️ Downloading image from URL: {image_url}\n")
                    
                    try:
                        response = requests.get(image_url, stream=True)
                        response.raise_for_status()
                        
                        with open(file_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        saved_paths.append(file_path)
                        sys.stderr.write(f"✅ Image saved to: {file_path}\n")
                        
                    except Exception as e:
                        sys.stderr.write(f"❌ Failed to download image from URL: {e}\n")
                        return [str(e)] # 修改：返回列表以匹配类型

                elif getattr(img_data, 'b64_json', None):
                    try:
                        img_bytes = base64.b64decode(img_data.b64_json)
                        with open(file_path, "wb") as f:
                            f.write(img_bytes)
                        
                        saved_paths.append(file_path)
                        sys.stderr.write(f"✅ Image saved from Base64 to: {file_path}\n")
                    except Exception as e:
                        sys.stderr.write(f"❌ Failed to decode Base64: {e}\n")
                        return [str(e)] # 修改：返回列表
                        
                else:
                    sys.stderr.write("❌ Error: No valid image data (URL or b64_json) found in response.\n")
                    return ["Error: No valid image data found"]
                
        except Exception as e:
            # 修改：返回列表以匹配类型
            return [str(e)]
            
    # ------------------------
    # 文本生成（无图像）
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
            return [str(e)] # 修改：返回列表

        for index, img_data in enumerate(result.data, start=1):
            filename = f"textgen.png"
            file_path = os.path.join(TEMP_DIR, filename)

            if getattr(img_data, 'url', None):
                image_url = img_data.url
                sys.stderr.write(f"⬇️ Downloading image from URL: {image_url}\n")
                
                try:
                    response = requests.get(image_url, stream=True)
                    response.raise_for_status()
                    
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    saved_paths.append(file_path)
                    sys.stderr.write(f"✅ Image saved to: {file_path}\n")
                    
                except Exception as e:
                    sys.stderr.write(f"❌ Failed to download image from URL: {e}\n")
                    return [str(e)]

            elif getattr(img_data, 'b64_json', None):
                try:
                    img_bytes = base64.b64decode(img_data.b64_json)
                    with open(file_path, "wb") as f:
                        f.write(img_bytes)
                    
                    saved_paths.append(file_path)
                    sys.stderr.write(f"✅ Image saved from Base64 to: {file_path}\n")
                except Exception as e:
                    sys.stderr.write(f"❌ Failed to decode Base64: {e}\n")
                    return [str(e)]
                    
            else:
                sys.stderr.write("❌ Error: No valid image data (URL or b64_json) found in response.\n")
                return ["Error: No valid image data found"]

    return saved_paths

# 启动 MCP 服务
if __name__ == "__main__":
    mcp.run()