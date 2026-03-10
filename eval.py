import os
import json
import base64
import re
import mimetypes
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 配置区域 =====================

# 1. API 配置 (使用你提供的)
API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gemini-3-pro-preview"   # 推荐使用能力最强的模型进行评估

# 2. 路径配置
# [重要] 这里应该是你之前生成的、包含 Checklist 字段的 JSONL 文件路径
# 如果你有多个不同类型的 jsonl，建议先合并，或者修改此处分别运行
META_JSONL_PATH = r"" 

# [重要] 待评估的图像文件夹 (文件夹名将被用作模型名)
GENERATED_IMAGES_DIR = r""

# [重要] 参考图片 (World_Knowledge) 的根目录
# 程序会拼接: REFERENCE_IMAGE_ROOT + JSONL里的图片相对路径
REFERENCE_IMAGE_ROOT = r"" 

# 输出结果文件夹
OUTPUT_DIR = r""

# 3. 运行配置
MAX_WORKERS = 32   # 线程数
SAVE_INTERVAL = 100 # 每处理多少张保存一次

# ===================== 初始化客户端 =====================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ===================== 系统提示词 =====================

SYSTEM_PROMPT = (
    "您是一位客观的 **AI 图像评估员**。您的任务是验证生成的图像是否符合特定要求。\n"
    "您将收到：\n"
    "1. 一张**生成的图像**。\n"
    "2. 一张**参考图像**（真实图像，可选）。如果提供参考图像，则生成的图像必须符合其视觉特征（身份、外观、视角）的要求。\n"
    "3. 一份**检查清单**，列出了各项要求。\n\n"

    "**评估规则**：\n"
    "- 依次对照检查清单中的每一项检查生成的图像。\n"
    "- **严格评估原则**：除温度外的所有细节（例如，特定徽标、文本拼写、物体是否存在、非温度数值）如果有误，必须标记为 False。\n"
    "- **温度容差例外**：仅对于涉及**温度数值**（摄氏度 °C 或华氏度 °F）的检查项，应用以下宽松标准：\n"
    "  1. **允许范围**：允许 **±3°C**（或等效的 **±5.4°F**）的误差。\n"
    "  2. **区间判定**：如果检查项要求的是一个温度区间（例如“3-10度”），则需要分别检查边界：\n"
    "     - 图像中的**最低温**若在（目标最低温 - 3°C或5.4°F）到（目标最低温 + 3°C或5.4°F）之间，视为符合。\n"
    "     - 图像中的**最高温**若在（目标最高温 - 3°C或5.4°F）到（目标最高温 + 3°C或5.4°F）之间，视为符合。\n"
    "- 如果检查清单中提到与参考图像的一致性，请仔细比较它们。\n\n"

    "**输出格式**：\n"
    "返回一个 JSON 对象，其中包含一个键 **'results'**，其值为一个布尔值列表。\n"
    "- 'results'：[true, false, true, ...]，对应于检查清单项目的顺序。"
)

# ===================== 工具函数 =====================

def encode_image(image_path):
    """读取图片并转为 Base64"""
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type if mime_type else "image/jpeg"

def parse_filename(filename):
    """
    解析文件名: 10_43.jpg -> type_prefix="10", id="43"
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    # 兼容 logic: 取第一个部分作为 type 前缀，最后一个部分作为 ID
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return None, None

def load_metadata_map(jsonl_path):
    """
    加载包含 Checklist 的元数据
    Key: (Type_Prefix_String, ID_String) -> Value: Entry Dict
    """
    meta_map = {}
    print(f"正在加载元数据: {jsonl_path} ...")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                entry = json.loads(line)
                
                type_str = str(entry.get("Type", "")) # e.g., "10_Poem"
                entry_id = str(entry.get("ID"))       # e.g., "43"
                
                # 提取 Type 前缀 (例如 "10")
                type_prefix = type_str.split('_')[0]
                
                # 存入映射表
                meta_map[(type_prefix, entry_id)] = entry
    except FileNotFoundError:
        print(f"❌ 错误: 找不到元数据文件 {jsonl_path}")
        return {}
            
    print(f"元数据加载完成，索引了 {len(meta_map)} 条记录。")
    return meta_map

def clean_json_string(content):
    if content.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match: return match.group(1)
    return content.strip()

# ===================== 核心评估函数 =====================

def evaluate_single_image(params):
    """
    评估单张图片，返回符合要求的字典格式
    """
    img_filename, img_path, meta_entry = params
    
    # 1. 准备基础返回结构
    result_entry = {
        "Type": meta_entry.get("Type"),
        "ID": meta_entry.get("ID"),
        "Image_Name": img_filename,
        "Eval": False,              # 默认为 False
        "Checklist": [],            # 默认为空
        "Checklist_results": []     # 默认为空
    }
    
    checklist = meta_entry.get("Checklist", [])
    result_entry["Checklist"] = checklist
    
    # 如果 Checklist 为空，直接返回（视作 False 或根据需求调整）
    if not checklist:
        return result_entry

    # 2. 读取生成图 (Generated Image)
    gen_b64 = encode_image(img_path)
    if not gen_b64:
        print(f"⚠️ 无法读取生成图: {img_path}")
        # 填充全 False
        result_entry["Checklist_results"] = [False] * len(checklist)
        return result_entry
    gen_mime = get_mime_type(img_path)

    # 3. 读取参考图 (Reference Image)
    # 处理逻辑：检查 meta_entry 中的 World_Knowledge 字段
    ref_b64 = None
    ref_mime = None
    
    wk_data = meta_entry.get("World_Knowledge", {})
    ref_img_rel_path = ""
    
    # 兼容之前生成的 {"text": "...", "image": "..."} 结构 或 直接字符串结构
    if isinstance(wk_data, dict):
        # 优先取 image 字段
        img_field = wk_data.get("image", "")
        if isinstance(img_field, list) and len(img_field) > 0:
            ref_img_rel_path = img_field[0] # 取第一张作为主参考
        elif isinstance(img_field, str):
            ref_img_rel_path = img_field
    elif isinstance(wk_data, str) and (wk_data.endswith('.jpg') or wk_data.endswith('.png')):
         ref_img_rel_path = wk_data
    
    # 如果找到了相对路径，拼接完整路径并读取
    if ref_img_rel_path:
        full_ref_path = os.path.join(REFERENCE_IMAGE_ROOT, ref_img_rel_path)
        ref_b64 = encode_image(full_ref_path)
        ref_mime = get_mime_type(full_ref_path)

    # 4. 构造 Prompt
    user_content = []
    
    # 4.1 放入 Checklist 文本
    checklist_str = "Checklist to Verify:\n"
    for i, item in enumerate(checklist):
        checklist_str += f"{i+1}. {item}\n"
    user_content.append({"type": "text", "text": checklist_str})
    
    # 4.2 放入生成图
    user_content.append({"type": "text", "text": "--- Generated Image (Target) ---"})
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{gen_mime};base64,{gen_b64}"}
    })
    
    # 4.3 放入参考图 (如有)
    if ref_b64:
        user_content.append({"type": "text", "text": "--- Reference Image (Ground Truth) ---"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"}
        })
    
    # 5. 调用 API
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0, # 评估需要确定性
            response_format={"type": "json_object"}
        )
        
        content = clean_json_string(response.choices[0].message.content)
        api_result = json.loads(content)
        
        # 获取布尔值列表
        bool_results = api_result.get("results", [])
        
        # 6. 后处理与对齐
        # 确保结果长度与 Checklist 一致 (如果不够长，补 False)
        if len(bool_results) < len(checklist):
            bool_results.extend([False] * (len(checklist) - len(bool_results)))
        elif len(bool_results) > len(checklist):
            bool_results = bool_results[:len(checklist)]
            
        result_entry["Checklist_results"] = bool_results
        
        # 计算最终 Eval：只有全部为 True 才算 True
        result_entry["Eval"] = all(bool_results) and len(bool_results) > 0

    except Exception as e:
        print(f"⚠️ API Error for {img_filename}: {e}")
        result_entry["Checklist_results"] = [False] * len(checklist)
        result_entry["Eval"] = False
        
    return result_entry

# ===================== 主函数 =====================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 加载元数据字典
    meta_map = load_metadata_map(META_JSONL_PATH)
    if not meta_map:
        return

    # 2. 扫描待评估图片
    model_name = os.path.basename(GENERATED_IMAGES_DIR.rstrip("\\/")) # 获取文件夹名
    output_filename = f"{model_name}.jsonl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    processed_images = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "Image_Name" in data:
                        processed_images.add(data["Image_Name"])
                except:
                    continue
    if processed_images:
        print(f"检测到续跑：已处理 {len(processed_images)} 条数据，将自动跳过。")
    
    print(f"正在扫描生成图文件夹: {GENERATED_IMAGES_DIR}")
    tasks = []
    
    for root, dirs, files in os.walk(GENERATED_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                if file in processed_images:
                    continue
                # 解析文件名 10_43.jpg -> 10, 43
                prefix, id_str = parse_filename(file)
                if prefix and id_str:
                    key = (prefix, id_str)
                    if key in meta_map:
                        full_img_path = os.path.join(root, file)
                        tasks.append((file, full_img_path, meta_map[key]))
                    else:
                        # 如果没有找到对应的 checklist，跳过
                        pass
    
    print(f"找到 {len(tasks)} 张图片匹配到 Checklist。开始评估...")
    
    # 3. 多线程执行与保存
    # 使用 'w' 模式覆盖旧文件，或 'a' 追加。这里使用 'w' 确保全新开始。
    with open(output_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(evaluate_single_image, t): t[0] for t in tasks}
            
            for i, future in tqdm(enumerate(as_completed(future_to_file)), total=len(tasks)):
                result = future.result()
                
                # 写入 JSONL
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                # 定期刷新到磁盘
                if (i + 1) % SAVE_INTERVAL == 0:
                    f.flush()

    print(f"✅ 评估完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    main()