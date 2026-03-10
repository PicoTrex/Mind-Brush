import os
import json
import base64
import re
import mimetypes
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== Configuration Area =====================

# 1. API Configuration
API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gemini-3-pro-preview"

# 2. Path Configuration
META_JSONL_PATH = r""

# [Important] Folder containing generated images (folder name will be used as model name)
GENERATED_IMAGES_DIR = r""

# [Important] Root directory for reference images (World_Knowledge)
REFERENCE_IMAGE_ROOT = r""

# Output results directory
OUTPUT_DIR = r""

# 3. Run Configuration
MAX_WORKERS = 32
SAVE_INTERVAL = 100

# ===================== Initialize Client =====================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ===================== System Prompt =====================

SYSTEM_PROMPT = (
    "You are an objective **AI Image Evaluator**. Your task is to verify if the generated image meets specific requirements.\n"
    "You will receive:\n"
    "1. A **generated image**.\n"
    "2. A **reference image** (real image, optional). If a reference image is provided, the generated image must comply with its visual characteristics (identity, appearance, perspective) requirements.\n"
    "3. A **checklist** listing various requirements.\n\n"

    "**Evaluation Rules**:\n"
    "- Check the generated image against each item in the checklist sequentially.\n"
    "- **Strict Evaluation Principle**: All details except temperature (e.g., specific logos, text spelling, presence of objects, non-temperature numerical values) must be marked as False if incorrect.\n"
    "- **Temperature Tolerance Exception**: Only for checklist items involving **temperature values** (Celsius °C or Fahrenheit °F), apply the following relaxed standards:\n"
    "  1. **Allowed Range**: Allow an error of **±3°C** (or equivalent **±5.4°F**).\n"
    "  2. **Interval Judgment**: If the checklist item requires a temperature range (e.g., '3-10 degrees'), check the boundaries separately:\n"
    "     - If the **minimum temperature** in the image is between (target min - 3°C/5.4°F) and (target min + 3°C/5.4°F), it is considered compliant.\n"
    "     - If the **maximum temperature** in the image is between (target max - 3°C/5.4°F) and (target max + 3°C/5.4°F), it is considered compliant.\n"
    "- If the checklist mentions consistency with the reference image, compare them carefully.\n\n"

    "**Output Format**:\n"
    "Return a JSON object containing a key **'results'**, which is a list of boolean values.\n"
    "- 'results': [true, false, true, ...], corresponding to the order of the checklist items."
)

# ===================== Helper Functions =====================

def encode_image(image_path):
    """Read image and convert to Base64"""
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
    Parse filename: 10_43.jpg -> type_prefix="10", id="43"
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return None, None

def load_metadata_map(jsonl_path):
    """
    Load metadata containing Checklist
    Key: (Type_Prefix_String, ID_String) -> Value: Entry Dict
    """
    meta_map = {}
    print(f"Loading metadata: {jsonl_path} ...")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                entry = json.loads(line)
                
                type_str = str(entry.get("Type", ""))
                entry_id = str(entry.get("ID"))
                
                type_prefix = type_str.split('_')[0]
                
                meta_map[(type_prefix, entry_id)] = entry
    except FileNotFoundError:
        print(f"Error: Could not find metadata file {jsonl_path}")
        return {}
            
    print(f"Metadata loaded, indexed {len(meta_map)} entries.")
    return meta_map

def clean_json_string(content):
    if content.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match: return match.group(1)
    return content.strip()

# ===================== Core Evaluation Function =====================

def evaluate_single_image(params):
    """
    Evaluate a single image, return the result in the required dictionary format
    """
    img_filename, img_path, meta_entry = params
    
    result_entry = {
        "Type": meta_entry.get("Type"),
        "ID": meta_entry.get("ID"),
        "Image_Name": img_filename,
        "Eval": False,
        "Checklist": [],
        "Checklist_results": []
    }
    
    checklist = meta_entry.get("Checklist", [])
    result_entry["Checklist"] = checklist
    
    if not checklist:
        return result_entry

    # 1. Read Generated Image
    gen_b64 = encode_image(img_path)
    if not gen_b64:
        print(f"Warning: Could not read generated image: {img_path}")
        result_entry["Checklist_results"] = [False] * len(checklist)
        return result_entry
    gen_mime = get_mime_type(img_path)

    # 2. Read Reference Image
    ref_b64 = None
    ref_mime = None
    
    wk_data = meta_entry.get("World_Knowledge", {})
    ref_img_rel_path = ""
    
    if isinstance(wk_data, dict):
        img_field = wk_data.get("image", "")
        if isinstance(img_field, list) and len(img_field) > 0:
            ref_img_rel_path = img_field[0]
        elif isinstance(img_field, str):
            ref_img_rel_path = img_field
    elif isinstance(wk_data, str) and (wk_data.endswith('.jpg') or wk_data.endswith('.png')):
         ref_img_rel_path = wk_data
    
    if ref_img_rel_path:
        full_ref_path = os.path.join(REFERENCE_IMAGE_ROOT, ref_img_rel_path)
        ref_b64 = encode_image(full_ref_path)
        ref_mime = get_mime_type(full_ref_path)

    # 3. Construct Prompt
    user_content = []
    
    checklist_str = "Checklist to Verify:\n"
    for i, item in enumerate(checklist):
        checklist_str += f"{i+1}. {item}\n"
    user_content.append({"type": "text", "text": checklist_str})
    
    user_content.append({"type": "text", "text": "--- Generated Image (Target) ---"})
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{gen_mime};base64,{gen_b64}"}
    })
    
    if ref_b64:
        user_content.append({"type": "text", "text": "--- Reference Image (Ground Truth) ---"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"}
        })
    
    # 4. Call API
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = clean_json_string(response.choices[0].message.content)
        api_result = json.loads(content)
        
        bool_results = api_result.get("results", [])
        
        # Ensure result length matches Checklist length
        if len(bool_results) < len(checklist):
            bool_results.extend([False] * (len(checklist) - len(bool_results)))
        elif len(bool_results) > len(checklist):
            bool_results = bool_results[:len(checklist)]
            
        result_entry["Checklist_results"] = bool_results
        result_entry["Eval"] = all(bool_results) and len(bool_results) > 0

    except Exception as e:
        print(f"API Error for {img_filename}: {e}")
        result_entry["Checklist_results"] = [False] * len(checklist)
        result_entry["Eval"] = False
        
    return result_entry

# ===================== Main Function =====================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load metadata dictionary
    meta_map = load_metadata_map(META_JSONL_PATH)
    if not meta_map:
        return

    # 2. Scan images to evaluate
    model_name = os.path.basename(GENERATED_IMAGES_DIR.rstrip("\\/"))
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
        print(f"Resume run detected: {len(processed_images)} images already processed, will skip them.")
    
    print(f"Scanning generated images folder: {GENERATED_IMAGES_DIR}")
    tasks = []
    
    for root, dirs, files in os.walk(GENERATED_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                if file in processed_images:
                    continue
                prefix, id_str = parse_filename(file)
                if prefix and id_str:
                    key = (prefix, id_str)
                    if key in meta_map:
                        full_img_path = os.path.join(root, file)
                        tasks.append((file, full_img_path, meta_map[key]))
    
    print(f"Found {len(tasks)} images matching Checklists. Starting evaluation...")
    
    # 3. Multi-threaded execution and saving
    with open(output_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(evaluate_single_image, t): t[0] for t in tasks}
            
            for i, future in tqdm(enumerate(as_completed(future_to_file)), total=len(tasks)):
                result = future.result()
                
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                if (i + 1) % SAVE_INTERVAL == 0:
                    f.flush()

    print(f"Evaluation completed! Results saved to: {output_path}")

if __name__ == "__main__":
    main()
