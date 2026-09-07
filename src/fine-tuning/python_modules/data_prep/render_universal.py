import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import textwrap
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(root_dir))

try:
    from src.config import FONT_PATH
    from src.utils.rendering_utils import download_font, score_region, get_watermark_color
except ImportError:
    print("[!] Native renderer dependencies unresolved. Verify root layout.")
    sys.exit(1)

def draw_watermark_voqa(img, text, font_path):
    clean_text = text.split('?')[0] + '?' if '?' in text else text
    img_np = np.array(img)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    h_img, w_img = img_np.shape[:2]
    w_size = max(40, min(h_img, w_img) // 4)
    stride = w_size
    
    best_score = float('inf')
    best_x, best_y = 0, 0
    best_patch = None
    
    for y in range(0, h_img - w_size + 1, stride):
        for x in range(0, w_img - w_size + 1, stride):
            patch = img_gray[y:y+w_size, x:x+w_size]
            score = score_region(patch)
            if score < best_score:
                best_score = score
                best_x, best_y = x, y
                best_patch = img_np[y:y+w_size, x:x+w_size]
                
    text_color = get_watermark_color(best_patch)
    optimal_wrap = max(20, int(len(clean_text) ** 0.5 * 2.5))
    wrapped_text = "\n".join(textwrap.wrap(clean_text, width=optimal_wrap))
    
    font_size = max(14, min(w_img, h_img) // 18)
    font = ImageFont.truetype(str(font_path), size=font_size)
    dummy_draw = ImageDraw.Draw(img)
    
    bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, stroke_width=2)
    
    while font_size > 10 and (bbox[2] - bbox[0] > w_img - 20 or bbox[3] - bbox[1] > h_img // 2):
        font_size -= 2
        font = ImageFont.truetype(str(font_path), size=font_size)
        bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, stroke_width=2)
        
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    offset_x = best_x + (w_size - text_w) // 2 - bbox[0]
    offset_y = best_y + (w_size - text_h) // 2 - bbox[1]
    offset_x = max(5 - bbox[0], min(offset_x, w_img - bbox[2] - 5))
    offset_y = max(5 - bbox[1], min(offset_y, h_img - bbox[3] - 5))
    
    draw = ImageDraw.Draw(img)
    stroke_c = (255, 255, 255) if sum(text_color) / 3 < 127 else (0, 0, 0)
    draw.multiline_text((offset_x, offset_y), wrapped_text, font=font, fill=text_color, spacing=4, stroke_width=2, stroke_fill=stroke_c)
    return img

def _worker_process(args):
    """Isolated pure multiprocessing worker for parallel image scaling."""
    q, raw_path_str, out_path_str, font_path_str = args
    out_path = Path(out_path_str)
    raw_path = Path(raw_path_str)
    
    if out_path.exists() or not raw_path.exists():
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if 'text' in q:
        text_layer = q['text']
    elif 'conversations' in q:
        human_conv = next((c for c in q['conversations'] if c['from'] == 'human'), None)
        if human_conv:
            text_layer = human_conv['value'].replace('<image>\n', '').replace('\n<image>', '')
        else: return False
    else:
        return False
        
    try:
        with Image.open(raw_path) as img:
            img = img.convert("RGB")
            watermarked_img = draw_watermark_voqa(img, text_layer, font_path_str)
            watermarked_img.save(out_path, quality=95)
        return True
    except Exception as e:
        return f"Error: {e}"

def main():
    print("======================================================================")
    print("  [Stage 0.2: Extreme Capacity Multiplex Image Renderer]")
    print("======================================================================")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default="universal_cross_dataset_500.jsonl")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="CPU cores to bind")
    args = parser.parse_args()

    if not download_font(FONT_PATH):
        sys.exit(1)

    subset_dir = root_dir / "src" / "fine-tuning" / "data" / "subsets"
    base_image_dir = root_dir / "voqa_test_benchmark" / "test"
    target_langs = ['ita', 'fin', 'nld', 'spa']

    for lang in target_langs:
        lang_jsonl = subset_dir / lang / args.input_jsonl
        render_target_dir = base_image_dir / lang / "cross_dataset_rerendered"
        
        if not lang_jsonl.exists():
            continue

        print(f"\n>>> Compiling Parallel Matrix for Target: {lang.upper()} using {args.workers} CPU Cores")
        
        tasks = []
        with open(lang_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                q = json.loads(line)
                if 'image' not in q: continue
                
                raw_path = base_image_dir / q['image']
                out_path = render_target_dir / q['image']
                tasks.append((q, str(raw_path), str(out_path), str(FONT_PATH)))

        success_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_worker_process, task) for task in tasks]
            for i, future in enumerate(as_completed(futures)):
                res = future.result()
                if res is True: success_count += 1
                if (i + 1) % 100 == 0:
                    print(f"    ... {i+1} total pixel layers verified/rendered.")

        print(f"    [*] Completed. Net parallel renderings this cycle: {success_count}")

    print("\n[*] Extreme Scaling Render Engine Terminated Successfully.")

if __name__ == "__main__":
    main()
