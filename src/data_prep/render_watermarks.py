import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import textwrap

# Ensure absolute import path resolution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import DATA_MULTILINGUAL_DIR, DATASETS, FONT_PATH, PROJECT_ROOT
from src.utils.rendering_utils import download_font, score_region, get_watermark_color

def draw_watermark_voqa(img, text, font_path):
    """
    Renders VoQA Text Watermark using Contrast-Aware Sliding Windows (Appendix A.2).
    The watermark's side length is set to 1/4 of the short side of the scene image, 
    and candidate regions are generated using a stride of 1/4 the image side length.
    """
    # Strip any appended prompt instructions (e.g., "Answer the question using...")
    # across any language by truncating after the first question mark.
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
    
    # Relax wrap to create natural wide rectangles instead of tall "skyscraper" columns
    optimal_wrap = max(20, int(len(clean_text) ** 0.5 * 2.5))
    wrapped_text = "\n".join(textwrap.wrap(clean_text, width=optimal_wrap))
    
    # Anchor the font smoothly to 1/18th of the image scale so it reads like a watermark instead of a billboard
    font_size = max(14, min(w_img, h_img) // 18)
    font = ImageFont.truetype(str(font_path), size=font_size)
    dummy_draw = ImageDraw.Draw(img)
    
    # Use exact multiline metrics including stroke width to align flawlessly with the drawing renderer
    bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, stroke_width=2)
    
    # Scale down ONLY if it violates the actual image frame boundaries, preserving maximum readability
    while font_size > 10 and (bbox[2] - bbox[0] > w_img - 20 or bbox[3] - bbox[1] > h_img // 2):
        font_size -= 2
        font = ImageFont.truetype(str(font_path), size=font_size)
        bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, stroke_width=2)
        
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    offset_x = best_x + (w_size - text_w) // 2 - bbox[0]
    offset_y = best_y + (w_size - text_h) // 2 - bbox[1]
    
    # Absolute physical constraint: CLAMP offsets perfectly to the true textbbox limits!
    offset_x = max(5 - bbox[0], min(offset_x, w_img - bbox[2] - 5))
    offset_y = max(5 - bbox[1], min(offset_y, h_img - bbox[3] - 5))
    
    draw = ImageDraw.Draw(img)
    # Give the text a deeply contrasting border based strictly on luminance to prevent black-on-black blobs
    stroke_c = (255, 255, 255) if sum(text_color) / 3 < 127 else (0, 0, 0)
    draw.multiline_text((offset_x, offset_y), wrapped_text, font=font, fill=text_color, spacing=4, stroke_width=2, stroke_fill=stroke_c)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gqa", help="Target dataset mapped in config.py")
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        raise ValueError(f"Dataset {args.dataset} not found in config.py registry.")

    raw_images_dir = DATASETS[args.dataset]["images"]
    if not download_font(FONT_PATH):
        return

    translations_dir = DATA_MULTILINGUAL_DIR / "translations"
    rendered_dir = DATA_MULTILINGUAL_DIR / f"rendered_images_watermark_{args.dataset}"
    
    target_langs = ['eng', 'ita', 'fin', 'nld', 'spa']

    for lang in target_langs:
        # Fetch appropriate JSON array
        if lang == 'eng':
             # English baseline split natively provided
            lang_jsonl = PROJECT_ROOT / "data_multilingual" / "subsets" / "subset_1k.jsonl" 
        else:
            lang_jsonl = translations_dir / lang / "questions.jsonl"
            
        if not lang_jsonl.exists():
            continue

        out_lang_dir = rendered_dir / lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRendering watermarks ({args.dataset.upper()}) for {lang.upper()} using Extensible API...")
        
        with open(lang_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                q = json.loads(line)
                qid, raw_image_filename, text = q['question_id'], q['image'], q['text']
                raw_path = raw_images_dir / raw_image_filename
                out_path = out_lang_dir / raw_image_filename
                
                if out_path.exists() or not raw_path.exists():
                    continue
                    
                try:
                    with Image.open(raw_path) as img:
                        img = img.convert("RGB")
                        watermarked_img = draw_watermark_voqa(img, text, FONT_PATH)
                        watermarked_img.save(out_path, quality=95)
                except Exception as e:
                    print(f"Error processing {qid}: {e}")

if __name__ == "__main__":
    main()
