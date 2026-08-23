import sys
import json
import random
import argparse
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_MULTILINGUAL_DIR, DATASETS, FONT_PATH, PROJECT_ROOT
from src.utils.rendering_utils import download_font, generate_text_image

def concat_no_resizing(is_img, iq_img, direction):
    """
    Concatenation without resizing (VoQA Appendix A.1).
    Aligns image centers and leverages uniform whitespace padding for bounding normalization.
    """
    # direction: 0:Top, 1:Bottom, 2:Left, 3:Right
    w_s, h_s = is_img.size
    w_q, h_q = iq_img.size
    
    if direction in [0, 1]:  # Vertical concat (padding center)
        total_w = max(w_s, w_q)
        total_h = h_s + h_q
        concat_img = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))
        
        # VoQA Appendix A.1: Align the center of the two images and fill blank space with white background padding.
        x_s = (total_w - w_s) // 2
        x_q = (total_w - w_q) // 2
        
        if direction == 0: # Iq on Top
            concat_img.paste(iq_img, (x_q, 0))
            concat_img.paste(is_img, (x_s, h_q))
        else: # Iq on Bottom
            concat_img.paste(is_img, (x_s, 0))
            concat_img.paste(iq_img, (x_q, h_s))
    else:  # Horizontal concat (padding center)
        total_h = max(h_s, h_q)
        total_w = w_s + w_q
        concat_img = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))
        
        # VoQA Appendix A.1: Align the center of the two images and fill blank space with white background padding.
        y_s = (total_h - h_s) // 2
        y_q = (total_h - h_q) // 2
        
        if direction == 2: # Left
            concat_img.paste(iq_img, (0, y_q))
            concat_img.paste(is_img, (w_q, y_s))
        else: # Right
            concat_img.paste(is_img, (0, y_s))
            concat_img.paste(iq_img, (w_s, y_q))
    return concat_img

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
    rendered_dir = DATA_MULTILINGUAL_DIR / f"rendered_concat_padding_{args.dataset}"
    
    target_langs = ['eng', 'ita', 'fin', 'nld', 'spa']

    for lang in target_langs:
        if lang == 'eng':
            lang_jsonl = PROJECT_ROOT / "data_multilingual" / "subsets" / "subset_1k.jsonl"
        else:
            lang_jsonl = translations_dir / lang / "questions.jsonl"
            
        if not lang_jsonl.exists(): continue
        
        out_lang_dir = rendered_dir / lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Rendering Concat-Padding ({args.dataset.upper()}) for {lang.upper()} using Extensible API...")
        with open(lang_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                q = json.loads(line)
                qid, raw_image_filename, text = q['question_id'], q['image'], q['text']
                raw_path = raw_images_dir / raw_image_filename
                out_path = out_lang_dir / raw_image_filename
                
                if out_path.exists() or not raw_path.exists():
                    continue
                    
                random.seed(int(qid))
                direction = random.choice([0, 1, 2, 3]) 
                
                try:
                    with Image.open(raw_path) as img:
                        img = img.convert("RGB")
                        iq_img = generate_text_image(text, FONT_PATH)
                        final_img = concat_no_resizing(img, iq_img, direction)
                        final_img.save(out_path, quality=95)
                except Exception as e:
                    pass

if __name__ == "__main__":
    main()
