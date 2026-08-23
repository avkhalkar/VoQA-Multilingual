import sys
import json
import random
import argparse
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_MULTILINGUAL_DIR, DATASETS, FONT_PATH, PROJECT_ROOT
from src.utils.rendering_utils import download_font, generate_text_image

def concat_resizing(is_img, iq_img, direction):
    """
    Concatenation with resizing (VoQA Appendix A.1). 
    Fixes the larger dimension and enlarges the smaller to dynamically map aspect sizes.
    """
    # direction: 0:Top, 1:Bottom, 2:Left, 3:Right
    w_s, h_s = is_img.size
    w_q, h_q = iq_img.size
    
    if direction in [0, 1]:  # Vertical concat (match widths)
        # VoQA Appendix A.1: Fix the larger size in Is and Iq, and enlarge the smaller size to the same side length.
        target_w = max(w_s, w_q)
        if w_s < target_w:
            new_h = int(h_s * (target_w / w_s))
            is_img = is_img.resize((target_w, new_h), Image.LANCZOS)
        if w_q < target_w:
            new_h = int(h_q * (target_w / w_q))
            iq_img = iq_img.resize((target_w, new_h), Image.LANCZOS)
            
        concat_img = Image.new('RGB', (target_w, is_img.size[1] + iq_img.size[1]))
        if direction == 0: # Iq on Top
            concat_img.paste(iq_img, (0, 0))
            concat_img.paste(is_img, (0, iq_img.size[1]))
        else: # Iq on Bottom
            concat_img.paste(is_img, (0, 0))
            concat_img.paste(iq_img, (0, is_img.size[1]))
    else:  # Horizontal concat (match heights)
        # VoQA Appendix A.1: Fix the larger size in Is and Iq, and enlarge the smaller size to the same side length.
        target_h = max(h_s, h_q)
        if h_s < target_h:
            new_w = int(w_s * (target_h / h_s))
            is_img = is_img.resize((new_w, target_h), Image.LANCZOS)
        if h_q < target_h:
            new_w = int(w_q * (target_h / h_q))
            iq_img = iq_img.resize((new_w, target_h), Image.LANCZOS)
            
        concat_img = Image.new('RGB', (is_img.size[0] + iq_img.size[0], target_h))
        if direction == 2: # Left
            concat_img.paste(iq_img, (0, 0))
            concat_img.paste(is_img, (iq_img.size[0], 0))
        else: # Right
            concat_img.paste(is_img, (0, 0))
            concat_img.paste(iq_img, (is_img.size[0], 0))
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
    rendered_dir = DATA_MULTILINGUAL_DIR / f"rendered_concat_resizing_{args.dataset}"
    
    target_langs = ['eng', 'ita', 'fin', 'nld', 'spa']
    for lang in target_langs:
        if lang == 'eng':
            lang_jsonl = PROJECT_ROOT / "data_multilingual" / "subsets" / "subset_1k.jsonl"
        else:
            lang_jsonl = translations_dir / lang / "questions.jsonl"
            
        if not lang_jsonl.exists(): continue
        
        out_lang_dir = rendered_dir / lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Rendering Concat-Resizing ({args.dataset.upper()}) for {lang.upper()} using Extensible API...")
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
                        final_img = concat_resizing(img, iq_img, direction)
                        final_img.save(out_path, quality=95)
                except Exception as e:
                    pass

if __name__ == "__main__":
    main()
