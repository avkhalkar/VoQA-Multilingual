import sys
import json
import random
import shutil
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.rendering_utils import download_font, generate_text_image
from src.data_prep.render_watermarks import draw_watermark_voqa
from src.data_prep.render_concatenation_with_resizing import concat_resizing
from src.data_prep.render_concatenation_without_resizing import concat_no_resizing
from src.config import FONT_PATH, DATASETS, PROJECT_ROOT

def main():
    smoke_dir = PROJECT_ROOT / "tools" / "smoke_test_output"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
        
    eng_json = PROJECT_ROOT / "data_multilingual" / "subsets" / "subset_1k.jsonl"
    langs = [("eng", eng_json)]
    for lang in ["fin", "ita", "nld", "spa"]:
        langs.append((lang, PROJECT_ROOT / "data_multilingual" / "translations" / lang / "questions.jsonl"))
        
    raw_images_dir = Path(DATASETS["gqa"]["images"])
    
    download_font(FONT_PATH)
    
    for lang, json_path in langs:
        if not json_path.exists(): continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]
            
        random.seed(42)
        sample_lines = random.sample(lines, min(100, len(lines)))
        print(f"Executing Smoke Test: 100 samples for {lang.upper()}...", flush=True)
        
        for i, line in enumerate(sample_lines):
            try:
                q = json.loads(line)
                qid, img_name, text = q['question_id'], q.get('image', ''), q.get('text', '')
                raw_path = raw_images_dir / img_name
                
                if not raw_path.exists(): continue
                img = Image.open(raw_path).convert("RGB")
                
                wm_img = draw_watermark_voqa(img.copy(), text, FONT_PATH)
                wm_out = smoke_dir / lang / "watermark" / f"{qid}.jpg"
                wm_out.parent.mkdir(parents=True, exist_ok=True)
                wm_img.save(wm_out, quality=90)
                
                iq_img = generate_text_image(text, FONT_PATH)
                
                cr_img = concat_resizing(img.copy(), iq_img, direction=0)
                cr_out = smoke_dir / lang / "concat_resize" / f"{qid}.jpg"
                cr_out.parent.mkdir(parents=True, exist_ok=True)
                cr_img.save(cr_out, quality=90)
                
                cp_img = concat_no_resizing(img.copy(), iq_img, direction=0)
                cp_out = smoke_dir / lang / "concat_padding" / f"{qid}.jpg"
                cp_out.parent.mkdir(parents=True, exist_ok=True)
                cp_img.save(cp_out, quality=90)
                
            except Exception as e:
                pass
    print("Execution complete! You can view the files now in tools/smoke_test_output", flush=True)

if __name__ == "__main__":
    main()
