import os
import sys
import json
from pathlib import Path
from PIL import Image

sys.path.append(str(Path("src/data_prep").resolve()))
from render_watermarks import draw_watermark_voqa
from render_concatenation_with_resizing import concat_resizing, generate_text_image
import render_concatenation_without_resizing as no_resize

def main():
    root = Path(".")
    json_path = root / "data_multilingual/subsets/subset_1k.jsonl"
    
    with open(json_path) as f:
        q = json.loads(f.readline())
        
    raw_path = root / "voqa_gqa/test/gqa/images" / q['image']
    font_path = root / "src/data_prep/DejaVuSans-Bold.ttf"

    text = q['text']
    print(f"Testing on QID {q['question_id']}\nText: {text}")
    print(f"Source Image: {raw_path}")

    img = Image.open(raw_path).convert("RGB")
    
    # 1. Watermark
    print("Testing Watermark logic...")
    img1 = draw_watermark_voqa(img.copy(), text, font_path)
    img1.save("sandbox_watermark.jpg")
    print("-> sandbox_watermark.jpg")

    # 2. Concat with Resizing
    print("Testing Concat-Resizing logic...")
    iq = generate_text_image(text, font_path)
    img2 = concat_resizing(img.copy(), iq, direction=0)
    img2.save("sandbox_concat_resize.jpg")
    print("-> sandbox_concat_resize.jpg")

    # 3. Concat without Resizing
    print("Testing Concat-Padding logic...")
    img3 = no_resize.concat_no_resizing(img.copy(), iq, direction=0)
    img3.save("sandbox_concat_padding.jpg")
    print("-> sandbox_concat_padding.jpg")
    
    print("All tests passed.")

if __name__ == "__main__":
    main()
