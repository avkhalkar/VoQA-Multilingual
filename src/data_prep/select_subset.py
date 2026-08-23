import os
import json
import random
from pathlib import Path

def main():
    # Define root and paths relatively
    # Script is in src/data_prep/, so root is two levels up
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    
    gqa_jsonl = root_dir / "voqa_gqa" / "test" / "gqa" / "llava_gqa_testdev_balanced.jsonl"
    gqa_dir = root_dir / "voqa_gqa" / "test" / "gqa"
    images_dir = gqa_dir / "images"
    out_dir = root_dir / "data_multilingual" / "subsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subset_out_path = out_dir / "subset_1k.jsonl"
    ids_out_path = out_dir / "pilot_1k_ids.json"
    
    print(f"Loading full dataset from {gqa_jsonl}...")
    questions = []
    with open(gqa_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
                
    print(f"Total questions found: {len(questions)}")
    
    # Deterministic selection
    random.seed(42)
    random.shuffle(questions)
    
    selected_questions = []
    missing_images = 0
    
    for q in questions:
        image_name = q.get('image')
        image_path = images_dir / image_name
        
        # Verify the raw image actually exists in the local directory
        if image_path.exists():
            selected_questions.append(q)
        else:
            missing_images += 1
            
        if len(selected_questions) == 100:
            break
            
    if missing_images > 0:
        print(f"Warning: Skipped {missing_images} candidate questions because their underlying images were missing from {images_dir}")

    print(f"Successfully selected {len(selected_questions)} valid questions.")
    
    # Save the full subset data
    with open(subset_out_path, 'w', encoding='utf-8') as f:
        for q in selected_questions:
            f.write(json.dumps(q) + "\n")
            
    # Save just the IDs for easy reference
    selected_ids = [q['question_id'] for q in selected_questions]
    with open(ids_out_path, 'w', encoding='utf-8') as f:
        json.dump(selected_ids, f, indent=2)
        
    print(f"Saved {subset_out_path}")
    print(f"Saved {ids_out_path}")

if __name__ == "__main__":
    main()
