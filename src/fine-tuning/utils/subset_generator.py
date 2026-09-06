import json
import os
import random
from pathlib import Path

def extract_subset(input_path, sample_limit=100, seed=42):
    """
    Extracts a deterministic subset of an evaluation dataset in memory to merge into
    a universal mult-dataset instruction tuning array.
    """
    random.seed(seed)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[!] Critical Error reading {input_path}: {e}")
        return []

    random.shuffle(lines)
    return lines[:sample_limit]

if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    BASE_TEST_DIR = PROJECT_ROOT / "voqa_test_benchmark" / "test"
    OUTPUT_FT_DIR = PROJECT_ROOT / "src" / "fine-tuning" / "data" / "subsets"
    
    # Mapping the 5 available datasets strictly requested
    targets = {
        "gqa": "llava_gqa_testdev_balanced.jsonl",
        "pope": "llava_pope_test.jsonl",
        "scienceqa": "llava_test_CQM-A_selected_mm.jsonl",
        "textvqa": "llava_textvqa_val_v051_ocr_new_id_without_ocr_reference.jsonl",
        "vqav2": "llava_vqav2_mscoco_test-dev2015.jsonl"
    }
    
    universal_mix = []
    
    print("[*] Building 500-sample Universal Visual Reasoner Core...")
    for dataset, filename in targets.items():
        in_path = BASE_TEST_DIR / dataset / filename
        subset = extract_subset(str(in_path), sample_limit=100)
        universal_mix.extend(subset)
        print(f"   => Imported {len(subset)} samples from {dataset}")
        
    out_path = OUTPUT_FT_DIR / "universal_cross_dataset_500.jsonl"
    os.makedirs(OUTPUT_FT_DIR, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in universal_mix:
            f.write(line)
            
    print(f"[*] Successfully generated Instruction-Tuned array with {len(universal_mix)} total samples at {out_path}")
