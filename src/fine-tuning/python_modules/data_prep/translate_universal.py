import os
import json
import sys
import argparse
from pathlib import Path
import torch

try:
    from seamless_communication.inference import Translator
except ImportError:
    print("[!] Error: You need to install seamless_communication first before executing translation on the cluster.")
    sys.exit(1)

def chunked_iterable(iterable, size):
    """Yield successive n-sized chunks from an iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def main():
    print("======================================================================")
    print("  [Stage 0.1: Massively Scalable Batched Translation API]")
    print("======================================================================")
    
    # Optional argparse for passing massive explicit dataset targets
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default="universal_cross_dataset_500.jsonl")
    parser.add_argument("--batch_size", type=int, default=32, help="Translation GPU Batch Size")
    args = parser.parse_args()

    # Dynamically resolve root safely for SLURM
    root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    subset_dir = root_dir / "src" / "fine-tuning" / "data" / "subsets"
    
    input_jsonl = subset_dir / args.input_jsonl
    
    if not input_jsonl.exists():
        print(f"[!] Critical Error: Input array missing at {input_jsonl}")
        sys.exit(1)

    target_langs = ['ita', 'fin', 'nld', 'spa']

    print("[*] Allocating VRAM for SeamlessM4T Native Translator Array...")
    translator = Translator(
        "seamlessM4T_v2_large",
        "vocoder_v2",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float16,
    )

    for lang in target_langs:
        out_lang_dir = subset_dir / lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = out_lang_dir / args.input_jsonl

        # Extreme-Scale Resume Check
        processed_ids = set()
        if out_jsonl.exists():
            with open(out_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            q_data = json.loads(line)
                            raw_id = q_data.get('question_id') or q_data.get('id')
                            if raw_id is not None:
                                processed_ids.add(str(raw_id))
                        except Exception:
                            continue

        print(f"\n>>> Executing High-Throughput Batch Translation for Target: {lang.upper()} (Already Processed: {len(processed_ids)})")
        
        # Generator for 3.35M lines to prevent RAM exhaustion
        def record_generator():
            with open(input_jsonl, 'r', encoding='utf-8') as f_in:
                for i, line in enumerate(f_in):
                    if not line.strip(): continue
                    q = json.loads(line)
                    raw_id = q.get('question_id') or q.get('id') or f"synthetic_id_{i}"
                    # Skip already processed
                    if str(raw_id) in processed_ids:
                        continue
                    yield (q, str(raw_id))

        with open(out_jsonl, 'a', encoding='utf-8') as f_out:
            for batch in chunked_iterable(record_generator(), args.batch_size):
                # Prepare extraction vectors
                batch_texts = []
                batch_indices = []
                
                # Intelligent schema extraction for heterogeneous tensors
                for idx, (q, qid) in enumerate(batch):
                    text_content = None
                    if 'text' in q:
                        text_content = q['text']
                    elif 'conversations' in q:
                        human_conv = next((c for c in q['conversations'] if c['from'] == 'human'), None)
                        if human_conv:
                            text_content = human_conv['value'].replace('<image>\n', '').replace('\n<image>', '')
                    
                    if text_content:
                        batch_texts.append(text_content[:4096])
                        batch_indices.append(idx)
                
                if not batch_texts:
                    continue
                
                # Execute Parallel GPU Batch Predict Array
                try:
                    text_outputs, _ = translator.predict(
                        input=batch_texts,
                        task_str="t2tt",
                        tgt_lang=lang,
                        src_lang="eng"
                    )
                except Exception as e:
                    print(f"    [!] Fatal GPU Batch Translation Error: {e}")
                    # Fallback on failure logic could go here
                    continue
                
                # Reconstruct and dispatch heterogeneous JSON elements
                for out_idx_str, b_idx in enumerate(batch_indices):
                    q, _ = batch[b_idx]
                    translated_val = str(text_outputs[out_idx_str])
                    
                    if 'text' in q:
                        q['text'] = translated_val
                    elif 'conversations' in q:
                        for conv in q['conversations']:
                            if conv['from'] == 'human':
                                orig = conv['value']
                                conv['value'] = f"<image>\n{translated_val}" if "<image>" in orig else translated_val
                                break
                                
                    q['language'] = lang
                    f_out.write(json.dumps(q, ensure_ascii=False) + "\n")
                
                f_out.flush()

    print("\n[*] Translation Engine Terminated Successfully. Subsets prepared for rendering.")

if __name__ == "__main__":
    main()
