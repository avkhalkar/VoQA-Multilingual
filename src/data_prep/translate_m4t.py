import os
import json
import sys
import argparse
from pathlib import Path
import torch
from seamless_communication.inference import Translator

def translate_sent(translator, sentence, tgt_lang, src_lang="eng"):
    """
    Adapter function that invokes the SeamlessM4T pipeline.
    """
    if not sentence:
        return ""
    text_output, _ = translator.predict(
        input=sentence[:4096],
        task_str="t2tt",
        tgt_lang=tgt_lang,
        src_lang=src_lang
    )
    return str(text_output[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="all", help="Target language to translate (e.g. 'ita') or 'all'")
    args = parser.parse_args()

    # Identify repo root relatively
    root_dir = Path(__file__).resolve().parent.parent.parent
    subset_jsonl = root_dir / "data_multilingual" / "subsets" / "subset_1k.jsonl"
    out_dir = root_dir / "data_multilingual" / "translations"

    if not subset_jsonl.exists():
        print(f"Error: Could not find {subset_jsonl}. Have you run select_subset.py?")
        sys.exit(1)

    print(f"Loading input JSONL: {subset_jsonl}")
    questions = []
    with open(subset_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
                
    # Our restricted subset mapped for Latin scripts evaluation
    all_target_langs = ['ita', 'fin', 'nld', 'spa']
    if args.lang == "all":
        target_langs = all_target_langs
    elif args.lang == "eng":
        print("English is the source language — no translation needed.")
        return
    elif args.lang in all_target_langs:
        target_langs = [args.lang]
    else:
        print(f"Error: Unknown language '{args.lang}'. Valid options: {all_target_langs} or 'all'.")
        sys.exit(1)

    print("Loading SeamlessM4T v2 Large. This will take ~3GB VRAM...")
    translator = Translator(
        "seamlessM4T_v2_large",
        "vocoder_v2",
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    print("Model loaded successfully.")

    for lang in target_langs:
        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = lang_dir / "questions.jsonl"

        # Enable Resumability: Track what we already translated
        processed_ids = set()
        if out_jsonl.exists():
            with open(out_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        processed_ids.add(json.loads(line)['question_id'])

        print(f"\nProcessing target language: {lang.upper()} (Already done: {len(processed_ids)}/{len(questions)})")
        
        with open(out_jsonl, 'a', encoding='utf-8') as f:
            for i, q in enumerate(questions):
                qid = q['question_id']
                if qid in processed_ids:
                    continue

                original_text = q['text']
                
                try:
                    translated_text = translate_sent(translator, original_text, tgt_lang=lang)
                except Exception as e:
                    print(f"Error translating question ID {qid}: {e}")
                    continue

                # Prepare the output row identical to the input but with translated text and language flag
                q_out = q.copy()
                q_out['text'] = translated_text
                q_out['language'] = lang

                f.write(json.dumps(q_out, ensure_ascii=False) + "\n")
                f.flush()
                
                if (i + 1) % 100 == 0:
                    print(f"  ... translated {i+1} rows.")
                    
    print("\nMultilingual translation pipeline complete. You can now build watermarks out of these JSONL files.")

if __name__ == "__main__":
    main()
