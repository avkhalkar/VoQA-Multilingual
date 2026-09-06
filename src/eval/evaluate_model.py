import sys
import json
import torch
import re
import argparse
from pathlib import Path

"""
Academic Evaluation Implementation (Ref: VoQA Paper)
- Section 4.1 (Evaluation Metrics): Exact Match (EM) Accuracy.
- Section 4.1 (Evaluation Metrics): Semantic Match via BERTScore (F1 >= 0.90).
- Section 4.2 (Question Extraction): Question Extraction Accuracy (QAA).
  Note: QAA is evaluated heavily filtering on Semantic Match (Soft Match) 
  rather than Exact Match to prevent catastrophic QAA structural degradation
  due to conversational padding in VLMs.
"""

# Ensure package context mapping
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from seamless_communication.inference import Translator
from src.config import PROJECT_ROOT, DATASETS, MODELS
from src.eval.metrics import compute_semantic_score
from src.eval.qaa_metrics import calculate_qaa_score
from src.utils.response_pipeline import run_response_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gqa", help="Target dataset mapped in config.py")
    parser.add_argument("--model", type=str, default="internvl_1b", help="Target model mapped in config.py")
    parser.add_argument("--lang", type=str, default="all", help="Target execution language recursively natively")
    args = parser.parse_args()

    if args.dataset not in DATASETS: raise ValueError(f"Dataset {args.dataset} not found in config.py registry.")
    if args.model not in MODELS: raise ValueError(f"Model {args.model} not found in config.py registry.")

    # Dynamic Path Setup
    gt_file = DATASETS[args.dataset]["gt_questions"]
    output_dir = PROJECT_ROOT / "output" / args.model / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = output_dir / "evaluation_results.json"
    
    target_langs = ["eng", "ita", "fin", "nld", "spa"] if args.lang == "all" else [args.lang]
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    print(f"[{args.model.upper()} | {args.dataset.upper()}] Loading SeamlessM4T Back-Translation...")
    translator = Translator("seamlessM4T_v2_large", "vocoder_v2", device=torch.device("cuda"), dtype=torch.float16)
    
    # target_langs overriden implicitly by arg parser
    
    results = {}
    if metrics_log.exists():
        with open(metrics_log, 'r', encoding='utf-8') as f:
            try: results = json.load(f)
            except: pass
            
    for lang in target_langs:
        lang_dir = output_dir / lang
        if not lang_dir.exists(): continue
        
        if lang not in results: results[lang] = {}
        for pred_jsonl in lang_dir.glob("results_*.jsonl"):
            config_name = pred_jsonl.stem.replace("results_", "")
            
            predictions = [json.loads(line) for line in open(pred_jsonl, 'r', encoding='utf-8') if line.strip()]
            if not predictions: continue
                
            exact_match_count, cands, refs, evaluated_out = 0, [], [], []
            for p in predictions:
                qid = str(p['question_id'])
                clean_pred, q_hat, eng_pred = run_response_pipeline(p['model_prediction'], config_name, lang, translator)
                
                gt = gt_data.get(qid, {})
                gt_answer = gt.get('answer', "").strip().lower()
                gt_question = gt.get('question', "").strip()
                
                # Math abstracted to metrics api
                qaa_score = calculate_qaa_score(q_hat, gt_question)
                is_exact = (clean_pred == gt_answer)
                if is_exact: exact_match_count += 1
                
                cands.append(clean_pred if clean_pred.strip() else "<MISSING>")
                refs.append(gt_answer if gt_answer.strip() else "<MISSING>")
                
                row = p.copy()
                row.update({"eng_pred": eng_pred, "exact_match": is_exact, "qaa_score": qaa_score * 100})
                evaluated_out.append(row)
                
            print(f"[{lang.upper()} | {config_name}] Initiating BERTScore Evaluation Pipeline...")
            is_match, avg_f1, bert_acc, f1_list = compute_semantic_score(cands, refs, semantic_threshold=0.90)
            
            qaa_corrects, qaa_incorrects = [], []
            for i, eo in enumerate(evaluated_out):
                eo["bert_f1"] = f1_list[i]
                eo["semantic_match"] = is_match[i]
                
                # VoQA Decision: Bucket QAA heavily using strictly Semantic matches (Soft Match). 
                # This guarantees we do not destroy QAA structural scores just because 
                # the VLM output conversational padding instead of strict exact-length matching!
                if is_match[i]: qaa_corrects.append(eo["qaa_score"])
                else: qaa_incorrects.append(eo["qaa_score"])
                
            em_accuracy = exact_match_count / len(predictions) * 100
            qaa_c = sum(qaa_corrects) / max(1, len(qaa_corrects))
            qaa_i = sum(qaa_incorrects) / max(1, len(qaa_incorrects))
            
            evaluated_jsonl = pred_jsonl.parent / f"evaluated_{config_name}.jsonl"
            with open(evaluated_jsonl, 'w', encoding='utf-8') as ef:
                for eo in evaluated_out: ef.write(json.dumps(eo, ensure_ascii=False) + "\n")
                
            print(f"====> Semantic: {bert_acc:.2f}% | Exact: {em_accuracy:.2f}% | QAA(C): {qaa_c:.2f}% | QAA(I): {qaa_i:.2f}%")
            results[lang][config_name] = {
                "hard_accuracy": em_accuracy, "semantic_accuracy": bert_acc,
                "qaa_correct": qaa_c, "qaa_incorrect": qaa_i, "f1": avg_f1
            }
    
    with open(metrics_log, 'w') as f: json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
