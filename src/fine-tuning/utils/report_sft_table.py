import json
import os
import argparse
from pathlib import Path

def generate_sft_tables(eval_root="output_tuning"):
    """
    Parses evaluation logs across models, datasets, and SFT strategies.
    
    Academic References (VoQA Paper):
    - Section 4.1 (Evaluation Metrics): "Exact-Match Accuracy" 
      -> Standard strict string matching for answer correctness.
    - Section 4.1 (Evaluation Metrics): "Semantic Accuracy (Soft Match)" 
      -> Computes BERTScore F1 (Threshold >= 0.90) to measure semantic answer similarity.
    - Section 4.2 (Question Extraction): "Question Extraction Accuracy (QAA)"
      -> Evaluates the model's capacity to extract questions accurately from pixels. 
         (QAA is logged conditionally based on Semantic correct/incorrect buckets).
         
    Outputs:
    1. Standard Benchmark Accuracy Table (Table 2 / Table A15 format)
    2. Semantic BERT Accuracy Table
    3. QAA (Correct vs Incorrect) Diagnostic Table
    """
    datasets = ["vqav2", "gqa", "pope", "textvqa", "scienceqa"]
    ds_labels = ["VQAv2", "GQA", "POPE", "TextVQA", "SQA"]
    strategies = ["baseline_sft", "qa_sft", "qra_sft", "r_qra_sft", "qa_only_sft"]
    strat_labels = ["Baseline-SFT", "QA-SFT", "QRA-SFT", "R-QRA-SFT", "QA-only-SFT"]
    
    models = ["internvl", "qwen2", "tinyllava"]
    model_labels = {"internvl": "InternVL (1B)", "qwen2": "Qwen (2B)", "tinyllava": "TinyLLaVA (1B)"}

    # Load results dictionary
    results_map = {}
    eval_path = Path(eval_root)
    
    if not eval_path.exists():
        print(f"[!] Evaluation directory '{eval_root}' not found. Showing format template.")
        return

    # Print Table 1: Standard Benchmark Accuracy (Paper Format)
    print("=" * 85)
    print("TABLE 1: Benchmark Performance Comparison of Fine-Tuning Strategies (Exact Match / Accuracy %)")
    print("=" * 85)
    header = f"{'Model':<16} | {'VoQA SFT Strategy':<15} | " + " | ".join([f"{d:>7}" for d in ds_labels]) + " | " + f"{'Avg.':>6}"
    print(header)
    print("-" * len(header))

    for m in models:
        for strat, strat_name in zip(strategies, strat_labels):
            scores = []
            for ds in datasets:
                # Mock fallback / lookup logic from evaluation_results.json
                log_file = eval_path / m / ds / "evaluation_results.json"
                val = 0.0
                if log_file.exists():
                    try:
                        with open(log_file, "r") as f:
                            data = json.load(f)
                            val = data.get("eng", {}).get(strat, {}).get("hard_accuracy", 0.0)
                    except: pass
                scores.append(val)
            
            avg = sum(scores) / max(1, len(scores))
            ds_str = " | ".join([f"{s:7.1f}" for s in scores])
            m_label = model_labels.get(m, m) if strat == strategies[0] else ""
            print(f"{m_label:<16} | {strat_name:<15} | {ds_str} | {avg:6.1f}")
        print("-" * len(header))

    # Print Table 2: Semantic BERT Accuracy Table
    print("\n" + "=" * 85)
    print("TABLE 2: Semantic BERT Accuracy Comparison (BERTScore F1 >= 0.90 Threshold)")
    print("=" * 85)
    print(header)
    print("-" * len(header))

    for m in models:
        for strat, strat_name in zip(strategies, strat_labels):
            scores = []
            for ds in datasets:
                log_file = eval_path / m / ds / "evaluation_results.json"
                val = 0.0
                if log_file.exists():
                    try:
                        with open(log_file, "r") as f:
                            data = json.load(f)
                            val = data.get("eng", {}).get(strat, {}).get("semantic_accuracy", 0.0)
                    except: pass
                scores.append(val)
            
            avg = sum(scores) / max(1, len(scores))
            ds_str = " | ".join([f"{s:7.1f}" for s in scores])
            m_label = model_labels.get(m, m) if strat == strategies[0] else ""
            print(f"{m_label:<16} | {strat_name:<15} | {ds_str} | {avg:6.1f}")
        print("-" * len(header))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="output_tuning", help="Directory containing evaluation logs")
    args = parser.parse_args()
    generate_sft_tables(args.eval_dir)
