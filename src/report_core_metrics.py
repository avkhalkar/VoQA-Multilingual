import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gqa", help="Target dataset mapped in src/config.py")
    parser.add_argument("--model", type=str, default="internvl_1b", help="Target architecture mapped in src/config.py")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    results_path = root_dir / "output" / args.model / args.dataset / "evaluation_results.json"
    
    if not results_path.exists():
        print(f"evaluation_results.json not found structurally at {results_path}!")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    target_langs = ['eng', 'ita', 'fin', 'nld', 'spa']
    
    print(f"# Core Semantic Metrics ({args.dataset.upper()})\n")
    print(f"**Model:** {args.model.upper()}\n")
    
    for lang in target_langs:
        lang_data = data.get(lang, {})
        if not lang_data: continue
        
        print(f"### {lang.upper()} Evaluation Summary")
        print("| Prompt Configuration | Exact Match | Soft Match (BERT >= 0.90) | Average F1 | Average QAA Score (Correct) |")
        print("|----------------------|-------------|---------------------------|------------|-----------------------------|")
        
        for prompt_key, p_data in lang_data.items():
            em = p_data.get("hard_accuracy", 0.0)
            soft = p_data.get("semantic_accuracy", 0.0)
            avg_f1 = p_data.get("f1", 0.0)
            qaa_c = p_data.get("qaa_correct", 0.0)
            
            prompt_fmt = prompt_key.replace("prompt_", "").replace("_", " ").title()
            print(f"| {prompt_fmt.ljust(20)} | {em:05.2f}%     | {soft:05.2f}%                   | {avg_f1:.4f}     | {qaa_c:05.2f}%                      |")
        print("\n")

if __name__ == "__main__":
    main()
