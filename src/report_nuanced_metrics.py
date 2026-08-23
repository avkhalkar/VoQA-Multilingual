import json
import subprocess
from pathlib import Path
import re
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gqa", help="Target dataset mapped in config.py")
    parser.add_argument("--model", type=str, default="internvl_1b", help="Target model mapped in config.py")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    exp_dir = root_dir / "output" / args.model / args.dataset
    
    gqa_eval_dir = root_dir / f"voqa_{args.dataset}" / "test" / args.dataset / "eval"
    q_file = gqa_eval_dir / "testdev_balanced_questions.json"
    if not q_file.exists():
        q_file = root_dir / f"voqa_{args.dataset}" / "test" / args.dataset / "testdev_balanced_questions.json"
        
    tmp_pred_file = gqa_eval_dir / "testdev_balanced_predictions.json"
    
    target_langs = ['eng', 'ita', 'fin', 'nld', 'spa']
    
    print(f"# Nuanced Diagnostic Metrics ({args.dataset.upper()})\n")
    print(f"**Model:** {args.model.upper()}\n")
    print("> **Architecture Note:** I noticed that inside your repository, the original authors actually commented out Validity and Plausibility from `eval.py` to prevent it from having to load 500MB+ `sceneGraphs.json` blocks for every evaluation. Therefore, this compiled table captures all the active diagnostics (Chi-Square Distribution, Semantic/Structural Type Accuracies, and Binary/Open variances).\n")
    
    for lang in target_langs:
        lang_dir = exp_dir / lang
        if not lang_dir.exists(): continue
            
        print(f"### {lang.upper()} Nuanced Diagnostics")
        print("| Prompt Config       | Binary | Open  | Dist (Chi)| Compare | Choose | Logic | Query |")
        print("|---------------------|--------|-------|-----------|---------|--------|-------|-------|")
        
        # Dynamically loop over whatever outputs exist natively tracking OCR/Render configurations natively
        for pred_file in lang_dir.glob("evaluated_*.jsonl"):
            prompt = pred_file.stem.replace("evaluated_", "")
            
            # Bridge our output schema into GQA's required schema
            converted_preds = []
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    row = json.loads(line)
                    # Force back-translation for structural evaluation
                    eng_pred = row.get("eng_pred", row["prediction"])
                    converted_preds.append({
                        "questionId": str(row["question_id"]),
                        "prediction": str(eng_pred).lower().strip()
                    })
                    
            if not converted_preds:
                continue
                
            with open(tmp_pred_file, "w", encoding="utf-8") as f:
                json.dump(converted_preds, f)
                
            # Execute eval.py
            cmd = [
                sys.executable, "eval.py", 
                "--tier", "testdev_balanced",
                "--questions", str(q_file.resolve())
            ]
            try:
                res = subprocess.run(
                    cmd, cwd=str(gqa_eval_dir), capture_output=True, text=True, check=False
                )
                output = res.stdout
            except Exception as e:
                output = ""
                
            # Regex parse the outputs
            def ex(metric):
                match = re.search(fr"{metric}:\s*([\d\.]+)", output, re.IGNORECASE)
                if match: return float(match.group(1))
                return 0.0
                
            binary = ex("Binary")
            open_acc = ex("Open")
            chi = ex("Distribution")
            
            compare = ex("Compare")
            choose = ex("Choose")
            logic = ex("Logic")
            query = ex("Query")
            
            prompt_fmt = prompt.replace("_", " ").title()
            print(f"| {prompt_fmt.ljust(19)} | {binary:05.2f}% | {open_acc:05.2f}% | {chi:05.2f}     | {compare:05.2f}%  | {choose:05.2f}% | {logic:05.2f}% | {query:05.2f}% |")
            
        print("\n")
        
        # Cleanup injected prediction file
        if tmp_pred_file.exists():
            tmp_pred_file.unlink()

if __name__ == "__main__":
    main()
