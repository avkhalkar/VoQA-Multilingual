# 📊 VoQA Evaluation Metrics Architectures

This file acts as a centralized engineering map completely explaining exactly how the entire `VoQA-Multilingual` framework mathematically compiles, parses, and scores Vision-Language Model inferences. 

There are mathematically three interconnected scripts acting in sequence:

---

## 1. The Global Orchestrator: `evaluate_model.py`
*(Lives entirely inside `src/eval`)*

**Purpose:** 
This script mathematically links model outputs cleanly back into Ground Truth configurations seamlessly generating raw numeric metrics (QAA, F1, exact match).

**Execution Loop Breakdown:**
1. Loads Ground Truth GQA Dictionary `testdev_balanced_questions.json` and user Inference `results_*_Xshot.jsonl`.
2. Intercepts the predictions physically inside the `response_pipeline.py` centralized engine parsing:
    - Pre-Filtering: Aggressively uses standard Regex strictly isolating internal JSON dictionaries directly bypassing "Sure here is it!" conversational payload hallucinations dynamically. (VoQA Appendix B.2).
    - SeamlessM4T Backtranslation.
    - Post-Filtering: Runs rigorous exact-match logic bounds filtering string mappings organically down to semantic roots.
3. Invokes `calculate_qaa_score` physically mapping Levenshtein geometry strings detecting strictly how well the VLM read the internal watermark visually.
4. Invokes `compute_semantic_score` feeding two massive 12,000 array loops physically inside `BERTScore` on HuggingFace tensors mathematically verifying boolean semantic connections (e.g. `is_match=True` if F1 >= 0.90).
5. Exports arrays physically overwriting internal prediction strings back to disk mapped organically as `evaluated_{config_name}.jsonl`. 
6. Exports global scores mathematically directly into `output/internvl_1b/gqa/evaluation_results.json`.

---

## 2. The Core Compiler: `report_core_metrics.py`
*(Lives strictly in the Root directory `src/` limit)*

**Purpose:** 
Instantly compiles a highly-readable global Markdown matrix aggregating overall ablation conditions comparing variables organically without crashing limits.

**Internal Functionality:**
1. Scrapes the structural `evaluation_results.json` natively calculated identically by `evaluate_model.py`.
2. Outputs beautiful formatted columns globally matching Prompt configuration against Semantic Soft Match accurately comparing conditions directly on the command line safely without needing Excel calculations.

---

## 3. The Deep Diagnostic Sub-Metric Engine: `report_nuanced_metrics.py`
*(Lives strictly in the Root directory `src/` limit)*

**Purpose:** 
Drills aggressively internally through logic barriers calculating strictly Sub-type semantics bounding structural capabilities. 

**Internal Mathematical Bridge Mapping:**
Instead of dynamically re-inventing evaluating systems recursively for 5 different languages organically, the orchestrator constructs structurally a dummy file locally. 
1. Iterates aggressively mapping through all translated internal geometries inside `.jsonl`.
2. Triggers `subprocess.run(["eval.py"])` explicitly loading original monolithic LLaVA/GQA arrays organically running validation mapping externally over dummy internal files.
3. Automatically maps Regex (`re.search`) scraping the command-line tensor variables natively bypassing execution overhead accurately saving out distributions like `Chi-Square`, `Binary`, `Open`, `Compare`, and `Logic`.
