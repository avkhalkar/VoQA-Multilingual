# Architectural Decisions: Multilingual VoQA Pipeline

This document catalogs the finalized architectural frameworks, output standardizations, and execution protocols governing the Multilingual VoQA fine-tuning pipelines.

---

## 1. Execution Protocol
Because we are working across environments with heavy module dependencies, all terminal/shell operations dictating pipeline execution must adhere to the following sequence to guarantee library isolation:

1. Initiate `wsl` to bridge into the Linux environment.
2. Execute `conda activate internvl` to engage the isolated DeepSpeed PyTorch environment.
3. Run model pipeline scripts exclusively from within this active container.

---

## 2. Output Directory Standardization
To manage the heavy combinatorial explosion of LoRA weights (40 adapters generated across different configurations per model), the output directory strictly enforces a top-level **Stage Classification** tree.

### Expected Directory Hierarchy
```text
📁 output_tuning/
└── 📁 [model_name] (e.g., internvl-1B)
    │
    ├── 📁 S1_English_Task_Adapters/
    │   ├── 📁 [dataset] (e.g., gqa, pope, scienceqa)
    │   │   ├── 📄 evaluation_results.json
    │   │   └── 📁 [sft_strategy] (e.g., qa_sft)
    │   │       ├── adapter_model.safetensors
    │   │       ├── trainer_state.json
    │   │       └── evaluated_[sft_strategy].jsonl
    │
    ├── 📁 S2_Monolithic_Multi_Adapters/
    │   └── 📁 [language_code] (e.g., ita, fin, nld, spa)
    │       └── 📁 [dataset] / [sft_strategy]
    │
    ├── 📁 S3_Pure_Language_Adapters/
    │   └── 📁 [language_code] (e.g., ita, fin)
    │       └── 📁 cross_dataset / universal_lang_sft
    │
    └── 📁 S4_Exhaustive_Language_Adapters/
        └── 📁 [language_code] (e.g., ita, fin)
            └── 📁 [dataset] / [sft_strategy]
```

### Purpose of Output Typing
By segregating the root folders into `S1`, `S2`, `S3`, and `S4`, the directory fundamentally documents the experimental progression of the thesis logically:
- **S1:** The baseline Zero-Shot behavior.
- **S2:** The Monolithic trap (proving the existence of Multilingual Erosion).
- **S3 & S4:** The decoupled, composable SOTA solutions (Pure vs Exhaustive Modularity). 

---

## 3. The "English Exclusion" Adapter Mathematics
A critical mathematical design choice in our architectural logic is the explicit **exclusion of the English language from Stage 3 and Stage 4**, resulting in 32 adapters instead of 40 for Stage 4.

**The Logic:**
- The base model combined with the frozen Stage 1 (S1) Task Adapter intrinsically serves as the native English processor.
- A "Language Adapter" (built in S3 and S4) operates purely as a cross-lingual bridge, mapping foreign visual tokens to English concepts for the Task Adapter.
- Training an English Language Adapter would function as a redundant "Identity Layer" (mapping English tokens to English tokens).
- Therefore, the dataset distribution states: **Target English exclusively in Stage 1.** Train **Target Foreign Languages (Italian, Finnish, Dutch, Spanish) exclusively in Stages 2, 3, and 4**.
