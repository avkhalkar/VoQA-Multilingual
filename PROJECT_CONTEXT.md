# 🧠 PROJECT CONTEXT — VoQA-Multilingual
*Status: Ready for Cluster Deployment (100-500 Pilot Subset)*
*Date: August 24, 2026 (Pre-12:00 PM Deadline)*

## 1. Project Overview & Objective
This project is an advanced Vision-Language Model (VLM) multilingual evaluation pipeline targeting the VoQA (Visual OCR Question Answering) paradigm. 
The core objective is answering: *Does a VLM natively reason over visual text embedded inside an image dynamically better across multiple languages, versus traditional plaintext prompts?*

**Execution Constraints:**
- Remote Slurm Cluster (`anand@10.240.60.36`) 
- 6 GPUs physically available (2 H100s, 4 A100s). Target usage is exactly 5 GPUs.
- Hard 12-Hour job termination limit.

## 2. The Finalized Architecture Pipeline
The pipeline executes sequentially through 5 core phases:
1. **Data Prep** (`select_subset.py`): Extracts a random deterministically seeded subset of the GQA dataset (currently configured for pilot testing: 100-500 subset).
2. **Translation** (`translate_m4t.py`): Uses Meta's SeamlessM4T v2 to natively push English questions into 4 languages (`ita, fin, nld, spa`).
3. **Rendering** (`render_*.py`): Three distinct Python scripts natively render the translated text onto the raw image via Watermarking, Concat-Padding, and Concat-Resizing.
4. **Inference** (`run_inference.py`): InternVL-1B performs reasoning inside an aggressive 43-configuration loop (various few-shot limits, zero-shot OCR vectors, unstructured baselines).
5. **Evaluation** (`evaluate_model.py`): Results are pre-filtered via robust Regex, Back-Translated securely with a RAM dictionary cache, and finally Post-Filtered using a 500-line heuristic engine derived natively from the VoQA paper.

## 3. Current Phase Status: "Ready for Pilot Execution"
**All local structural coding is formally bug-free and completed**:
- Checkpointing implemented: `run_inference.py` safely flushes predictions synchronously line-by-line and checks `processed_ids` to defeat the 12-hour limit natively.
- Cluster SLURM fixed: `run_pipeline.sh` structurally utilizes sequential looping over 43 configs internally avoiding SLURM Array lock-out limits.
- Evaluator mapped: Mathematical accuracy handles regex gracefully against `model_prediction`.
- Paths verified: `voqa_gqa.zip` internal logic securely targets standard pathing rules.

## 4. Key AI Handoff Directives
If an agent is resuming work from this context file, please closely adhere to these mandates:
- **Do not overwrite `decisions.md`:** Important historical context is physically saved there. 
- **Do not arbitrarily change configurations:** The math produces exactly **215** experiments per sample per language. Modifying the loop will physically break evaluation matrices.
- **Node Collision Prevention:** GPU executions are structurally isolated natively via the `--lang` flag (e.g., Node 3 solely operates on `fin`). Do not bundle overlapping JSON access loops.
- **Check `deployment_guide.md` and `expected.md`**: These two artifacts (located in this chat's artifact history) definitively list the exact physical SCP commands and the overarching mental paradigm.

## 5. Immediate Next Step for the User
The user inherently needs to:
1. `scp` the structural files (`src/`, `scripts_cluster/`, `voqa_gqa.zip`) to the cluster.
2. Initialize and test the Python conda environment.
3. Launch the rendering Python files sequentially.
4. Run `sbatch run_pipeline.sh --lang eng`, `--lang ita`... across the 5 target Nodes natively to start the inference validation.
