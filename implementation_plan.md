# Multilingual VoQA — 1K Pilot Evaluation Pipeline

Build and execute a complete 1,000-image multilingual VoQA experiment on local hardware (RTX 4060, 8GB VRAM, InternVL2_5-1B).

## Codebase Architecture Summary

The VoQA project:
- **Dataset**: GQA-based, stored at `voqa_gqa/test/gqa/`. Full set = 12,578 questions in `llava_gqa_testdev_balanced.jsonl`. Test100 = 100 questions. Ground truth in `testdev_balanced_questions.json`.
- **Question schema**: `{question_id, image, text, category}` — `text` contains the question + "Answer the question using a single word or phrase."
- **Images**: Raw originals in `images/` (398 unique images shared across questions). Pre-rendered English watermark images in `gqa_watermark_rendering_image/{qid}.jpg`.
- **Existing watermark style**: Dark text overlaid on bottom-right of original image, with word-wrapping.
- **InternVL**: `models/InternVL/internvl_model.py` — loads `InternVL2_5-1B`, bfloat16, `batch_chat()`, `max_new_tokens=1024, do_sample=False`.
- **Prompts**: 8 configs (ID 0–7). Prompt 0 = blank (pure zero-shot). Others add instructions like "find the question in this image..."
- **Evaluation**: `eval/convert_gqa_for_eval.py` → `eval/process_answer.py` (response filtering) → `voqa_gqa/test/gqa/eval/eval.py` (GQA accuracy).
- **Translation**: `seamlesscode/` uses SeamlessM4T v2 large for LLeQA French→{eng, jpn, ita, fin, nld, spa}. **NOT compatible** with VoQA (different dataset, different schema). Only reuse the translation model + `translate_sent()` function pattern.

## User Review Required

> [!NOTE]
> **Target languages**: Based on the user's feedback, the project uses: **English (source), Italian, Finnish, Dutch, Spanish** (4 target languages). Japanese has been dropped to focus on Latin/European languages.

> [!NOTE]
> **Evaluation parser strategy**: Since `process_answer.py` uses strict English heuristics (like "yes", "no", "the answer is"), the multilingual model predictions will be translated **back to English** using SeamlessM4T before passing them to the evaluator.

> [!NOTE]
> **Execution Environment**: We will execute the inference exclusively inside WSL, making sure to `conda activate internvl` beforehand. We'll run the translation step first, and then the inference step sequentially due to VRAM constraints.

---

## Proposed Directory Structure

To maintain a clean, readable research codebase that scales to future extensions (few-shot, OCR, fine-tuning, other models), all new multilingual code, data, and experiment outputs will be strictly isolated from the original English pipeline:

```text
VoQA-Multilingual/
│
├── data_multilingual/               # All generated multilingual datasets
│   ├── subsets/                     # e.g., pilot_1k_ids.json, subset_1k.jsonl
│   ├── translations/                # spa/questions.jsonl, ita/questions.jsonl
│   └── rendered_images/             # e.g., spa/201307251.jpg
│
├── src/                             # Core Python modules for our pipeline
│   ├── data_prep/                   # Data generation scripts
│   │   ├── select_subset.py
│   │   ├── translate_m4t.py
│   │   └── render_watermarks.py
│   └── eval/                        # Evaluation logic
│       └── evaluate_multilingual.py # Handles translating back -> process_answer
│
├── scripts_multilingual/            # Bash execution scripts for the pipeline
│   ├── 01_run_translation.sh
│   ├── 02_run_rendering.sh
│   └── 03_run_zero_shot.sh          # Handles WSL internvl activation + inference
│
└── experiments/                     # Output directory for predictions & logs
    └── zero_shot_pilot_1k/          # The current experiment
        ├── internvl2_5_1b/
        │   ├── english/
        │   │   ├── prompt_0/        # Contains predictions.jsonl
        │   │   └── prompt_1/...
        │   ├── spanish/...
        └── metrics_summary.md
```

---

## Proposed Changes

### 1. Subset Selection Script

#### [NEW] [select_subset.py](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/src/data_prep/select_subset.py)
- Load `llava_gqa_testdev_balanced.jsonl` (12,578 questions)
- Deterministic selection: first 1,000 questions by file order (seed=42 shuffle if preferred)
- Verify each selected question's original image exists in `images/`
- Output: `data_multilingual/subsets/pilot_1k_ids.json` and `subset_1k.jsonl`

---

### 2. Translation Pipeline

#### [NEW] [translate_m4t.py](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/src/data_prep/translate_m4t.py)
- Thin adapter around SeamlessM4T v2 large
- Translates English question text → {Italian, Finnish, Dutch, Spanish}
- Incremental output to `data_multilingual/translations/{language}/questions.jsonl`
- Resumable: skips already-translated question_ids

---

### 3. Multilingual Watermark Renderer

#### [NEW] [render_watermarks.py](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/src/data_prep/render_watermarks.py)
- Replicates existing English watermark rendering style
- Uses PIL/Pillow with a Unicode-capable font (Noto Sans for Latin/European languages)
- Supports text wrapping, clipping prevention
- Outputs to `data_multilingual/rendered_images/{language}/{qid}.jpg`

---

### 4. Inference Pipeline

#### [NEW] [run_zero_shot.sh](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/scripts_multilingual/03_run_zero_shot.sh)
- Loops over languages × prompts
- Activates WSL `internvl` environment and calls `models_inference.py` wrapper
- Batch size = 1 (8GB VRAM constraint)
- Incremental JSONL output to `experiments/zero_shot_pilot_1k/internvl2_5_1b/{language}/prompt_{id}/predictions.jsonl`

---

### 5. Evaluation Pipeline

#### [NEW] [evaluate_multilingual.py](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/src/eval/evaluate_multilingual.py)
- **Translates model predictions back to English** using SeamlessM4T (to accommodate English-only heuristics).
- Reuses `eval/process_answer.py` for response filtering on the translated text.
- Outputs summary table to `experiments/zero_shot_pilot_1k/metrics_summary.json`

---

### 6. Experiment Config & README

#### [NEW] [README.md](file:///d:/Main/Rnd_Projects/VoQA-Multilingual/experiments/zero_shot_pilot_1k/README.md)
- Full documentation for reproducibility

---

## Verification Plan

### Automated Tests (step-by-step execution)

1. **Subset selection verification**
   ```bash
   cd /mnt/d/Main/Rnd_Projects/VoQA-Multilingual
   python pilot_1000/select_1k_subset.py
   python -c "import json; ids=json.load(open('pilot_1000/selected_ids.json')); print(f'Selected {len(ids)} samples'); assert len(ids)==1000"
   ```

2. **Translation smoke test** (5 samples)
   ```bash
   python pilot_1000/translate_voqa.py --limit 5
   # Verify: non-empty translations, correct language codes, Unicode preservation
   python -c "
   import json
   with open('pilot_1000/translated/translations.jsonl') as f:
       rows = [json.loads(l) for l in f]
   print(f'{len(rows)} translations')
   for r in rows[:2]:
       print(r['question_id'], r['language'], r['translated_text'][:80])
   "
   ```

3. **Renderer smoke test** (10 samples)
   ```bash
   python pilot_1000/render_voqa.py --limit 10
   # Visual inspection: open rendered images in pilot_1000/rendered/
   ```

4. **Single-sample inference test**
   ```bash
   python pilot_1000/run_inference.py --limit 1 --language english --prompt-id 0
   ```

5. **Evaluation test**
   ```bash
   python pilot_1000/evaluate_pilot.py
   cat pilot_1000/metrics/summary.md
   ```

### Manual Verification
- Visually inspect 2–3 rendered watermark images per language to verify text readability, positioning, and Unicode correctness
- Verify that the summary table shows non-trivial accuracy (comparable to English-only results)
- Confirm predictions JSONL files have correct format for downstream analysis
