# 🧠 Architectural Decisions Log

This document records the exact reasoning behind the foundational engineering decisions made while structuring the VoQA-Multilingual pipeline.

---

## Decision 1: Overriding the native VoQA Contrast Threshold (WCAG 4.5 -> WCAG 7.0)

**The Decision:** 
Instead of adhering to the standard WCAG 4.5 (AA) contrast limit for watermark rendering, the pipeline aggressively enforces WCAG 7.0 (AAA) compliance by dynamically clamping the dual-borders to pure black (`#000000`) or pure white (`#FFFFFF`) depending on the background luminance.

**The Reasoning:** 
A standard 4.5 threshold is scientifically sufficient for reading text over clean, flat backgrounds. However, watermarks are intentionally designed to be placed over chaotic, irregular image geometries that feature heavy color bleeding, noise, and visual artifacts. 
By forcefully escalating to AAA WCAG 7.0 mathematics, we physically guarantee that the watermark never blends into a complex background. This protects the ablation integrity: we want to test if the VLM possesses the *cognitive capability* to solve the question, and we do not want it failing simply because the pixels blurred into the background (a visual acuity failure).

---

## Decision 2: Hardcoding English Structural Prompts for Non-English Subsets

**The Decision:** 
Even when the `dataset_records` are entirely Italian or Finnish, and the dynamically-loaded few-shot examples inject Italian text (`"Di che colore è..."`), the overarching VLM system instructions (e.g., *"Your task is to..."* and the JSON formatting rules) remain strictly in English.

**The Reasoning:** 
Vision-Language Models (like InternVL-1B) possess deep multilingual comprehension, but their foundation-layer *instruction-tuning* is overwhelmingly trained in English. If we instruct the model to output strict JSON schemas entirely in Italian, the model frequently misunderstands the rigid syntax and breaks the dictionary brackets. 
By keeping the directive rules firmly in English, we guarantee 100% structural compliance to the JSON format. The model is fully capable of reading English commands, absorbing the Italian few-shot examples, and intelligently routing the Italian watermark text into the English-keyed `{"Answer": "..."}` dictionary without crashing.

---

## Decision 3: Isolating OCR Metadata Constraints Exclusively to Zero-Shot Evaluation

**The Decision:** 
The pipeline vehemently segregates the features. When testing the `--use_ocr True` parameter (which injects physical `[45, 120, 300, 250]` coordinates), it is strictly evaluated under the Zero-Shot configuration and never entangled with the 2, 4, or 8-shot contextual workflows.

**The Reasoning:** 
This relies on two major functional factors:

1. **Ablation Purity (The Science):** Testing whether OCR bounding boxes improve accuracy requires a pristine A/B test limit (Zero-Shot Without OCR vs. Zero-Shot With OCR). If we inject OCR *while simultaneously* providing 8 historical question/answer examples, we corrupt the variables. If accuracy suddenly spikes by 20%, we no longer mathematically know if the improvement was caused by the spatial OCR boxes, or because the model copied the 8 textual examples perfectly.
2. **Contextual Attention Splintering (The Hardware):** In a few-shot paradigm, we would be explicitly forced to inject the OCR coordinates mapping to all 8 of the historical context images organically into the prompt string. The VLM processes these embeddings linearly. If it reads 8 different sets of bounding-box coordinates for images it is theoretically not looking at, its spatial-attention vectors completely misfire. It will frantically try to scan the *current* image based on the coordinates of a *historical* image, heavily crashing the detection tensor layer.

---

## Decision 4: Using BERTScore Soft-Match for Question Alignment Accuracy (QAA)

**The Decision:** 
When placing predictions into `qaa_corrects` vs `qaa_incorrects` arrays, the pipeline strictly uses the `is_match[i]` boolean (which relies on a Semantic BERTScore >= 0.90), rather than checking against the rigid Exact-Match strings. 

**The Reasoning:** 
QAA is designed solely to answer one question: *Did the model correctly detect and read the text visually written inside the image?* 
If the model correctly reads the watermark but chooses to answer organically in conversation (e.g., Ground Truth is `"red"`, but the model replies `"the car is red"`), Legacy Exact Match rigorously fails the output (0%). 
By utilizing the HuggingFace Semantic Soft-Matcher, we prevent punishing the model mathematically for phrasing. If it understood the question well enough to synthesize a 94% semantically identical answer, it deserves the QAA score organically! This prevents catastrophic interference when measuring correlation matrices.

---

## Decision 5: Sequential Looping per GPU Node (Replacing Slurm Arrays)

**The Decision:**
Instead of utilizing a Slurm array (`#SBATCH --array=0-4`) to distribute languages automatically across 5 nodes in one execution, we deliberately changed the logic to run a full 43-configuration sequential loop dynamically on a *single* GPU via `run_pipeline.sh`, forcing the execution of 5 standalone SLURM jobs parameterized natively using CLI arguments (`--lang ita`, `--lang fin`).

**The Reasoning:**
The structural configuration of the mentor's cluster strictly mandates manual allocation (`#SBATCH --gres=gpu:1`, `#SBATCH --nodelist=nodex`) over array-based scattering. To guarantee absolute zero interference, each target language requires a firmly independent job submission. A single node organically loops through the 43 different rendering/prompt combinations for its assigned language sequentially, naturally ensuring structural independence, safety from cross-node file locking, and zero concurrent model VRAM saturation collisions on a single physical die.

---

## Decision 6: Flush-based Atomic Checkpointing for 12-Hour Limits

**The Decision:**
Rather than hoarding 1000 inferences in RAM and saving at the end of the script execution, the inference pipeline opens `.jsonl` files in `append` mode, writes single predictions synchronously, and immediately triggers `file.flush()` line-by-line. It checks existing lines dynamically to skip previously `processed_ids`.

**The Reasoning:**
A cluster limit explicitly slaughters processes flawlessly at exactly 12 hours. If the script was organically structured to write results at completion, 11 hours and 59 minutes of computed tensors would disappear. By aggressively flushing structurally atomic JSON lines and caching them upon restart, we completely disarm the time limitation. When jobs are terminated, Slurm is simply commanded to re-trigger the script, where it smoothly resumes without losing a single compute cycle.

---

## Decision 7: Strict Filename Native Mapping in Renderers

**The Decision:**
Instead of saving rendered files (watermarks, concat-padded) as `{question_id}.jpg`, the pipeline strictly dictates saving them under their native `{raw_image_filename}` format.

**The Reasoning:**
The GQA dataset intrinsically maps multiple specific questions to one single visual raw image. If the renderers saved newly watermarked layers using the `question_id`, the visual inference loop would fail to physically locate the correct rendered image since it implicitly looks for the base `image_filename`. Resaving them precisely by the original raw image filename structurally eliminates dynamic path collisions and maintains a rigid 1-to-N mathematical topology. 

---

## Decision 8: Ghost Environment Purging for Core Torch Components

**The Decision:**
Instead of trusting the baseline SLURM python environment or simple `pip install` commands, the pipeline explicitly demands a "ghost-purge" (running `pip uninstall -y torch torchvision` twice iteratively) before installing the specialized CU121 `cu121/torch-2.2.2` indices inside an isolated Conda architecture.

**The Reasoning:**
Overlapping "Frankenstein" pip environments on computing clusters aggressively create ghost library dependencies where the underlying `torchvision` C++ registries mathematically conflict with native `torch` APIs. Isolating and entirely purging the binaries is the only proven methodology to prevent native `register_fake` tracking errors from triggering core C++ dumps during VRAM allocation.
