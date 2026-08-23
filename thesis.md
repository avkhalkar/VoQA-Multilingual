# VoQA-Multilingual Scientific Rationale & Expected Outcomes

## 1. The Core Scientific Hypothesis
The primary scientific question of this research is: **Do Vision-Language Models (VLMs) uniformly retain their visual-OCR reasoning capabilities across cross-lingual boundaries?**

While the original VoQA paper demonstrated that physical watermarks and concatenated text improve VLM reasoning in English, it remains unknown whether this property is a universal visual processing phenomenon or an artifact heavily overfit to English instruction-tuning. 

## 2. Experimental Baseline Integrity (The English Control)
By structurally recreating the exact methodologies of the original paper (InternVL2.5-1B model, WCAG 4.5/7.0 contrast rendering, 500-line post-filtration evaluation engine), this codebase guarantees a 100% physically identical baseline. We anticipate our English control group to perfectly mirror the following `InternVL` benchmarks:
- **English VoQA OCR-Assisted:** ~43.0%
- **English VoQA Workflow Prompts (Zero-Shot):** ~25.0% - 27.0%

## 3. Predicted Multilingual Divergence
We hypothesize a severe divergence between High-Resource and Low-Resource linguistic pipelines, even when the visual text extraction (OCR) behaves identically.

### Scenario A: High-Resource Languages (Spanish & Italian)
**Prediction:** Accuracy will closely trail the English baseline (achieving ~85% to 90% of the English metrics, e.g., ~35%-38% OCR-Assisted). 
**Reasoning:** InternVL's multi-modal pre-training data inherently contains vast amounts of Spanish and Italian textual reasoning. The vision encoder will successfully isolate the text from the pixels, and the internal language model will successfully bridge the semantic mapping from the image to the target language.

### Scenario B: Low-Resource Languages (Finnish & Dutch)
**Prediction:** A severe mathematical drop (achieving only ~50% to 60% of the English baseline).
**Reasoning:** The failure point is cognitive, not visual. The vision encoder will still perfectly extract the Finnish watermark (e.g., *"Minkä värinen kissa on?"*), but the internal transformer blocks lack sufficient context mapping to reason about the physical image geometry using Finnish grammatical structures.

## 4. Conclusion & Impact
If the metrics mathematically align with these projections, this experimentation confirms a massive, highly publishable insight into modern VLMs:
**Physical Visual-OCR extraction works uniformly across alphabets, but cognitive semantic reasoning does not transfer out-of-the-box.** A VLM can "see" a non-English sentence dynamically stamped on an image, but it cannot universally understand it.
