# Multilingual VoQA: Thesis Experimental Progression

**Target Architecture:** InternVL-1B and Qwen-2B (TinyLLaVA deferred)
**Core Challenge:** Establishing a lightweight, decoupled framework for cross-lingual visual text comprehension (Multilingual VoQA) that avoids catastrophic forgetting and multilingual erosion.

---

## The 4-Stage Experimental Progression

This 4-stage narrative is designed to scientifically prove the necessity of composable modularity (Task + Language adapters) combined with pixel-level deep supervision (Watermark re-rendering).

### Stage 1: The "Zero-Shot" (Translate-Test) Baseline
* **Experiment Setup:** Train on 100% English VoQA datasets. Test on non-English VoQA queries (e.g., Italian) where the text input is translated at test time.
* **Objective:** Establish the failure of zero-shot cross-lingual transfer.
* **Hypothesized Result:** Severe accuracy drop. Proves that visual text grounding (reading watermarked English text from pixels) does not naturally transfer to non-English semantic spaces without explicit training.

### Stage 2: The "Monolithic Multilingual" (The Naive Trap)
* **Experiment Setup:** Physically re-render the watermarks in the target language (e.g., Italian). Train a standard, monolithic LoRA block using these translated images and translated JSONL targets. (No modular separation).
* **Objective:** Induce and measure "Multilingual Erosion" and "Catastrophic Forgetting."
* **Hypothesized Result:** The model improves at Italian OCR but its core English VoQA reasoning degrades. Proves that forcing complex VoQA task structure (QRA/RQA) and new language acquisition into the same small Rank-8 bottleneck causes severe knowledge interference.

### Stage 3: Pure Modularity SOTA (The 4 + 8 Proof)
* **Experiment Setup:** Decouple the architecture. Freeze the 8 English Task Adapters (which have mastered the VoQA structural settings). Train exactly **4 Universal Language Adapters** (Rank 4) on the re-rendered images (1 purely for Italian, 1 purely for Finnish, etc.). 
* **Adapter Count:** 8 Task + 4 Language = **12 total adapters** per base model.
* **Objective:** Prove the philosophical concept of "Language-as-a-LEGO-block." Demonstrate that a single, universal Italian language adapter can successfully plug into an English `QRA` task adapter *and* an English `QA-only` task adapter without requiring retraining for specific structural settings.

### Stage 4: Exhaustive Modularity SOTA (The 32 + 8 Performance Ceiling)
* **Experiment Setup:** Train highly specialized language adapters mapped explicitly to specific task adapters. (e.g., Train the Italian adapter specifically while it is plugged into the `QRA` Task Adapter, so it perfectly masters both the Italian translation *and* the `QRA` target formatting constraint).
* **Adapter Count:** 8 Task + 32 Specialized Language (4 langs × 8 settings) = **40 total adapters** per base model.
* **Objective:** Establish the absolute maximum theoretical accuracy ceiling for the Multilingual VoQA architecture. Provides a brilliant engineering tradeoff analysis comparing the lightweight flexibility of Stage 3 (12 adapters) against the maximum performance of Stage 4 (40 adapters).

---

## Next Steps for Execution (100-Sample Subsets)
Before running the full 3.35M scaling, we will validate the base pipeline (Stage 1 Task Adapters) on the 100-sample subsets for GQA, POPE, and ScienceQA using the 8 SFT mappings.
