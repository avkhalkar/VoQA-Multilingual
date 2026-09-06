# State-of-the-Art Multilingual VoQA Fine-Tuning Architecture

**Target Languages:** English (eng), Italian (ita), Finnish (fin), Dutch (nld), Spanish (spa)
**Models:** InternVL-1B, Qwen-2B
**Compute:** NVIDIA B200 GPU (no compute constraints)
**Training Data:** 3.35M samples (VoQA Training Set)

---

## The Core Problem

VoQA is unique because the question is **physically rendered onto image pixels**. For multilingual VoQA, this creates a **dual challenge** that standard multilingual VLM approaches don't address:

1. **Visual Text Reading Challenge:** The model must OCR/read text from watermarked pixels — but now that text is in Italian/Finnish/Dutch/Spanish, requiring cross-lingual visual text comprehension.
2. **Response Generation Challenge:** The model must generate answers in the corresponding target language, not just English.

---

## Research Sources & Inspirations

| Paper / Source | Key Idea | Link |
|:---|:---|:---|
| **Parrot (ICML 2025)** | MoE-based multilingual visual instruction tuning. Addresses "multilingual erosion" where VLMs lose non-English ability during vision-language alignment. Uses textual guidance to condition visual tokens on language. | [arxiv.org/abs/2406.02539](https://arxiv.org/abs/2406.02539) |
| **Multi-Linguistic LoRA Merging (NeurIPS 2024)** | Decouples Task Adapters and Language Adapters into separate LoRA modules. Enables composable, language-specific refinement without retraining. | [openreview.net](https://openreview.net) |
| **LoRA-LEGO (arXiv 2024)** | Treats LoRA rank parameters as Minimal Semantic Units (MSUs) that can be disassembled and reassembled like building blocks for flexible skill composition. | [arxiv.org](https://arxiv.org) |
| **Deep Supervision Fine-Tuning (AAAI 2024)** | Discovers that English-centric VLMs internally pivot through English even for non-English inputs. Proposes intermediate-layer supervision to improve native language reasoning. | [aaai.org](https://aaai.org) |
| **CAST: Cross-modal Alignment Similarity Test (2025)** | New evaluation paradigm testing whether VLM's visual and textual reasoning are self-consistent across modalities. | Various |
| **mLoRA Pipeline Parallelism** | LoRA-aware distributed training enabling simultaneous multi-adapter training across GPU clusters. | Various |

---

## The SOTA Architecture: 3-Stage Composable Pipeline

### Stage 1: English VoQA Task Mastery (Task Adapter)

Fine-tune on ALL 3.35M English VoQA samples using ALL 8 SFT strategies:
```
InternVL-1B (frozen) + LoRA_task (Rank 8, LR 1e-5)
                         │
         Learns: How to read watermarked text from pixels
                 How to extract questions visually
                 How to generate correct answers
                 8 different Q/R/A structural patterns
```
**Output:** 8 Task Adapter checkpoints (one per SFT strategy)

---

### Stage 2: Multilingual Watermark Re-Rendering

This is the critical step that makes VoQA multilingual **authentic** rather than just translated text.

```
Original English watermarked image:
┌──────────────────────────────────┐
│  🖼️ Kitchen Photo                │
│   "Is there a mug on the        │
│    counter?"                     │
│   (English text burned on pixels)│
└──────────────────────────────────┘
                │
                │ SeamlessM4T translates the question text
                │
                ▼
┌────────────────────────────────────────────────────────────────┐
│  Re-render watermark in target language using PIL + Noto fonts │
└────────────────────────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┬───────────┬───────────┐
    ▼           ▼           ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ 🇬🇧 ENG  │ │ 🇮🇹 ITA  │ │ 🇫🇮 FIN  │ │ 🇳🇱 NLD  │ │ 🇪🇸 SPA  │
│"Is there│ │"C'è una│ │"Onko   │ │"Is er  │ │"¿Hay un│
│ a mug?" │ │ tazza?"│ │ muki?" │ │een mok?"│ │ taza?" │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

**Why this matters:** The model must learn to **visually read** text in non-English scripts/characters from pixels. Just translating the JSONL text answer is NOT enough — the watermarked image itself must contain the target-language text for authentic VoQA evaluation.

**Implementation:**
- Use **Google Noto Fonts** (covers all 5 target languages with proper glyphs)
- Use **PIL/Pillow with libraqm** for correct multilingual text shaping
- SeamlessM4T translates question strings → PIL renders them onto the base image
- Validate rendered characters against source Unicode sequences

---

### Stage 3: Language-Specific LoRA Adapters (Language Adapter)

After Stage 1 gives us a strong English VoQA Task Adapter, we train **separate Language Adapters** per target language:

```
┌──────────────────────────────────────────────────────────────────┐
│  InternVL-1B (frozen)                                            │
│       +                                                          │
│  LoRA_task (frozen from Stage 1 — VoQA knowledge)                │
│       +                                                          │
│  LoRA_lang (NEW, trainable — language-specific adaptation)       │
│                                                                  │
│  Training Data: Re-rendered watermarked images in Language X     │
│  + Translated JSONL targets in Language X                        │
└──────────────────────────────────────────────────────────────────┘
```

**Key Architectural Insight (from MLM paper):**
The Task Adapter already knows HOW to do VoQA (read text, extract questions, generate answers). The Language Adapter only needs to learn the LINGUISTIC BRIDGE — mapping Italian/Finnish/Dutch/Spanish token distributions to the task logic the Task Adapter already mastered.

**This means:**
- Language Adapters can be MUCH smaller (Rank 4 instead of 8)
- They require FAR fewer training samples (the VoQA task logic transfers)
- They are completely modular and swappable at inference time

---

## Complete Architecture Map

```
                    STAGE 1                    STAGE 2                    STAGE 3
              (English Task Mastery)     (Multilingual Rendering)   (Language Adaptation)

   3.35M English    ──► LoRA_task ──┐
   VoQA samples         (Rank 8)   │
                                    │    SeamlessM4T + PIL     ──► LoRA_lang_ita (Rank 4)
                                    ├──► Re-render images      ──► LoRA_lang_fin (Rank 4)
                                    │    in 4 languages        ──► LoRA_lang_nld (Rank 4)
                                    │                          ──► LoRA_lang_spa (Rank 4)
                                    │
                                    ▼
                              AT INFERENCE:
                    ┌─────────────────────────────────────┐
                    │  InternVL-1B (frozen)                │
                    │  + LoRA_task (frozen)                │
                    │  + LoRA_lang_X (swap per language)   │
                    │                                     │
                    │  Input: Watermarked image in Lang X  │
                    │  Output: Answer in Lang X            │
                    └─────────────────────────────────────┘
```

---

## Alternative Approaches Considered (and Why This is Better)

### ❌ Approach A: Translate-Train (Naive)
Translate English JSONL → fine-tune per language.
**Problem:** The watermarked IMAGE still has English text! The model learns Italian answers but sees English questions on pixels. Fundamentally misaligned.

### ❌ Approach B: Single Polyglot Model
Mix all 5 languages into one JSONL and train one model.
**Problem:** Languages compete for LoRA capacity. Finnish (low-resource) gets dominated by Spanish/English. No modularity.

### ❌ Approach C: Translate-Test Only (Zero-Shot)
Fine-tune English, translate at test time.
**Problem:** Model never learns to visually read non-English characters from pixels. Pure reliance on translation quality.

### ✅ Approach D (Our SOTA): 3-Stage Composable Pipeline
Re-render watermarks + composable Task/Language adapters.
**Why best:**
- Model authentically learns to read non-English text from pixels
- Task knowledge transfers efficiently via frozen Task Adapter
- Language Adapters are modular, small, and swappable
- Scales cleanly to new languages without retraining task logic
- Backed by ICML 2025 (Parrot) and NeurIPS 2024 (MLM) research

---

## Checkpoint Count (Full Scale)

| Component | Count |
|:---|:---|
| Task Adapters (8 SFT × 1 English) | 8 |
| Language Adapters (4 langs × 8 SFT) | 32 |
| **Total Adapters** | **40 per model** |
| **Total across InternVL + Qwen** | **80 adapters** |

Each adapter is only ~5-10MB (LoRA weights), so total storage is ~400-800MB. Trivial on B200.

---

## Evaluation Matrix

For EVERY combination, we report:
1. **Exact Match Accuracy** (hard string comparison)
2. **BERTScore Semantic Accuracy** (F1 ≥ 0.90 threshold)
3. **QAA(Correct)** — Question extraction accuracy on correct answers
4. **QAA(Incorrect)** — Question extraction accuracy on wrong answers

Cross-lingual evaluation uses SeamlessM4T back-translation to English ground truth.
