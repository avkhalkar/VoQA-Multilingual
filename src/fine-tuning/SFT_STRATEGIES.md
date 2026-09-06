# VoQA SFT Strategy Reference (8 Configurations)

## Legend
- 🟡 **System Prompt** — Fixed, never graded
- 🔵 **User Prompt / Input** — Input tokens, never graded (loss = -100)
- 🟢 **Supervised Target** — Loss IS calculated on these tokens
- ⬛ **ASSISTANT: (dark)** — Role token in INPUT, NOT graded
- 🟢 **ASSISTANT: (green)** — Role token in TARGET, IS graded

## Notation Key
- **"R-"** (with dash prefix) = Role token placed in **INPUT** side (⬛ dark, NOT graded)
- **"R"** (no dash, inline) = Role token placed in **TARGET** side (🟢 green, IS graded)

---

## Baselines

### 1. VQA Baseline-SFT (Control — Standard VQA, NOT VoQA)
```
🟡System  🔵USER:  🔵<image>(clean!)  🔵"Question as TEXT"  ⬛ASSISTANT:  🟢<Answer>
```
- **Image:** Clean original (NO watermark)
- **Question:** Given as normal text string in input
- **Loss on:** Answer ONLY
- **Purpose:** Control experiment to compare against VoQA strategies

### 2. VoQA Baseline-SFT
```
🟡System  🔵USER:  🔵<image>(watermarked!)  ⬛ASSISTANT:  🟢<Answer>
```
- **Image:** Watermarked (question printed on pixels)
- **Question:** Must be read from image (not given as text)
- **Loss on:** Answer ONLY
- **Purpose:** Simplest VoQA — model not explicitly trained to extract question

---

## Group 1

### 3. R-QA-SFT
```
🟡System  🔵USER:  🔵<image>  ⬛ASSISTANT:  🟢<Question>  🟢<Answer>
```
- **Loss on:** Question + Answer
- **ASSISTANT: in input** (dark, NOT graded) — serves as generation hint

### 4. QRA-SFT
```
🟡System  🔵USER:  🔵<image>  🟢<Question>  🟢ASSISTANT:  🟢<Answer>
```
- **Loss on:** Question + ASSISTANT: + Answer (ALL three graded)
- **No ASSISTANT: in input** — Role token appears BETWEEN Q and A in target

---

## Group 2

### 5. R-QRA-SFT
```
🟡System  🔵USER:  🔵<image>  ⬛ASSISTANT:  🟢<Question>  🟢ASSISTANT:  🟢<Answer>
```
- **Loss on:** Question + ASSISTANT: + Answer
- **Two ASSISTANT: tokens:** first is dark (input hint), second is green (graded in target)

### 6. QA-only-SFT
```
🟡System  🔵USER:  🔵<image>  🟢<Question>  🟢<Answer>
```
- **Loss on:** Question + Answer
- **No ASSISTANT: anywhere at all** — cleanest possible structure

---

## Group 3

### 7. RQA-SFT
```
🟡System  🔵USER:  🔵<image>  🟢ASSISTANT:  🟢<Question>  🟢<Answer>
```
- **Loss on:** ASSISTANT: + Question + Answer
- **KEY DIFFERENCE from R-QA-SFT:** ASSISTANT: is GREEN (graded!), placed as FIRST token of target BEFORE question

### 8. RQRA-SFT
```
🟡System  🔵USER:  🔵<image>  🟢ASSISTANT:  🟢<Question>  🟢ASSISTANT:  🟢<Answer>
```
- **Loss on:** ASSISTANT: + Question + ASSISTANT: + Answer
- **Two ASSISTANT: tokens, BOTH green (both graded)** — maximum supervision

---

## Quick Reference Table

| # | Strategy | Image | Input ASSISTANT:? | Target ASSISTANT:? | Loss Tokens |
|:--|:---------|:------|:-:|:-:|:---|
| 1 | VQA Baseline-SFT | 🖼️ Clean | ⬛ Yes | — | A |
| 2 | VoQA Baseline-SFT | 🖼️ Watermarked | ⬛ Yes | — | A |
| 3 | R-QA-SFT | 🖼️ Watermarked | ⬛ Yes (not graded) | ❌ No | Q + A |
| 4 | QRA-SFT | 🖼️ Watermarked | ❌ No | 🟢 Yes (graded) | Q + R + A |
| 5 | R-QRA-SFT | 🖼️ Watermarked | ⬛ Yes (not graded) | 🟢 Yes (graded) | Q + R + A |
| 6 | QA-only-SFT | 🖼️ Watermarked | ❌ No | ❌ Absent | Q + A |
| 7 | RQA-SFT | 🖼️ Watermarked | ❌ No | 🟢 Yes (graded, BEFORE Q) | R + Q + A |
| 8 | RQRA-SFT | 🖼️ Watermarked | ❌ No | 🟢 Yes (both graded) | R + Q + R + A |

---

## Hyperparameters (Section C.1)

| Parameter | InternVL-1B | Qwen-2B |
|:----------|:------------|:--------|
| Optimizer | AdamW | AdamW |
| Scheduler | Cosine (0.1 warmup) | Cosine (0.1 warmup) |
| Learning Rate | 1e-5 | 1.4e-5 |
| LoRA Rank | 8 | 8 |
| Batch Size | 64 | 64 |
| Epochs | 1 | 1 |
