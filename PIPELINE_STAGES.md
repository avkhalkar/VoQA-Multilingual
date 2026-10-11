# VoQA Multilingual Pipeline: Master Stage Tracker

Last Updated: 2026-10-11 18:00 Local Time

---

## 📊 Summary of Stages & Milestones

| Stage ID | Component | Description | Status |
| :--- | :--- | :--- | :---: |
| **STAGE 0** | **Environment & Hardware** | B200 MIG sm_100 CUDA compatibility & container setup | ✅ **DONE** |
| **STAGE 1A** | **Data Preparation (Rendering)** | Multilingual watermark rendering pipeline on 16TB NFS | 🔄 **IN PROGRESS** (45.4% ita) |
| **STAGE 1B** | **TextVQA Special Logic** | 4-sided canvas padding below image to preserve OCR scene | ✅ **DONE** |
| **STAGE 2A** | **Model Loading: DeepSpeed Engine** | ZeRO-3 partitioner & NVTX profiler bug neutralization | ✅ **DONE** |
| **STAGE 2B** | **Model Loading: Language Model (LLM)** | Qwen2-0.5B loading, parameter partitioning & weights setup | ✅ **DONE** |
| **STAGE 2C** | **Model Loading: Vision Tower** | `InternVisionModel` loading & `all_tied_weights_keys` descriptor | ✅ **DONE** |
| **STAGE 2D** | **Model Loading: Multi-modal Connector** | 2-layer MLP projection layer (`connector._connector`: 918,400 params) | ✅ **DONE** |
| **STAGE 3A** | **LoRA Architecture & Dataset Loader** | LoRA Lego adapters & 600-sample JSON/image data pipeline | ✅ **DONE** |
| **STAGE 3B** | **English Task SFT (Stage 1 Adapters)** | `voqa_baseline` and `baseline_sft` training steps on B200 | 🔄 **RUNNING NOW** |
| **STAGE 4** | **Multilingual LoRA SFT (Stage 3)** | Sequential fine-tuning (`ita` -> `fra` -> `spa` -> `nld`) | ⏳ **PENDING** |

---

## 🔍 Detailed Progress Trace: Are We Making Progress or In Circles?

Here is the exact step-by-step history proving continuous linear progress:

### Step 1: NVIDIA B200 Architecture Compatibility
- **Initial Problem**: PyTorch 2.4 crashed on `sm_100` (NVIDIA B200 Blackwell MIG slice).
- **Action Taken**: Deployed NVIDIA PyTorch 24.10 NGC container (`b200_sandbox`) with native PyTorch 2.5 `sm_100` support.
- **Result**: ✅ **PASSED** (Container initializes cleanly).

### Step 2: Container Dependencies
- **Initial Problem**: Missing `timm` library in the container environment.
- **Action Taken**: Installed `timm` inside `b200_sandbox`.
- **Result**: ✅ **PASSED** (`timm.layers` imports cleanly).

### Step 3: DeepSpeed Parameter Partitioning (NVTX Bug)
- **Initial Problem**: DeepSpeed ZeRO-3 parameter partitioner crashed during `model.load_llm` with:
  `AttributeError: module 'nvtx' has no attribute 'get_domain'`.
- **Action Taken**: Neutralized DeepSpeed NVTX profiling calls and mocked `nvtx.get_domain` in `train_orchestrator.py`.
- **Result**: ✅ **PASSED & VERIFIED** (DeepSpeed partitioned parameters with zero NVTX errors).

### Step 4: Language Model (LLM) Loading
- **Initial State**: Blocked by Step 3.
- **Status in Run 1830**: **`Qwen2ForCausalLM` successfully loaded and partitioned under ZeRO-3!**
  Notice in log: `loading language model from /scratch/users/anandk03/pretrained/InternVL2_5-1B`.
- **Result**: ✅ **PASSED**.

### Step 5: Vision Tower (`InternVisionModel`) Loading
- **Error in Run 1832**:
  `AttributeError: 'NoneType' object has no attribute 'keys'` at line 4856 (`getattr(self, "all_tied_weights_keys", {}).keys()`).
- **Root Cause**: `InternVisionModel` defined `_tied_weights_keys = None`, causing `all_tied_weights_keys` to return `None` instead of `{}`.
- **Fix Applied**: Upgraded `_TiedWeightsDescriptor` in `train_orchestrator.py` so `__get__` strictly guarantees returning a mapping with `.keys()`, never `None`.
- **Result in Run 1833**: ✅ **PASSED & VERIFIED** (Vision tower loaded completely).

### Step 6: Trainer Initialization & DataLoader Construction
- **Errors in Runs 1833-1834**:
  - `TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'`.
  - `TypeError: LLaVATrainer._get_train_sampler() takes 1 positional argument but 2 were given`.
- **Root Causes**: Modern Hugging Face `transformers` (4.45+):
  1. Replaced `tokenizer` with `processing_class` in `Trainer.__init__`.
  2. Passes `dataset` as an argument to `_get_train_sampler(dataset)`.
- **Fixes Applied**:
  1. Adapted `tokenizer` → `processing_class` dynamically via signature inspection in `train_orchestrator.py`.
  2. Updated `_get_train_sampler` in both `tinyllava_trainer.py` and `train_orchestrator.py` to accept `dataset=None, *args, **kwargs`.
- **Result**: ✅ **PASSED & VERIFIED**.

### Step 7: Training Engine Git Synchronization (Run 1840 Root Cause)
- **Error in Run 1840**:
  `ModuleNotFoundError: No module named 'tinyllava.train.train'`.
- **Root Cause**: `git reset --hard origin/main` pulled from a commit that didn't include uncommitted local engine modifications (`train/__init__.py` still had `from .train import *` instead of `from .train_uni import *`).
- **Fix Applied**: Fully committed all engine modules (`train/`, `train_orchestrator.py`, `zero3.json`, and vision tower extensions) to `main` and pushed to GitHub.
- **Result**: ✅ **SYNCHRONIZED**.

### Step 8: Stage 1 600-Sample Test Training (Active) 👈
- **Task**: DeepSpeed ZeRO-3 forward/backward passes on NVIDIA B200 MIG slice.
- **Status**: 🔄 **Launching Now**.

---

## 📋 What is Left After Step 8:

1. **Loss Progression Verification**: Verify initial training steps log loss decreases (1 epoch on 600 samples).
2. **LoRA Checkpoint Saving**: Verify `adapter_model.safetensors` saved in `output_tuning/`.
3. **Full Multilingual LoRA SFT (Stage 3)**: Scale from the 600-sample test to full dataset with rendered watermarks from `/mnt/gpu_users/Aditya/rendered_images/`.

