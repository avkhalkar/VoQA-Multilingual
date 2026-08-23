# 🧮 VoQA Multilingual Pipeline: Cluster Time Estimations

This document calculates the strict mathematical execution bounds for running the full evaluation matrix across the SLURM HPC Cluster against the massive 12.5k target dataset natively.

## 1. Parameters & Overhead Estimates
- **S:** `12,500` (Samples per Language)
- **L:** `5` (Target Languages: `eng, ita, fin, nld, spa`)
- **C:** `43` (Independent Logic Configurations mapping 0-8 shots, OCR limits, and geometric renderings)
- **p:** `~0.25` seconds (A100 Execution latency for InternVL 1B to process an image array and sequentially generate a 10-token string without batching)
- **t:** `~0.08` seconds (A100 Execution latency for SeamlessM4T to structurally translate short strings into English natively)

## 2. Configuration Logic Bounds
- **Configurations requiring Translation:** 4 languages (`ita, fin, nld, spa`) $\times 43$ configs $= 172$ execution pipelines natively.
- **Configurations bypassing Translation:** 1 language (`eng`) $\times 43$ configs $= 43$ execution pipelines natively.

---

## 3. Global Compute Latency (Sequential Single GPU)
If this was structurally executed on a single GPU node legally linearly:

* **Non-English Inference Compute:** 
  `172 * 12,500 * 0.25s = 537,500` seconds **(149.3 hours)**

* **English Inference Compute:** 
  `43 * 12,500 * 0.25s = 134,375` seconds **(37.3 hours)**

* **Non-English Translation Compute:** 
  `172 * 12,500 * 0.08s = 172,000` seconds **(47.7 hours)**

**Total Global Compute Required:**
`149.3 + 37.3 + 47.7` = **234.3 Global GPU Hours**

---

## 4. The `nlp1` Cluster Array Parallelization
Because the architecture strictly decouples the physical hardware using the Execution Array (`#SBATCH --array=0-4`), mapping exactly 1 Language uniquely per Node, the workload is distributed structurally.

The heaviest computational load physically rests on a Non-English node (e.g., Italian, Finnish), which computes exactly:
* **Node Inference:** $43 \times 12,500 \times 0.25s$ = **37.3 hours**
* **Node Translation:** $43 \times 12,500 \times 0.08s$ = **11.9 hours**

**Maximum Execution Time Per Language Node:**
$37.3 \text{ hours} + 11.9 \text{ hours}$ = **49.2 Physical Wall-Clock Hours**

*(Note: The `eng` English Node will evaluate significantly faster strictly bypassing Translation and shutting down early at 37.3 hours).*

---

## 5. 🚨 Architecture Solutions for the 24-Hour Deadline 🚨
Because 49.2 hours mathematically exceeds your Professor's tight 24-hour turnaround window natively, you must execute one of the following architectural contingencies immediately:

1. **Sub-Sampling Validation (Recommended):** Instead of executing the monstrous `testdev` 12,500-sample limit, dynamically truncate the prediction bound down to the standard 2,500-shot or 1,000-shot distribution (e.g. `subset_1k.json`). This instantly drops your maximum execution time from 49 hours strictly down to **< 5 Hours** yielding rapid, scientifically sound statistical spreads.
2. **GPU Vectorization (Batching):** Instead of sequentially mapping `batch_size=1` organically on `model.chat`, you must rewrite the internal HuggingFace matrices into custom PyTorch dataloaders tracking exactly 4 or 8 bounds seamlessly dropping the 37-hour inference time geometrically!
3. **Hard Sharding by Prompt:** Instead of sharding your 6 GPUs purely by Language, dynamically shard your cluster specifically by workflow (e.g. Node 1 = Italian Short Workflow, Node 2 = Italian Long Workflow).

---

## 6. Local Data Preparation Compute (Image Rendering)
Because the natively compiled visual Watermarks physically inject natively translated text (e.g., Italian strings onto physical pixel geometries), image pools **cannot be shared** across languages. 

* **Images:** `12,500` 
* **Render Settings:** `3` (Watermark, Concat Resize, Concat Pad)
* **Languages:** `5` 

**Total Render Computations:** $12,500 \times 3 \times 5$ = **187,500 Physical Image Files**

Because rendering relies on highly optimized CPU memory (`PIL.Image`) rather than GPU tensors, generating geometries, applying fonts, and saving the patch back functionally to an SSD takes roughly `~0.03` seconds per image natively.

* **Total Render Compute (Strictly Sequential):** 187,500 $\times 0.03$s = `5,625 seconds` = **~1.56 Hours**

*(If you execute the three Python rendering scripts simultaneously across three parallel terminal windows locally on your CPU hardware *before* deploying to the cluster, this completely computationally drops to less than 40 minutes!)*

---

## 7. Deep-Dive: The Evaluation Engine Overhead (`evaluate_model.py`)
If we zoom directly into the computational overhead of the Post-Inference Evaluation block on a single Non-English SLURM GPU node (43 Total Configs, 12,500 samples per config):

1. **Pre-Response Filtering (JSON Extraction):** *(Physically executes ONLY on the 36 Short/Long Workflow Configurations. Ignored explicitly for Baseline/Light/No Prompt)*
   * $36 \text{ Workflow Configs} \times 12,500 = 450,000\text{ computations}$
   * $450,000 \times 0.0001\text{s} = \mathbf{45 \text{ seconds}}$
2. **Back-Translation (SeamlessM4T):** *(Executes heavily on all 43 configs on Non-English nodes natively)*
   * $43 \text{ Configs} \times 12,500 = 537,500\text{ translations}$
   * $537,500 \times 0.08\text{s} = \mathbf{43,000 \text{ seconds (11.94 Hours)}}$
3. **Post-Response Filtering (English Heuristics):** *(Executes iteratively on all 43 configs stripping trailing data)*
   * $43 \text{ Configs} \times 12,500 = 537,500\text{ loops}$
   * $537,500 \times 0.00005\text{s} = \mathbf{26 \text{ seconds}}$
4. **Metric Generation (BERTScore & QAA Math):** *(BERTScore natively computes all 43 configurations. QAA evaluates only the 36 pipelines that passed Pre-filtering!)*
   * $537,500 \times 0.005\text{s} = \mathbf{2,687 \text{ seconds (0.74 Hours)}}$

**Summary:** You are absolutely correct; by violently skipping Pre-Filtering on Baseline models dynamically, your native Python text execution completes entirely inside $\sim 71$ seconds! $99.9\%$ of your physical Evaluation Limit intrinsically rests purely on your GPU Transformers!
