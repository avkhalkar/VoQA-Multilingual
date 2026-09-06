#!/bin/bash
#SBATCH --job-name=voqa_multi_grid
#SBATCH --output=voqa_job_%j.out
#SBATCH --error=voqa_job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:3g.90gb:1
#SBATCH --time=48:00:00
#SBATCH --partition=mig90g

echo "=========================================="
echo "    VoQA MULTILINGUAL PEFT PIPELINE       "
echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "CUDA_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "=========================================="
nvidia-smi -L
echo "=========================================="

# 1. Activate Environment
# Modify this if your cluster uses a different method to load conda (e.g. module load)
source ~/.bashrc
conda activate internvl

# 2. Navigate to Project Root
# Assuming you submit this from the root of your VoQA-Multilingual repository!
cd "${SLURM_SUBMIT_DIR}"

# 3. Grant execution permissions
chmod +x ./src/fine-tuning/scripts/execute_pipeline.sh

# 4. DISPATCH PIPELINE!
echo "[*] Triggering Master Orchestrator for ALL 4 Stages..."
./src/fine-tuning/scripts/execute_pipeline.sh --stage all

echo "=========================================="
echo "[*] PIPELINE COMPLETED SUCCESSFULLY."
echo "=========================================="
