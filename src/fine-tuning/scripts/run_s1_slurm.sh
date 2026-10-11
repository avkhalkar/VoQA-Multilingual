#!/bin/bash
#SBATCH --job-name=s1_finetune
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=mig45g,mig90g
#SBATCH --gres=gpu:1

echo "=========================================================="
echo "    Launching Stage 1 DeepSpeed Fine-Tuning on SLURM      "
echo "=========================================================="

# Activate exactly the internvl environment where Deepspeed and PIP torch reside
source /home/anandk03/miniconda3/etc/profile.d/conda.sh
conda activate internvl

# Supress the pynvml deprecation warning to keep logs clean
pip install nvidia-ml-py --quiet

# Execute the local fine tuning script seamlessly passing the subset flag!
bash run_stage1_english_watermarked.sh --subset

echo "[*] DONE."
