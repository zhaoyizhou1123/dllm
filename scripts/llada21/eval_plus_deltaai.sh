#!/usr/bin/env bash
#SBATCH --job-name=eval_humaneval_plus
#SBATCH --output=slurm/%x/job_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-node=16
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --partition=ghx4-interactive
#SBATCH --account=bgqz-dtai-gh
#SBATCH --chdir=/u/zzhou24/projects/dllm

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ~/miniconda3/bin/activate
conda activate smdm2

export HF_HOME=/projects/bgqz/zzhou24/.cache/huggingface
export PYTHONPATH=/u/zzhou24/projects/dllm
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

python scripts/llada21/eval_humaneval_plus.py \
    --samples_jsonl results/proseco_humaneval_len256/gibbs_edit_bidir_block32_steps256_editstep32_editstart64_global-earlyexit/kuleshov-group__proseco-llada-sft/samples_humaneval_2026-04-30T01-27-39.043443.jsonl