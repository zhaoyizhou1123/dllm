#!/usr/bin/env bash
#SBATCH --job-name=eval_llada21_mdm_block_humaneval
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --partition=ghx4-interactive
#SBATCH --account=bgqz-dtai-gh
#SBATCH --chdir=/u/zzhou24/projects/dllm

# ------------------------------------------------------------------------
# HumanEval (humaneval_instruct_llada) with inclusionAI/LLaDA2.1-mini
# using standard block diffusion with block-causal attention (block_size=32).
#
# Uses LLaDA21BlockSampler (fixed-schedule topk, no editing).
# Steps per block is swept via --array over (32 16 8), corresponding
# to unmasking (1 2 4) tokens per step within each block.
#
# Submit:
#   sbatch scripts/llada21/humaneval_mdm_block_deltaai.sh
#
# Optional flags:
#   --model_name_or_path  (default inclusionAI/LLaDA2.1-mini)
#   --max_new_tokens      (default 256)
#   --block_size          (default 32)
#   --temperature         (default 0.0)
#   --batch_size          (default 4)
#   --output_dir          (optional)
#   --limit               (optional; lm-eval --limit)
# ------------------------------------------------------------------------

set -euo pipefail

# ===== Defaults =====
model_name_or_path="inclusionAI/LLaDA2.1-mini"
max_new_tokens=256
block_size=32
# Sweep: steps_per_block = block_size / unmask_per_step → (32 16 8)
steps_sweep=(32 16 8 4)
steps_per_block="${steps_sweep[${SLURM_ARRAY_TASK_ID:-0}]}"
unmask_per_step=$((block_size / steps_per_block))
temperature=0.0
batch_size=1
limit="16"
output_dir="results/llada21_humaneval_len${max_new_tokens}_limit${limit}/block_unmask${unmask_per_step}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --max_new_tokens)     max_new_tokens="$2";     shift 2 ;;
    --block_size)         block_size="$2";          shift 2 ;;
    --steps_per_block)    steps_per_block="$2";     shift 2 ;;
    --temperature)        temperature="$2";         shift 2 ;;
    --batch_size)         batch_size="$2";         shift 2 ;;
    --output_dir)         output_dir="$2";         shift 2 ;;
    --limit)              limit="$2";              shift 2 ;;
    *) echo "Error: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

unmask_per_step=$((block_size / steps_per_block))

if [[ -z "${output_dir}" ]]; then
  output_dir="results/llada21_humaneval_mdm_block${block_size}_unmask${unmask_per_step}"
fi

echo "===== Eval settings ====="
echo "model_name_or_path=${model_name_or_path}"
echo "max_new_tokens=${max_new_tokens}  block_size=${block_size}  steps_per_block=${steps_per_block}"
echo "unmask_per_step=${unmask_per_step}  temperature=${temperature}"
echo "batch_size=${batch_size}"
echo "output_dir=${output_dir}"
[[ -n "${limit}" ]] && echo "limit=${limit}"
echo "========================="

# ===== Conda activation =====
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ~/miniconda3/bin/activate
conda activate smdm2

# NCCL fix for deltaai cluster
unset NCCL_NET_PLUGIN
LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/sw/user/nccl/' | tr '\n' ':' | sed 's/:*$//')
export LD_LIBRARY_PATH

export HF_HOME=/projects/bgqz/zzhou24/.cache/huggingface
export PYTHONPATH=/u/zzhou24/projects/dllm
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

extra_args=()
if [[ -n "${output_dir}" ]]; then
  mkdir -p "${output_dir}"
  extra_args+=(--output_path "${output_dir}" --log_samples)
  echo "[eval_llada21_mdm_block] Saving results under: ${output_dir}"
fi
if [[ -n "${limit}" ]]; then
  extra_args+=(--limit "${limit}")
  echo "[eval_llada21_mdm_block] Subsetting eval with --limit ${limit}"
fi

python dllm/pipelines/llada21/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 \
    --model llada21_block --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps_per_block=${steps_per_block},block_size=${block_size},temperature=${temperature},eos_early_stop=True" \
    --confirm_run_unsafe_code \
    "${extra_args[@]}"
