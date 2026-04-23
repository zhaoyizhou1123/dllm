#!/usr/bin/env bash
#SBATCH --job-name=eval_llada21_mdm_humaneval
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=0-2
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
# using standard MDM sampling (single block = gen_len).
#
# Tokens unmasked per step is swept via --array over (1 2 4).
# steps = gen_len / unmask_per_step.
#
# Submit:
#   sbatch scripts/llada21/humaneval_mdm_deltaai.sh
#
# Optional flags:
#   --model_name_or_path  (default inclusionAI/LLaDA2.1-mini)
#   --gen_len             (default 512; max_new_tokens = block_size)
#   --temperature         (default 0.0)
#   --batch_size          (default 4)
#   --output_dir          (optional)
#   --limit               (optional; lm-eval --limit)
# ------------------------------------------------------------------------

set -euo pipefail

# ===== Defaults =====
model_name_or_path="inclusionAI/LLaDA2.1-mini"
gen_len=512
unmask_sweep=(1 2 4)
unmask_per_step="${unmask_sweep[${SLURM_ARRAY_TASK_ID:-0}]}"
temperature=0.0
batch_size=4
output_dir=""
limit="16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --gen_len)            gen_len="$2";            shift 2 ;;
    --unmask_per_step)    unmask_per_step="$2";    shift 2 ;;
    --temperature)        temperature="$2";        shift 2 ;;
    --batch_size)         batch_size="$2";         shift 2 ;;
    --output_dir)         output_dir="$2";         shift 2 ;;
    --limit)              limit="$2";              shift 2 ;;
    *) echo "Error: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

max_new_tokens="${gen_len}"
block_size="${gen_len}"
steps=$((gen_len / unmask_per_step))

if [[ -z "${output_dir}" ]]; then
  output_dir="results/llada21_humaneval_len${gen_len}/mdm_unmask${unmask_per_step}"
fi

echo "===== Eval settings ====="
echo "model_name_or_path=${model_name_or_path}"
echo "gen_len=${gen_len}  steps=${steps}  unmask_per_step=${unmask_per_step}  temperature=${temperature}"
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
  echo "[eval_llada21_mdm] Saving results under: ${output_dir}"
fi
if [[ -n "${limit}" ]]; then
  extra_args+=(--limit "${limit}")
  echo "[eval_llada21_mdm] Subsetting eval with --limit ${limit}"
fi

python dllm/pipelines/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 \
    --model llada --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},temperature=${temperature},cfg_scale=0.0,suppress_tokens=[156892],begin_suppress_tokens=[]" \
    --confirm_run_unsafe_code \
    "${extra_args[@]}"
