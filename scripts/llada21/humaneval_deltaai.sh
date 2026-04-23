#!/usr/bin/env bash
#SBATCH --job-name=eval_llada21_humaneval
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=0
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
# using LLaDA21Sampler (block diffusion with iterative editing).
#
# Default sampling parameters match the HF model card (Speed Mode):
#   threshold=0.5  editing_threshold=0.0  temperature=0.0
#   block_size=32  max_new_tokens=512  max_post_steps=16
#
# Submit:
#   sbatch scripts/llada21/humaneval_deltaai.sh
#
# Optional flags:
#   --model_name_or_path  (default inclusionAI/LLaDA2.1-mini)
#   --max_new_tokens      (default 512)
#   --block_size          (default 32)
#   --threshold           (default 0.5)
#   --editing_threshold   (default 0.0; set 0.5 for Quality Mode)
#   --max_post_steps      (default 16)
#   --temperature         (default 0.0)
#   --output_dir          (optional; default results/llada21_humaneval)
#   --limit               (optional; lm-eval --limit)
# ------------------------------------------------------------------------

set -euo pipefail

# ===== Defaults (HF Speed Mode) =====
model_name_or_path="inclusionAI/LLaDA2.1-mini"
max_new_tokens=512
block_size=32
threshold=0.5
editing_threshold=0.0
max_post_steps=4
num_to_transfer=1
temperature=0.0
limit="16"
offset="65"  # last 16
output_dir=""  # set after arg parsing

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --block_size)         block_size="$2";          shift 2 ;;
    --threshold)          threshold="$2";           shift 2 ;;
    --editing_threshold)  editing_threshold="$2";   shift 2 ;;
    --max_post_steps)     max_post_steps="$2";      shift 2 ;;
    --num_to_transfer)    num_to_transfer="$2";     shift 2 ;;
    --temperature)        temperature="$2";         shift 2 ;;
    --output_dir)         output_dir="$2";          shift 2 ;;
    --limit)              limit="$2";               shift 2 ;;
    --offset)             offset="$2";              shift 2 ;;
    *) echo "Error: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Build output_dir from max_new_tokens and limit
if [[ -z "${output_dir}" ]]; then
  output_dir="results/llada21_humaneval_len${max_new_tokens}_middle_limit${limit}/default_max_post_steps${max_post_steps}"
fi

echo "===== Eval settings ====="
echo "model_name_or_path=${model_name_or_path}"
echo "max_new_tokens=${max_new_tokens}  block_size=${block_size}"
echo "threshold=${threshold}  editing_threshold=${editing_threshold}"
echo "max_post_steps=${max_post_steps}  num_to_transfer=${num_to_transfer}"
echo "temperature=${temperature}"
echo "output_dir=${output_dir}"
[[ -n "${limit}" ]] && echo "limit=${limit}"
[[ -n "${offset}" ]] && echo "offset=${offset}"
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
fi
if [[ -n "${offset}" && -n "${limit}" ]]; then
  # Use --samples to select a specific slice [offset, offset+limit)
  task="humaneval_instruct_llada"
  samples_json=$(python3 -c "import json; print(json.dumps({\"${task}\": list(range(int(${offset}), int(${offset})+int(${limit})))}))")
  extra_args+=(--samples "${samples_json}")
elif [[ -n "${limit}" ]]; then
  extra_args+=(--limit "${limit}")
fi

python dllm/pipelines/llada21/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 \
    --model llada21 --apply_chat_template \
    --batch_size 1 \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},block_size=${block_size},threshold=${threshold},editing_threshold=${editing_threshold},max_post_steps=${max_post_steps},num_to_transfer=${num_to_transfer},temperature=${temperature},eos_early_stop=True" \
    --confirm_run_unsafe_code \
    "${extra_args[@]}"
