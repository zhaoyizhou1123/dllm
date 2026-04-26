#!/usr/bin/env bash
#SBATCH --job-name=eval_llada21_mbpp_plus_quality
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --gres=gpu:h200:1
#SBATCH --time=3:00:00
#SBATCH --chdir=/home/zhaoyiz/projects/dllm

# ------------------------------------------------------------------------
# MBPP+ (mbpp_plus_instruct_llada) with inclusionAI/LLaDA2.1-mini
# using LLaDA21Sampler (block diffusion with iterative editing) — Quality Mode.
#
# Mirrors scripts/llada21/humaneval_quality_deltaai.sh but for MBPP+ on
# the ECE cluster. Sweeps response length:
#   SLURM_ARRAY_TASK_ID 0,1,2 -> max_new_tokens 128, 256, 512
#
# Quality Mode defaults:
#   threshold=0.7  editing_threshold=0.5  temperature=0.0
#   block_size=32  max_post_steps=16  num_to_transfer=1
#
# Submit:
#   sbatch scripts/llada21_mbpp/mbpp_plus_quality_ece.sh
# ------------------------------------------------------------------------

set -euo pipefail

# ===== Array job: map SLURM_ARRAY_TASK_ID -> max_new_tokens =====
RESPONSE_LENGTHS=(2048)
max_new_tokens=${RESPONSE_LENGTHS[${SLURM_ARRAY_TASK_ID:-0}]}

# ===== Defaults (Quality Mode) =====
model_name_or_path="inclusionAI/LLaDA2.1-mini"
block_size=32
threshold=0.9
editing_threshold=0.5
max_post_steps=16
num_to_transfer=1
temperature=0.0
batch_size=1
limit=""
offset=""
output_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --max_new_tokens)     max_new_tokens="$2";     shift 2 ;;
    --block_size)         block_size="$2";          shift 2 ;;
    --threshold)          threshold="$2";           shift 2 ;;
    --editing_threshold)  editing_threshold="$2";   shift 2 ;;
    --max_post_steps)     max_post_steps="$2";      shift 2 ;;
    --num_to_transfer)    num_to_transfer="$2";     shift 2 ;;
    --temperature)        temperature="$2";         shift 2 ;;
    --batch_size)         batch_size="$2";          shift 2 ;;
    --output_dir)         output_dir="$2";          shift 2 ;;
    --limit)              limit="$2";               shift 2 ;;
    --offset)             offset="$2";              shift 2 ;;
    *) echo "Error: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${output_dir}" ]]; then
  output_dir="results/llada21_mbpp_plus_quality_full/len_${max_new_tokens}"
fi

echo "===== Eval settings ====="
echo "model_name_or_path=${model_name_or_path}"
echo "max_new_tokens=${max_new_tokens}  block_size=${block_size}"
echo "threshold=${threshold}  editing_threshold=${editing_threshold}"
echo "max_post_steps=${max_post_steps}  num_to_transfer=${num_to_transfer}"
echo "temperature=${temperature}  batch_size=${batch_size}"
echo "output_dir=${output_dir}"
[[ -n "${limit}" ]] && echo "limit=${limit}"
[[ -n "${offset}" ]] && echo "offset=${offset}"
echo "========================="

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# ===== Conda activation =====
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate dllm

# Match the CUDA version torch was built against (see scripts/eval_gibbs_humaneval_ece.sh).
module load cuda13.0/toolkit

export PYTHONPATH=.:${PYTHONPATH:-}
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
  task="mbpp_plus_instruct_llada"
  samples_json=$(python3 -c "import json; print(json.dumps({\"${task}\": list(range(int(${offset}), int(${offset})+int(${limit})))}))")
  extra_args+=(--samples "${samples_json}")
elif [[ -n "${limit}" ]]; then
  extra_args+=(--limit "${limit}")
fi

python dllm/pipelines/llada21/eval.py \
    --tasks mbpp_plus_instruct_llada --num_fewshot 0 \
    --model llada21 --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},block_size=${block_size},threshold=${threshold},editing_threshold=${editing_threshold},max_post_steps=${max_post_steps},num_to_transfer=${num_to_transfer},temperature=${temperature},eos_early_stop=True" \
    --confirm_run_unsafe_code \
    "${extra_args[@]}"
