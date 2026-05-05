#!/usr/bin/env bash
#SBATCH --job-name=eval_proseco_baseline_humaneval
#SBATCH --output=slurm/%x/job_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --partition=ghx4-interactive
#SBATCH --account=bgqz-dtai-gh
#SBATCH --chdir=/u/zzhou24/projects/dllm

set -euo pipefail

model_name_or_path="kuleshov-group/proseco-llada-sft"
max_new_tokens=256
block_size=32
steps=256
temperature=0.0
batch_size=1
limit=""
output_dir="results/proseco_humaneval_len${max_new_tokens}_notemplate/baseline"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --max_new_tokens)     max_new_tokens="$2";     shift 2 ;;
    --block_size)         block_size="$2";          shift 2 ;;
    --steps)              steps="$2";               shift 2 ;;
    --temperature)        temperature="$2";         shift 2 ;;
    --batch_size)         batch_size="$2";          shift 2 ;;
    --output_dir)         output_dir="$2";          shift 2 ;;
    --limit)              limit="$2";               shift 2 ;;
    *) echo "Error: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "===== Eval settings ====="
echo "model_name_or_path=${model_name_or_path}"
echo "max_new_tokens=${max_new_tokens}  block_size=${block_size}  steps=${steps}"
echo "temperature=${temperature}"
echo "batch_size=${batch_size}"
echo "output_dir=${output_dir}"
[[ -n "${limit}" ]] && echo "limit=${limit}"
echo "========================="

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ~/miniconda3/bin/activate
conda activate smdm2

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
if [[ -n "${limit}" ]]; then
  extra_args+=(--limit "${limit}")
fi

python dllm/pipelines/llada/eval.py \
    --tasks humaneval --num_fewshot 0 \
    --model llada \
    --batch_size "${batch_size}" \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},block_size=${block_size},steps=${steps},temperature=${temperature},remasking=low_confidence,eos_early_stop=True,begin_suppress_tokens=[]" \
    --confirm_run_unsafe_code \
    "${extra_args[@]}"
