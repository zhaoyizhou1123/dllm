#!/usr/bin/env bash
#SBATCH --job-name=eval_proseco_gibbs_edit_bidir_humaneval
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=0-6
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
steps=256
block_size=32
remasking="low_confidence"
temperature=0.0
batch_size=1
limit=""
offset=""
edit_freq=1
edit_start=64
edit_step_sweep=(1 2 4 8 16 32 64)
edit_step="${edit_step_sweep[${SLURM_ARRAY_TASK_ID:-0}]}"
edit_strategy="gibbs_edit"
remasking_strategy="random"
num_workers=4
output_dir="results/proseco_humaneval_len${max_new_tokens}_final/gibbs_edit_bidir_block${block_size}_steps${steps}_editstep${edit_step}_editstart${edit_start}_global-earlyexit"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --max_new_tokens)     max_new_tokens="$2";     shift 2 ;;
    --block_size)         block_size="$2";          shift 2 ;;
    --steps)              steps="$2";               shift 2 ;;
    --remasking)          remasking="$2";           shift 2 ;;
    --temperature)        temperature="$2";         shift 2 ;;
    --batch_size)         batch_size="$2";          shift 2 ;;
    --edit_freq)          edit_freq="$2";           shift 2 ;;
    --edit_step)          edit_step="$2";           shift 2 ;;
    --edit_start)         edit_start="$2";          shift 2 ;;
    --edit_strategy)      edit_strategy="$2";       shift 2 ;;
    --remasking_strategy)  remasking_strategy="$2";  shift 2 ;;
    --num_workers)         num_workers="$2";         shift 2 ;;
    --output_dir)          output_dir="$2";          shift 2 ;;
    --limit)              limit="$2";               shift 2 ;;
    --offset)             offset="$2";              shift 2 ;;
    *) echo "Error: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${output_dir}" ]]; then
  output_dir="results/proseco_humaneval_len${max_new_tokens}/gibbs_edit_bidir_block${block_size}_steps${steps}_editstep${edit_step}"
fi

echo "===== Eval settings ====="
echo "model_name_or_path=${model_name_or_path}"
echo "max_new_tokens=${max_new_tokens}  steps=${steps}  block_size=${block_size}  remasking=${remasking}"
echo "temperature=${temperature}"
echo "edit_freq=${edit_freq}  edit_step=${edit_step}  edit_start=${edit_start}  edit_strategy=${edit_strategy}"
echo "remasking_strategy=${remasking_strategy}"
echo "batch_size=${batch_size}  num_workers=${num_workers}"
echo "output_dir=${output_dir}"
[[ -n "${limit}" ]] && echo "limit=${limit}"
[[ -n "${offset}" ]] && echo "offset=${offset}"
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
if [[ -n "${offset}" ]]; then
  extra_args+=(--offset "${offset}")
fi

python dllm/pipelines/llada/eval.py \
    --tasks humaneval --num_fewshot 0 \
    --model llada_gibbs --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},temperature=${temperature},eos_early_stop=True,cfg_scale=0.0,remasking=${remasking},begin_suppress_tokens=[],edit_freq=${edit_freq},edit_step=${edit_step},edit_start=${edit_start},edit_strategy=${edit_strategy},remasking_strategy=${remasking_strategy},early_exit_number=1,num_workers=${num_workers},output_dir=${output_dir}" \
    --confirm_run_unsafe_code \
    "${extra_args[@]}"
