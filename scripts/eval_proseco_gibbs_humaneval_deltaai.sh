#!/usr/bin/env bash
#SBATCH --job-name=eval_gibbs_humaneval_proseco
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=2
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
# HumanEval (humaneval_instruct_llada) on a finetuned LLaDA checkpoint
# using GibbsSampler (gibbs_standard / gibbs_edit / gibbs_edit_v2).
#
# Submit:
#   sbatch scripts/eval_gibbs_humaneval_deltaai.sh \
#       --model_name_or_path /work/nvme/bgqz/zzhou24/checkpoints/progressive_edit_rstar/step_3500
#
# Optional flags:
#   --num_gpu             (default 1)
#   --edit_freq           (default -1;  -1 disables Gibbs = plain MDLM baseline)
#   --edit_step           (default 10)
#   --edit_strategy       (default gibbs_standard; gibbs_standard | gibbs_edit | gibbs_edit_v2)
#   --remasking_strategy  (default random; random | low_confidence)
#   --keep_original_mask  (default true; true | false. When true, the final
#                          remask of gibbs_standard/gibbs_edit uses the input
#                          sequence's mask pattern from before the edit step;
#                          otherwise falls back to --remasking_strategy.)
#   --gen_len             (default 256; sets max_new_tokens = steps = block_size)
#   --temperature         (default 1.0; 0.0 = greedy)
#   --base_model          (default GSAI-ML/LLaDA-8B-Base; used only when
#                          --model_name_or_path is an FSDP checkpoint dir
#                          containing pytorch_model_fsdp.bin — it will be
#                          auto-converted to <ckpt>_hf before eval.)
#   --output_dir          (optional; default derived as
#                          results/<ckpt_series>/<edit_strategy>_freq<edit_freq>_len<gen_len>)
#   --batch_size          (default 8; samples per generation forward pass.)
#   --limit               (optional; lm-eval --limit. Integer = first N examples
#                          per task, float in (0,1) = fraction. Defaults to full eval.)
# ------------------------------------------------------------------------

set -euo pipefail

# ===== Defaults =====
model_name_or_path="/work/nvme/bgqz/zzhou24/checkpoints/proseco_rstar/step_500"
num_gpu=1
edit_freq=1
edit_step_sweep=(0 10 20 50)   # swept by #SBATCH --array; index 0 when not an array job
edit_step="${edit_step_sweep[${SLURM_ARRAY_TASK_ID:-0}]}"
edit_start=32
edit_strategy="gibbs_standard"
remasking_strategy="random"
keep_original_mask="true"
gen_len=128
temperature=0.0
base_model="GSAI-ML/LLaDA-8B-Base"
output_dir="results/humaneval_len${gen_len}/greedy_${edit_strategy}_original_freq${edit_freq}_step${edit_step}_start${edit_start}"
batch_size=16
limit="96"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --num_gpu)            num_gpu="$2";            shift 2 ;;
    --edit_freq)          edit_freq="$2";          shift 2 ;;
    --edit_step)          edit_step="$2";          shift 2 ;;
    --edit_strategy)      edit_strategy="$2";      shift 2 ;;
    --remasking_strategy) remasking_strategy="$2"; shift 2 ;;
    --keep_original_mask) keep_original_mask="$2"; shift 2 ;;
    --gen_len)            gen_len="$2";            shift 2 ;;
    --temperature)        temperature="$2";        shift 2 ;;
    --base_model)         base_model="$2";         shift 2 ;;
    --output_dir)         output_dir="$2";         shift 2 ;;
    --batch_size)         batch_size="$2";         shift 2 ;;
    --limit)              limit="$2";              shift 2 ;;
    *) echo "Error: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${model_name_or_path}" ]]; then
  echo "Error: --model_name_or_path is required" >&2
  exit 1
fi

# Derive the generation tuple from gen_len — the sampler expects all three.
max_new_tokens="${gen_len}"
steps="${gen_len}"
block_size="${gen_len}"

echo "===== Eval settings ====="
echo "model_name_or_path=${model_name_or_path}"
echo "num_gpu=${num_gpu}"
echo "edit_freq=${edit_freq} edit_step=${edit_step}"
echo "edit_strategy=${edit_strategy} remasking_strategy=${remasking_strategy} keep_original_mask=${keep_original_mask}"
echo "gen_len=${gen_len} (max_new_tokens=steps=block_size)  temperature=${temperature}"
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

# ===== Auto-convert FSDP checkpoint to HF format if needed =====
# accelerator.save_state() writes pytorch_model_fsdp.bin (no config.json), which
# from_pretrained can't load. Convert once into a sibling _hf dir, then evaluate.
if [[ -d "${model_name_or_path}" \
      && -f "${model_name_or_path}/pytorch_model_fsdp.bin" \
      && ! -f "${model_name_or_path}/config.json" ]]; then
  echo "[eval_gibbs_humaneval] Detected FSDP checkpoint at ${model_name_or_path}; converting to HF format..."
  resolved_model=$(python dllm/tools/convert_fsdp_checkpoint.py \
      --checkpoint_dir "${model_name_or_path}" \
      --base_model "${base_model}" | tail -n 1)
  echo "[eval_gibbs_humaneval] Using converted checkpoint: ${resolved_model}"
  model_name_or_path="${resolved_model}"
fi

extra_args=()
if [[ -n "${output_dir}" ]]; then
  mkdir -p "${output_dir}"
  extra_args+=(--output_path "${output_dir}" --log_samples)
  echo "[eval_gibbs_humaneval] Saving results + per-sample generations under: ${output_dir}"
fi
if [[ -n "${limit}" ]]; then
  extra_args+=(--limit "${limit}")
  echo "[eval_gibbs_humaneval] Subsetting eval with --limit ${limit}"
fi

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 \
    --model llada_gibbs --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},temperature=${temperature},cfg_scale=0.0,suppress_tokens=[126081],begin_suppress_tokens=[],edit_freq=${edit_freq},edit_step=${edit_step},edit_start=${edit_start},edit_strategy=${edit_strategy},remasking_strategy=${remasking_strategy},keep_original_mask=${keep_original_mask}" \
    --confirm_run_unsafe_code \
    "${extra_args[@]}"
