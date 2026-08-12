#!/usr/bin/env bash
#SBATCH --job-name=eval_llada21_conf_block_std_sweep
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=0-47
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --partition=a100
#SBATCH --chdir=/home/zhaoyiz/personal/dllm

# R2D-STANDARD (edit_strategy=gibbs_standard) HumanEval+ accuracy-vs-NFE curve for
# LLaDA2.1-mini. Companion to conf_block_edit_sweep_local.sh (that one is the EDIT
# variant, gibbs_edit). Uses the carry-over fix in confidence_block_sampler.py: the
# resample-all post-edit pass is skipped for gibbs_standard so committed tokens stay
# sticky (ReMDM SUBS); correction is done inside the gibbs_standard corrector during
# generation. Without the fix R2D-standard collapses (same failure as the SAT
# standard 0.0 baseline before the mdm_correction fix).
#
# 2-D SLURM array: 6 edit_step values x 8 chunks = 48 tasks (--array=0-47).
#   task_id / 8 -> index into EDIT_STEPS ; task_id % 8 -> chunk id.
# All other hyperparameters are held at the reproduced headline config
# (threshold=0.9, early_exit_number=2, ...), matching the edit sweep for a fair
# side-by-side curve.

set -euo pipefail

model_name_or_path="inclusionAI/LLaDA2.1-mini"
max_new_tokens=512
block_size=32
threshold=0.9
min_transfer=1
temperature=0.0
batch_size=1
edit_freq=1
edit_strategy="gibbs_standard"
remasking_strategy="random"
early_exit_number=2
num_workers=4
reuse=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reuse) reuse=true; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# ---- 2-D sweep: edit_step x chunk ----
EDIT_STEPS=(1 2 4 8 16 32)
TOTAL_SAMPLES=164
NUM_CHUNKS=8
TASK=${SLURM_ARRAY_TASK_ID:-0}
edit_step=${EDIT_STEPS[$(( TASK / NUM_CHUNKS ))]}
CHUNK_ID=$(( TASK % NUM_CHUNKS ))

chunk_size=$(( (TOTAL_SAMPLES + NUM_CHUNKS - 1) / NUM_CHUNKS ))
offset=$(( CHUNK_ID * chunk_size ))
limit=$chunk_size
if (( offset + limit > TOTAL_SAMPLES )); then
  limit=$(( TOTAL_SAMPLES - offset ))
fi

output_dir="results/llada21_humaneval_len${max_new_tokens}/confidence${threshold}_block_${edit_strategy}_step${edit_step}_early_exit${early_exit_number}_postedit/chunk_${CHUNK_ID}"

echo "===== Eval settings ====="
echo "model_name_or_path=${model_name_or_path}"
echo "max_new_tokens=${max_new_tokens}  block_size=${block_size}"
echo "threshold=${threshold}  min_transfer=${min_transfer}  temperature=${temperature}"
echo "edit_freq=${edit_freq}  edit_step=${edit_step}  edit_strategy=${edit_strategy}"
echo "remasking_strategy=${remasking_strategy}  early_exit_number=${early_exit_number}"
echo "batch_size=${batch_size}  num_workers=${num_workers}  reuse=${reuse}"
echo "output_dir=${output_dir}"
echo "TASK=${TASK}  edit_step=${edit_step}  CHUNK_ID=${CHUNK_ID}  offset=${offset}  limit=${limit}  (samples ${offset}..$(( offset + limit - 1 )))"
echo "========================="

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# conda's activate.d hooks (e.g. cuda-nvcc) reference unbound vars; relax `set -u` here.
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate mdm
set -u

# Offline: compute nodes have no internet; huggingface.co is blocked everywhere.
# Weights are cached under ~/.cache/huggingface/hub; datasets are staged locally.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTHONPATH=/home/zhaoyiz/personal/dllm
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

# /tmp is a 1G tmpfs (too small for code_eval); use the shared FSx scratch.
export TMPDIR=/home/zhaoyiz/tmp
mkdir -p "${TMPDIR}"

# Per-task code_eval cache to avoid the shared-NFS ArrowInvalid/.nfs-busy race
# when many code_eval procs run concurrently.
export HF_METRICS_CACHE="${TMPDIR}/hf_metrics_${SLURM_ARRAY_JOB_ID:-0}_${TASK}"
export HF_EVALUATE_CACHE="${TMPDIR}/hf_evaluate_${SLURM_ARRAY_JOB_ID:-0}_${TASK}"
mkdir -p "${HF_METRICS_CACHE}" "${HF_EVALUATE_CACHE}"

mkdir -p "${output_dir}"

task="humaneval_instruct_llada"
samples_json=$(python3 -c "import json; print(json.dumps({\"${task}\": list(range(${offset}, ${offset}+${limit}))}))")

model_args="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},block_size=${block_size},threshold=${threshold},min_transfer=${min_transfer},temperature=${temperature},eos_early_stop=True,edit_freq=${edit_freq},edit_step=${edit_step},edit_strategy=${edit_strategy},remasking_strategy=${remasking_strategy},early_exit_number=${early_exit_number},num_workers=${num_workers},output_dir=${output_dir}"
if [ "${reuse}" = "true" ]; then
  model_args="${model_args},reuse=True"
fi

python dllm/pipelines/llada21/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 \
    --model llada21_confidence_block --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "${model_args}" \
    --confirm_run_unsafe_code \
    --output_path "${output_dir}" --log_samples \
    --samples "${samples_json}"
