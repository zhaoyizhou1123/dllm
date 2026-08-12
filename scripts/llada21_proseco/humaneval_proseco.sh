#!/usr/bin/env bash
# ProSeCo inference-only baseline on HumanEval, LLaDA2.1-mini.
# ProSeCo = the "no-remask (t=0)" corner: unmask by confidence + re-predict all
# revealed tokens each step. Accuracy-vs-NFE scaling sweep over correction_step.
#
# Cluster-adapted for the a100 partition (see /home/zhaoyiz/personal/CLAUDE.md).
# Each array task = one sweep point, running the full 164 problems.
#SBATCH --job-name=eval_llada21_proseco_humaneval
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --chdir=/home/zhaoyiz/personal/dllm

set -eo pipefail

model_name_or_path="inclusionAI/LLaDA2.1-mini"
max_new_tokens=512
block_size=32
temperature=0.0
batch_size=1
unmasking_num=1
num_workers=4
reuse=true

# Compute sweep: extra full-block re-prediction passes after reveal.
CORRECTION_STEP_SWEEP=(0 1 2 4 8 16 32 64)
correction_step=${CORRECTION_STEP_SWEEP[${SLURM_ARRAY_TASK_ID:-0}]}

output_dir="results/llada21_humaneval_len${max_new_tokens}/proseco_unmask${unmasking_num}_corr${correction_step}"

echo "===== ProSeCo eval settings ====="
echo "model=${model_name_or_path} max_new_tokens=${max_new_tokens} block_size=${block_size}"
echo "unmasking_num=${unmasking_num} correction_step=${correction_step}"
echo "batch_size=${batch_size} num_workers=${num_workers} reuse=${reuse}"
echo "output_dir=${output_dir}"
echo "================================="

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TMPDIR=/home/zhaoyiz/tmp

source /opt/conda/etc/profile.d/conda.sh
conda activate mdm
set -u  # enable nounset only after conda activation (cuda-nvcc activate.d isn't -u clean)

# Compute nodes are offline; HF hub is blocked. Weights + data are cached/staged.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/zhaoyiz/personal/dllm
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

mkdir -p "${output_dir}"

model_args="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},block_size=${block_size},temperature=${temperature},eos_early_stop=True,unmasking_num=${unmasking_num},correction_step=${correction_step},num_workers=${num_workers},output_dir=${output_dir}"
if [ "${reuse}" = "true" ]; then
  model_args="${model_args},reuse=True"
fi

# Stagger array-task cold starts: the code_eval metric runs a warm-up compute() at
# task import that grabs a shared cache filelock on NFS; simultaneous starts can race
# (OSError 116 Stale file handle). A per-index sleep avoids the collision.
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 20 ))

python dllm/pipelines/llada21/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 \
    --model llada21_proseco --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "${model_args}" \
    --confirm_run_unsafe_code \
    --output_path "${output_dir}" --log_samples
