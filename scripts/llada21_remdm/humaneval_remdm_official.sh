#!/usr/bin/env bash
# OFFICIAL ReMDM inference baseline on HumanEval, LLaDA2.1-mini.
# This is the *published* ReMDM algorithm (kuleshov-group/remdm, arXiv:2503.00307),
# not the argmax approximation:
#   * carry_over=True      -> SUBS parameterization (decoded tokens sticky one-hot)
#   * temperature=1.0      -> STOCHASTIC posterior sampling (_sample_categorical)
#   * early_exit_number=0  -> NO early exit (stochastic has no deterministic fixed
#                             point; the official sampler runs the full schedule)
# Accuracy-vs-NFE scaling sweep over the inner-correction budget (edit_step).
#
# num_workers=1 -> one problem at a time so gen_time_s is clean isolated latency.
# Per-sample NFE and wall time are logged incrementally to generations.jsonl, so
# even if a job is stopped early the completed samples retain both metrics.
#
# NOTE: no early exit + stochastic makes high edit_steps VERY slow (NFE grows
# ~linearly with edit_step and nothing terminates the inner loop). Walltime is set
# to the 7-day partition max; partial results are expected/acceptable for step16/32.
#
# Cluster-adapted for the a100 partition (see /home/zhaoyiz/personal/CLAUDE.md).
#SBATCH --job-name=eval_llada21_remdm_official_humaneval
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --partition=a100
#SBATCH --chdir=/home/zhaoyiz/personal/dllm

set -eo pipefail

model_name_or_path="inclusionAI/LLaDA2.1-mini"
max_new_tokens=512
block_size=32
batch_size=1
num_workers=1          # sequential -> clean per-sample wall-time measurement
reuse=true

# Official ReMDM knobs
variant="cap"          # cap | rescale | markovian | conf
eta=0.4
carry_over=True        # SUBS carry-over parameterization
temperature=1.0        # stochastic posterior sampling (official)
early_exit_number=0    # NO early exit (official runs the full schedule)

# Compute sweep: inner correction iterations per outer decode step.
EDIT_STEP_SWEEP=(1 2 4 8 16 32)
edit_step=${EDIT_STEP_SWEEP[${SLURM_ARRAY_TASK_ID:-0}]}

output_dir="results/llada21_humaneval_len${max_new_tokens}/remdm_official_${variant}_eta${eta}_step${edit_step}"

echo "===== OFFICIAL ReMDM eval settings ====="
echo "model=${model_name_or_path} max_new_tokens=${max_new_tokens} block_size=${block_size}"
echo "variant=${variant} eta=${eta} carry_over=${carry_over} temperature=${temperature} edit_step=${edit_step} early_exit_number=${early_exit_number}"
echo "batch_size=${batch_size} num_workers=${num_workers} reuse=${reuse}"
echo "output_dir=${output_dir}"
echo "========================================"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TMPDIR=/home/zhaoyiz/tmp

source /opt/conda/etc/profile.d/conda.sh
conda activate mdm
set -u  # enable nounset only after conda activation (cuda-nvcc activate.d isn't -u clean)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/zhaoyiz/personal/dllm
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

mkdir -p "${output_dir}"

model_args="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},block_size=${block_size},temperature=${temperature},eos_early_stop=True,variant=${variant},eta=${eta},carry_over=${carry_over},edit_step=${edit_step},early_exit_number=${early_exit_number},num_workers=${num_workers},output_dir=${output_dir}"
if [ "${reuse}" = "true" ]; then
  model_args="${model_args},reuse=True"
fi

# Stagger array-task cold starts to avoid the code_eval NFS filelock race.
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 20 ))

python dllm/pipelines/llada21/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 \
    --model llada21_remdm --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "${model_args}" \
    --confirm_run_unsafe_code \
    --output_path "${output_dir}" --log_samples
