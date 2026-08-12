#!/usr/bin/env bash
# QUICK TEST: ReMDM carry-over vs no-carry on a matched small problem subset.
# Both runs: argmax (temperature=0), early_exit_number=5, edit_step=8, variant=cap.
# Goal: see whether SUBS carry-over changes HumanEval+ accuracy on LLaDA2.1-mini.
#SBATCH --job-name=remdm_carryover_test
#SBATCH --output=slurm/%x/job_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --partition=a100
#SBATCH --chdir=/home/zhaoyiz/personal/dllm

set -eo pipefail

model_name_or_path="inclusionAI/LLaDA2.1-mini"
task="humaneval_instruct_llada"
# 12-problem spread across the 164 for a quick directional signal.
samples_json='{"'"$task"'": [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132]}'

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TMPDIR=/home/zhaoyiz/tmp

source /opt/conda/etc/profile.d/conda.sh
conda activate mdm
set -u

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/zhaoyiz/personal/dllm
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=True

run () {
  local name="$1"; local carry="$2"
  local out="results/_carryover_test/${name}"
  rm -rf "$out"; mkdir -p "$out"
  echo "########## ${name}  (carry_over=${carry}) ##########"
  python dllm/pipelines/llada21/eval.py \
    --tasks "$task" --num_fewshot 0 \
    --model llada21_remdm --apply_chat_template \
    --batch_size 1 \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=512,block_size=32,temperature=0.0,eos_early_stop=True,variant=cap,eta=0.4,edit_step=8,early_exit_number=5,carry_over=${carry},num_workers=4,output_dir=${out}" \
    --confirm_run_unsafe_code \
    --output_path "$out" --log_samples \
    --samples "$samples_json"
}

run "carry"    "True"
run "nocarry"  "False"
echo "########## CARRYOVER TEST DONE ##########"
