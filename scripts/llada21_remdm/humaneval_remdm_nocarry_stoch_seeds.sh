#!/usr/bin/env bash
# ReMDM (NO carry-over + STOCHASTIC) on HumanEval / HumanEval+, LLaDA2.1-mini.
#
# Config under test:
#   carry_over=False      -> raw-softmax "resample-all" p_x0 (NOT the official SUBS
#                            parameterization; decoded tokens are re-drawn each inner
#                            iteration rather than pinned one-hot)
#   temperature=1.0       -> stochastic posterior sampling via _sample_categorical
#   early_exit_number=0   -> NO early exit. A stochastic sampler has no deterministic
#                            fixed point, so the argmax-stability test is meaningless
#                            and would truncate the schedule arbitrarily.
#
# Compute axis: steps/token = 1 + edit_step exactly (num_outer = block_size /
# unmasking_num = 32 outer steps per 32-token block, each running 1 + edit_step
# forwards; no early exit => the bound is tight). The sweep therefore caps at
# edit_step=15 -> 16 steps/token, on a power-of-two NFE axis:
#   edit_step  0  1  3  7 15
#   steps/tok  1  2  4  8 16
#
# 3 seeds per config (1234 / 1 / 2) matching scripts/llada21/scaling_curve_seeds.py,
# varied through lm-eval's --seed (torch slot only). edit_step=0 is NOT deterministic
# here (temperature=1.0 still draws from q_xs), so all 5 points get all 3 seeds.
#
# 164 problems are split into NUM_CHUNKS SLURM array tasks; recombine with
#   python /u/zzhou24/projects/dllm/scripts/llada21/combine_humaneval_chunks.py \
#       --base_dir <config_dir>
# then score HumanEval+ with
#   python /u/zzhou24/projects/dllm/scripts/llada21/eval_humaneval_plus.py ...
#
# Array layout (5 edit_steps x 3 seeds x 8 chunks = 120 tasks):
#   chunk    = task %  8
#   seed     = (task /  8) % 3
#   edit_step= task / 24
#
# Submit:
#   sbatch /u/zzhou24/projects/dllm/scripts/llada21_remdm/humaneval_remdm_nocarry_stoch_seeds.sh
#
#SBATCH --job-name=eval_llada21_remdm_nocarry_stoch
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --array=0-119
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --partition=ghx4
#SBATCH --account=bgqz-dtai-gh
#SBATCH --chdir=/u/zzhou24/projects/dllm

set -eo pipefail

model_name_or_path="inclusionAI/LLaDA2.1-mini"
max_new_tokens=512
block_size=32
batch_size=1
num_workers=4
reuse=true

# ReMDM knobs (the config under test)
variant="cap"          # cap | rescale | markovian | conf
eta=0.4
carry_over=False       # no SUBS carry-over
temperature=1.0        # stochastic
early_exit_number=0    # no early exit

# ---- 3-D sweep: edit_step x seed x chunk ----
EDIT_STEPS=(0 1 3 7 15)   # -> 1 2 4 8 16 steps/token
SEEDS=(1234 1 2)
NUM_CHUNKS=8
TOTAL_SAMPLES=164

TASK=${SLURM_ARRAY_TASK_ID:-0}
CHUNK_ID=$(( TASK % NUM_CHUNKS ))
SEED=${SEEDS[$(( (TASK / NUM_CHUNKS) % ${#SEEDS[@]} ))]}
edit_step=${EDIT_STEPS[$(( TASK / (NUM_CHUNKS * ${#SEEDS[@]}) ))]}

chunk_size=$(( (TOTAL_SAMPLES + NUM_CHUNKS - 1) / NUM_CHUNKS ))
offset=$(( CHUNK_ID * chunk_size ))
limit=$chunk_size
if (( offset + limit > TOTAL_SAMPLES )); then
  limit=$(( TOTAL_SAMPLES - offset ))
fi

config_dir="results/llada21_humaneval_len${max_new_tokens}/remdm_nocarry_stoch_${variant}_eta${eta}_step${edit_step}_seed${SEED}"
output_dir="${config_dir}/chunk_${CHUNK_ID}"

echo "===== ReMDM (no carry-over, stochastic) eval settings ====="
echo "model=${model_name_or_path}  max_new_tokens=${max_new_tokens}  block_size=${block_size}"
echo "variant=${variant}  eta=${eta}  carry_over=${carry_over}  temperature=${temperature}"
echo "edit_step=${edit_step}  early_exit_number=${early_exit_number}  steps/token=$(( 1 + edit_step ))"
echo "batch_size=${batch_size}  num_workers=${num_workers}  reuse=${reuse}"
echo "TASK=${TASK}  SEED=${SEED}  CHUNK_ID=${CHUNK_ID}  samples ${offset}..$(( offset + limit - 1 ))"
echo "output_dir=${output_dir}"
echo "==========================================================="

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ~/miniconda3/bin/activate
conda activate smdm2
set -u  # only after conda activation (activate.d hooks are not -u clean)

# DeltaAI: the site NCCL plugin breaks single-GPU runs; drop it.
unset NCCL_NET_PLUGIN
LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/sw/user/nccl/' | tr '\n' ':' | sed 's/:*$//')
export LD_LIBRARY_PATH

export HF_HOME=/projects/bgqz/zzhou24/.cache/huggingface
export PYTHONPATH=/u/zzhou24/projects/dllm
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

# Isolate the HF `evaluate` code_eval metric cache per array task: the shared default
# dir races on NFS when many array tasks start at once (ArrowInvalid / stale handle).
export TMPDIR=/projects/bgqz/zzhou24/tmp
mkdir -p "${TMPDIR}"
export HF_METRICS_CACHE="${TMPDIR}/hf_metrics_${SLURM_ARRAY_JOB_ID:-0}_${TASK}"
export HF_EVALUATE_CACHE="${TMPDIR}/hf_evaluate_${SLURM_ARRAY_JOB_ID:-0}_${TASK}"
mkdir -p "${HF_METRICS_CACHE}" "${HF_EVALUATE_CACHE}"

mkdir -p "${output_dir}"

task="humaneval_instruct_llada"
samples_json=$(python3 -c "import json; print(json.dumps({\"${task}\": list(range(${offset}, ${offset}+${limit}))}))")

model_args="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},block_size=${block_size},temperature=${temperature},eos_early_stop=True,variant=${variant},eta=${eta},carry_over=${carry_over},edit_step=${edit_step},early_exit_number=${early_exit_number},num_workers=${num_workers},output_dir=${output_dir}"
if [ "${reuse}" = "true" ]; then
  model_args="${model_args},reuse=True"
fi

# Stagger cold starts to avoid the code_eval warm-up filelock race.
sleep $(( TASK * 10 ))

# Vary only the torch seed (3rd slot); keep random / numpy / fewshot at defaults.
python dllm/pipelines/llada21/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 \
    --model llada21_remdm --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "${model_args}" \
    --seed "0,1234,${SEED},1234" \
    --confirm_run_unsafe_code \
    --output_path "${output_dir}" --log_samples \
    --samples "${samples_json}"
