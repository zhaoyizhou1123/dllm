#!/usr/bin/env bash
#SBATCH --job-name=he_compute
#SBATCH --output=slurm/%x/job_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=ghx4
#SBATCH --account=bgqz-dtai-gh
#SBATCH --chdir=/u/zzhou24/projects/dllm

# ------------------------------------------------------------------------
# HumanEval+ compute measurement (per-sample wall time, NFE) on LLaDA2.1-mini.
# Report: r2d-expt/final_experiments/reports/D_compute/D2_humaneval_compute.md
#
# One method per job, one GPU per job, batch 1, num_workers=1 (sequential ->
# the harness' cuda-synchronized `gen_time_s` is an uncontended latency), plus a
# discarded one-block warmup. Per-sample `nfe` / `gen_time_s` land in
# ${output_dir}/generations.jsonl.
#
#   METHOD   config                                                 samples
#   r2d      confidence_block gibbs_edit, edit_step=32 (32/token)    probe[TASK], array 0-4
#   proseco  proseco, correction_step=32                             probe[0]
#   remdm    remdm cap eta=0.4 no-carry stoch, edit_step=31 (32/tok) probe[0]
#   remdm_ee5 rebuttal ReMDM column (humaneval_remdm.sh): cap eta=0.4, no-carry,
#            argmax, early_exit 5, edit_step=32                     probe[TASK], array 0-4
#   qmode    official LLaDA21Sampler, threshold 0.7 / edit 0.5       all 164
#   smode    official LLaDA21Sampler, threshold 0.5 / edit 0.0       all 164
#
# probe = np.random.default_rng(0).choice(164, 5, replace=False), in draw order.
#
# Full-set mode: CHUNKS=8 --array=0-7 runs all 164 problems (21 per chunk) under
# results/.../compute_full/<METHOD>/chunk_<i>/ (report-exact configs at step 32).
#
# Submit (see submit_compute_walltime.sh):
#   sbatch --export=ALL,METHOD=r2d --array=0-4 scripts/llada21/compute_walltime.sh
#   sbatch --export=ALL,METHOD=remdm           scripts/llada21/compute_walltime.sh
# ------------------------------------------------------------------------

set -eo pipefail
: "${METHOD:?METHOD must be one of r2d|proseco|remdm|remdm_ee5|qmode|smode}"

model_name_or_path="inclusionAI/LLaDA2.1-mini"
max_new_tokens=${MAX_NEW_TOKENS:-512}   # smoke-test overrides: MAX_NEW_TOKENS, N_ALL, OUT_ROOT
block_size=32
TOTAL_SAMPLES=164
TASK=${SLURM_ARRAY_TASK_ID:-0}
common="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},block_size=${block_size},eos_early_stop=True,num_workers=1,warmup=1,reuse=True"

case "${METHOD}" in
  r2d)
    model=llada21_confidence_block; seed=1234; probe_idx=${TASK}
    margs="threshold=0.9,min_transfer=1,temperature=0.0,edit_freq=1,edit_step=32,edit_strategy=gibbs_edit,remasking_strategy=random,early_exit_number=2" ;;
  proseco)
    model=llada21_proseco; seed=1234; probe_idx=0
    margs="temperature=0.0,unmasking_num=1,correction_step=32" ;;
  remdm)
    model=llada21_remdm; seed=1234; probe_idx=0
    margs="temperature=1.0,variant=cap,eta=0.4,carry_over=False,edit_step=31,early_exit_number=0" ;;
  remdm_ee5)
    model=llada21_remdm; seed=1234; probe_idx=${TASK}
    margs="temperature=0.0,variant=cap,eta=0.4,edit_step=32,early_exit_number=5" ;;
  qmode)
    model=llada21; seed=1234; probe_idx=""
    margs="temperature=0.0,threshold=0.7,editing_threshold=0.5,max_post_steps=16,num_to_transfer=1" ;;
  smode)
    model=llada21; seed=1234; probe_idx=""
    margs="temperature=0.0,threshold=0.5,editing_threshold=0.0,max_post_steps=16,num_to_transfer=1" ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 1 ;;
esac

task="humaneval_instruct_llada"
if [[ -n "${CHUNKS:-}" ]]; then
  # Full 164-problem run split into CHUNKS array tasks (TASK = chunk id), same
  # ceil(164/CHUNKS) split as conf_block_edit_sweep_local.sh -> with CHUNKS=8, R2D's
  # per-chunk RNG stream matches the rebuttal sweep.
  chunk_size=$(( (TOTAL_SAMPLES + CHUNKS - 1) / CHUNKS ))
  offset=$(( TASK * chunk_size ))
  limit=$(( offset + chunk_size > TOTAL_SAMPLES ? TOTAL_SAMPLES - offset : chunk_size ))
  sel_args=(--samples "$(python3 -c "import json; print(json.dumps({'${task}': list(range(${offset}, ${offset}+${limit}))}))")")
  output_dir="results/llada21_humaneval_len${max_new_tokens}/${OUT_ROOT:-compute_full}/${METHOD}/chunk_${TASK}"
elif [[ -n "${probe_idx}" ]]; then
  doc_id=$(python3 -c "import numpy as np; print(int(np.random.default_rng(0).choice(${TOTAL_SAMPLES}, 5, replace=False)[${probe_idx}]))")
  sel_args=(--samples "{\"${task}\": [${doc_id}]}")
  output_dir="results/llada21_humaneval_len${max_new_tokens}/${OUT_ROOT:-compute}/${METHOD}/doc_${doc_id}"
else
  # docs [0, N_ALL): --limit, since a 164-id --samples string overflows the
  # filename lm-eval first tries to open it as.
  sel_args=(--limit "${N_ALL:-${TOTAL_SAMPLES}}")
  output_dir="results/llada21_humaneval_len${max_new_tokens}/${OUT_ROOT:-compute}/${METHOD}"
fi

echo "METHOD=${METHOD} model=${model} seed=${seed} probe_idx=${probe_idx:-all}"
echo "model_args=${common},${margs}"
echo "output_dir=${output_dir}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
source ~/miniconda3/bin/activate
conda activate smdm2
set -u  # only after conda activation (activate.d hooks are not -u clean)

unset NCCL_NET_PLUGIN
LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/sw/user/nccl/' | tr '\n' ':' | sed 's/:*$//')
export LD_LIBRARY_PATH

export HF_HOME=/projects/bgqz/zzhou24/.cache/huggingface
export PYTHONPATH=/u/zzhou24/projects/dllm
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TMPDIR=/projects/bgqz/zzhou24/tmp
export HF_METRICS_CACHE="${TMPDIR}/hf_metrics_${SLURM_JOB_ID:-0}"
export HF_EVALUATE_CACHE="${TMPDIR}/hf_evaluate_${SLURM_JOB_ID:-0}"
mkdir -p "${TMPDIR}" "${HF_METRICS_CACHE}" "${HF_EVALUATE_CACHE}" "${output_dir}"

nvidia-smi --query-gpu=name --format=csv,noheader | tee "${output_dir}/gpu.txt"

python dllm/pipelines/llada21/eval.py \
    --tasks "${task}" --num_fewshot 0 \
    --model "${model}" --apply_chat_template \
    --batch_size 1 \
    --model_args "${common},${margs},output_dir=${output_dir}" \
    --seed "0,1234,${seed},1234" \
    --confirm_run_unsafe_code \
    --output_path "${output_dir}" --log_samples \
    "${sel_args[@]}"
