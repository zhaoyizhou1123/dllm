#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Re-evaluate existing MBPP generations against the MBPP+ test suite.
#
# Reuses the already-sanitized predictions in a JSONL produced by
# mbpp_plus_instruct_llada and scores them against evalplus/mbppplus's full
# `test` field (the extended harness with ~100 inputs/results per task).
#
# Usage:
#   bash scripts/llada21_mbpp/eval_mbpp_plus.sh <samples_jsonl> [extra args...]
# ------------------------------------------------------------------------

set -euo pipefail

# Default samples JSONL — override by passing a path as the first arg.
default_samples_jsonl="results/llada21_mbpp_plus_quality_full/len_512/inclusionAI__LLaDA2.1-mini/samples_mbpp_plus_instruct_llada_2026-04-25T04-16-42.954602.jsonl"

if [[ $# -ge 1 && "$1" != --* ]]; then
  samples_jsonl="$1"
  shift
else
  samples_jsonl="${default_samples_jsonl}"
fi

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate dllm

export PYTHONPATH=.:${PYTHONPATH:-}
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

python scripts/llada21_mbpp/eval_mbpp_plus.py \
    --samples_jsonl "${samples_jsonl}" \
    "$@"
