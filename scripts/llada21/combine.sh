#!/usr/bin/env bash
set -euo pipefail

cd /u/zzhou24/projects/dllm

source ~/miniconda3/bin/activate
conda activate smdm2

python scripts/llada21/combine_humaneval_chunks.py \
    --base_dir results/llada21_humaneval_len512/confidence0.9_block_gibbs_edit_step20_early_exit2_postedit
