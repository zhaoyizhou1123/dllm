#!/usr/bin/env bash
#SBATCH --job-name=merge_chunk9_reeval
#SBATCH --output=slurm/%x/job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --partition=ghx4-interactive
#SBATCH --account=bgqz-dtai-gh
#SBATCH --chdir=/u/zzhou24/projects/dllm

set -euo pipefail

model_name_or_path="inclusionAI/LLaDA2.1-mini"
max_new_tokens=512
block_size=32
threshold=0.9
min_transfer=1
temperature=0.0
batch_size=1
edit_freq=1
edit_step=50
edit_strategy="gibbs_edit"
remasking_strategy="random"
early_exit_number=2
num_workers=4

base_dir="results/llada21_humaneval_len512/confidence${threshold}_block_${edit_strategy}_step${edit_step}_early_exit${early_exit_number}_postedit"
output_dir="${base_dir}/chunk_9"

# ---- Step 1: Merge generations.jsonl ----
echo "===== Merging generations ====="

python3 << 'PYEOF'
import json
from pathlib import Path

base = Path("results/llada21_humaneval_len512/confidence0.9_block_gibbs_edit_step50_early_exit2_postedit")
chunk_dir = base / "chunk_9"
gen_path = chunk_dir / "generations.jsonl"

# Read existing chunk_9 generations (local doc_ids 0-11, missing 4,6,12,13)
existing = {}
with open(gen_path) as f:
    for line in f:
        entry = json.loads(line)
        existing[entry["doc_id"]] = entry
print(f"Existing chunk_9 doc_ids: {sorted(existing.keys())} ({len(existing)} entries)")

# Absolute doc → local doc_id within chunk_9 (offset=126)
doc_map = {130: 4, 132: 6, 138: 12, 139: 13}

for abs_id, local_id in doc_map.items():
    doc_gen = base / f"doc_{abs_id}" / "generations.jsonl"
    if not doc_gen.exists():
        print(f"ERROR: {doc_gen} not found — job for doc {abs_id} may not be finished")
        raise SystemExit(1)
    with open(doc_gen) as f:
        entry = json.loads(f.readline())
    entry["doc_id"] = local_id
    existing[local_id] = entry
    print(f"  doc_{abs_id} → local doc_id {local_id}")

# Write merged, sorted by doc_id
with open(gen_path, "w") as f:
    for doc_id in sorted(existing.keys()):
        f.write(json.dumps(existing[doc_id], ensure_ascii=False) + "\n")

print(f"Merged {len(existing)} generations into {gen_path}")
print(f"Doc IDs: {sorted(existing.keys())}")
assert len(existing) == 14, f"Expected 14 entries, got {len(existing)}"
PYEOF

# ---- Step 2: Re-run eval harness with reuse=True ----
echo "===== Re-evaluating with reuse=True ====="

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

# chunk_9: offset=126, limit=14 → docs [126..139]
task="humaneval_instruct_llada"
samples_json=$(python3 -c "import json; print(json.dumps({\"${task}\": list(range(126, 140))}))")

echo "output_dir=${output_dir}"
echo "samples=${samples_json}"

python dllm/pipelines/llada21/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 \
    --model llada21_confidence_block --apply_chat_template \
    --batch_size "${batch_size}" \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},block_size=${block_size},threshold=${threshold},min_transfer=${min_transfer},temperature=${temperature},eos_early_stop=True,edit_freq=${edit_freq},edit_step=${edit_step},edit_strategy=${edit_strategy},remasking_strategy=${remasking_strategy},early_exit_number=${early_exit_number},num_workers=${num_workers},output_dir=${output_dir},reuse=True" \
    --confirm_run_unsafe_code \
    --output_path "${output_dir}" --log_samples \
    --samples "${samples_json}"

echo "===== Done ====="
