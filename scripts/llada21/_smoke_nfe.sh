#!/usr/bin/env bash
# GPU smoke test: tiny 2-problem run of the new ReMDM + ProSeCo samplers and R2D,
# verifying generations.jsonl carries per-doc "nfe" and pass@1 is computed.
set -eo pipefail
export TMPDIR=/home/zhaoyiz/tmp
source /opt/conda/etc/profile.d/conda.sh
conda activate mdm
set -u  # enable nounset only after conda activation (cuda-nvcc activate.d isn't -u clean)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/zhaoyiz/personal/dllm
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=True
cd /home/zhaoyiz/personal/dllm

MODEL="inclusionAI/LLaDA2.1-mini"
task="humaneval_instruct_llada"
samples_json='{"'"$task"'": [41, 110]}'

run () {
  local name="$1"; local model="$2"; local extra="$3"
  local out="results/_smoke/${name}"
  rm -rf "$out"; mkdir -p "$out"
  echo "########## SMOKE: ${name} (${model}) ##########"
  python dllm/pipelines/llada21/eval.py \
    --tasks "$task" --num_fewshot 0 \
    --model "$model" --apply_chat_template \
    --batch_size 1 \
    --model_args "pretrained=${MODEL},max_new_tokens=128,block_size=32,temperature=0.0,eos_early_stop=True,${extra},output_dir=${out}" \
    --confirm_run_unsafe_code \
    --output_path "$out" --log_samples \
    --samples "$samples_json"
  echo "----- ${name} generations.jsonl (nfe field?) -----"
  python - "$out/generations.jsonl" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
for r in rows:
    print(f"  doc_id={r['doc_id']} nfe={r.get('nfe')} gen_len={len(r['generated'])}")
assert all(r.get("nfe") is not None for r in rows), "MISSING nfe!"
print(f"  OK: {len(rows)} rows, all have nfe")
PY
}

run "remdm"      "llada21_remdm"            "variant=cap,eta=0.4,edit_step=2,early_exit_number=5"
run "proseco"    "llada21_proseco"          "unmasking_num=1,correction_step=2"
run "r2d"        "llada21_confidence_block" "threshold=0.9,min_transfer=1,edit_freq=1,edit_step=2,edit_strategy=gibbs_edit,remasking_strategy=random,early_exit_number=2"

echo "########## ALL SMOKE RUNS DONE ##########"
