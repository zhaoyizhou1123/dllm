"""Matched carry-over vs no-carry comparison for the ReMDM edit_step=8 point.

carry:    results/_carryover_test/carry/generations.jsonl   (--samples subset run,
          doc_id renumbered 0..N-1; mapped back to original via SAMPLES list)
no-carry: results/llada21_humaneval_len512/remdm_cap_eta0.4_step8_earlyexit5/
          generations.jsonl  (full run, doc_id == original HumanEval index)

Both use identical settings (cap, eta=0.4, edit_step=8, temp=0, early_exit=5);
the only difference is SUBS carry-over. Scores HumanEval+ pass@1 on the set of
problems completed in BOTH runs, and prints a per-problem table.
"""
import json, os
from pathlib import Path
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import pandas as pd
from lm_eval.tasks.humaneval.sanitize_utils import sanitize

SAMPLES = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132]
CARRY = "results/_carryover_test/carry/generations.jsonl"
NOCARRY = "results/llada21_humaneval_len512/remdm_cap_eta0.4_step8_earlyexit5/generations.jsonl"
BASE = "local_data/openai_humaneval/test.parquet"
PLUS = "local_data/humanevalplus/test.parquet"


def sanitize_pred(generated, prompt, entry_point):
    code = generated.split("```python\n", 1)[-1].split("```")[0]
    return sanitize(prompt + "\n" + code, entry_point)


def load_carry():
    d = {}
    for l in open(CARRY):
        if l.strip():
            r = json.loads(l)
            d[SAMPLES[r["doc_id"]]] = r["generated"]
    return d


def load_nocarry(orig_ids):
    d = {}
    for l in open(NOCARRY):
        if l.strip():
            r = json.loads(l)
            if r["doc_id"] in orig_ids:
                d[r["doc_id"]] = r["generated"]
    return d


def main():
    base = pd.read_parquet(BASE).reset_index(drop=True)
    plus = {row["task_id"]: row for _, row in pd.read_parquet(PLUS).iterrows()}
    import evaluate as hf_evaluate
    code_eval = hf_evaluate.load("code_eval")

    carry = load_carry()
    nocarry = load_nocarry(set(carry))
    common = sorted(set(carry) & set(nocarry))
    print(f"carry has {len(carry)}, matched in both: {len(common)} -> {common}\n")
    if not common:
        return

    def score(gen_map):
        preds, refs = [], []
        for oid in common:
            row = base.iloc[oid]
            tid = row["task_id"]
            preds.append([sanitize_pred(gen_map[oid], row["prompt"], row["entry_point"])])
            refs.append(plus[tid]["test"] + "\ncheck(" + plus[tid]["entry_point"] + ")")
        _, det = code_eval.compute(references=refs, predictions=preds, k=[1],
                                   num_workers=4, timeout=10.0)
        return [1 if det.get(i) and any(x[1]["passed"] for x in det[i]) else 0
                for i in range(len(common))]

    cp = score(carry)
    np_ = score(nocarry)
    print(f"{'orig_id':>7} {'task':>12} {'carry':>6} {'nocarry':>8}")
    for i, oid in enumerate(common):
        print(f"{oid:>7} {base.iloc[oid]['task_id']:>12} {cp[i]:>6} {np_[i]:>8}")
    n = len(common)
    print(f"\n{'='*40}")
    print(f"matched n = {n}")
    print(f"carry   HE+ pass@1 = {sum(cp)/n:.4f}  ({sum(cp)}/{n})")
    print(f"nocarry HE+ pass@1 = {sum(np_)/n:.4f}  ({sum(np_)}/{n})")


if __name__ == "__main__":
    main()
