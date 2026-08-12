"""Estimate HumanEval+ pass@1 for a (possibly partial) run from generations.jsonl.

For still-running configs, lm-eval hasn't written samples_*.jsonl yet (that happens
only at the end), but our sampler writes generations.jsonl incrementally with one row
per completed problem. This script reconstructs the *exact* sanitized prediction used
by the `humaneval_instruct_llada` task (utils.build_predictions_llada_fastdllm) from
each generation, scores it against the HumanEval+ test suite, and reports pass@1 over
the N problems completed so far — an unbiased estimate of the final number.

On a completed config it reproduces the official results_humaneval_plus.json value,
which is used as the correctness check.

Usage:
    python scripts/llada21/estimate_humaneval_plus.py <config_dir> [<config_dir> ...]
"""

import json
import math
import os
import sys
from pathlib import Path

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import pandas as pd
from lm_eval.tasks.humaneval.sanitize_utils import sanitize

BASE_PARQUET = "local_data/openai_humaneval/test.parquet"
PLUS_PARQUET = "local_data/humanevalplus/test.parquet"


def sanitize_pred(generated: str, prompt: str, entry_point: str) -> str:
    # Exact mirror of utils.build_predictions_llada_fastdllm
    code = generated.split("```python\n", 1)[-1].split("```")[0]
    return sanitize(prompt + "\n" + code, entry_point)


def score_config(cfg_dir: Path, base_df, plus_lookup, code_eval, timeout, num_workers):
    gen_path = cfg_dir / "generations.jsonl"
    if not gen_path.exists():
        return None
    rows = [json.loads(l) for l in open(gen_path) if l.strip()]
    if not rows:
        return None

    predictions, references = [], []
    for r in rows:
        doc_id = r["doc_id"]
        base_row = base_df.iloc[doc_id]
        task_id = base_row["task_id"]
        if task_id not in plus_lookup:
            continue
        prompt = base_row["prompt"]
        entry_point = base_row["entry_point"]
        pred = sanitize_pred(r["generated"], prompt, entry_point)
        plus_row = plus_lookup[task_id]
        ref = plus_row["test"] + "\ncheck(" + plus_row["entry_point"] + ")"
        predictions.append([pred])
        references.append(ref)

    n = len(predictions)
    _, detailed = code_eval.compute(
        references=references, predictions=predictions,
        k=[1], num_workers=num_workers, timeout=timeout,
    )
    passed = 0
    for i in range(n):
        tr = detailed.get(i, [])
        if tr and any(x[1]["passed"] for x in tr):
            passed += 1
    mean = passed / n
    stderr = math.sqrt(mean * (1 - mean) / n) if n else 0.0
    return {"n": n, "passed": passed, "pass@1": mean, "stderr": stderr}


def main():
    dirs = [Path(d) for d in sys.argv[1:]]
    if not dirs:
        print("usage: estimate_humaneval_plus.py <config_dir> [...]", file=sys.stderr)
        sys.exit(1)

    base_df = pd.read_parquet(BASE_PARQUET).reset_index(drop=True)
    plus_df = pd.read_parquet(PLUS_PARQUET)
    plus_lookup = {row["task_id"]: row for _, row in plus_df.iterrows()}

    import evaluate as hf_evaluate
    code_eval = hf_evaluate.load("code_eval")

    print(f"{'config':<44} {'n':>4} {'pass':>5} {'HE+ pass@1':>12}")
    print("-" * 70)
    results = {}
    for d in dirs:
        res = score_config(d, base_df, plus_lookup, code_eval, timeout=10.0, num_workers=4)
        if res is None:
            print(f"{d.name:<44} {'--':>4}  (no generations)")
            continue
        results[d.name] = res
        tag = "" if res["n"] == 164 else f"  (partial {res['n']}/164)"
        print(f"{d.name:<44} {res['n']:>4} {res['passed']:>5} "
              f"{res['pass@1']:>10.4f}  +/-{res['stderr']:.3f}{tag}")
    return results


if __name__ == "__main__":
    main()
