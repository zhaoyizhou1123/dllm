"""Re-evaluate HumanEval generations against HumanEval+ test cases.

Reuses already-sanitized predictions from lm-evaluation-harness JSONL output
and scores them against the extended HumanEval+ test suite (80x more tests).

Usage:
    python scripts/llada21/eval_humaneval_plus.py \
        --samples_jsonl results/.../samples_humaneval_instruct_llada_*.jsonl
"""

import json
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(
        description="Re-evaluate HumanEval generations against HumanEval+ test cases"
    )
    parser.add_argument(
        "--samples_jsonl",
        required=True,
        help="Path to the JSONL samples file from lm-evaluation-harness",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout per test case in seconds (default: 10.0, higher than code_eval "
        "default of 3.0 because HumanEval+ tests are much larger)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for code execution (default: 4)",
    )
    args = parser.parse_args()

    # Enable code execution
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Load JSONL samples
    samples_path = Path(args.samples_jsonl)
    samples = []
    with open(samples_path) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {samples_path}")
    if len(samples) < 164:
        print(
            f"WARNING: Only {len(samples)} of 164 samples present", file=sys.stderr
        )

    # Load HumanEval+ dataset
    from datasets import load_dataset

    ds_plus = load_dataset("evalplus/humanevalplus", split="test")
    plus_lookup = {row["task_id"]: row for row in ds_plus}
    print(f"Loaded HumanEval+ dataset with {len(plus_lookup)} problems")

    # Build aligned predictions and references
    predictions = []
    references = []
    task_ids = []
    original_scores = []
    for sample in samples:
        task_id = sample["doc"]["task_id"]
        if task_id not in plus_lookup:
            print(
                f"WARNING: {task_id} not found in HumanEval+, skipping",
                file=sys.stderr,
            )
            continue
        plus_row = plus_lookup[task_id]
        preds = sample["filtered_resps"][0]  # list of candidate code strings
        ref = plus_row["test"] + "\ncheck(" + plus_row["entry_point"] + ")"
        predictions.append(preds)
        references.append(ref)
        task_ids.append(task_id)
        original_scores.append(sample.get("pass@1", None))

    # Run evaluation
    import evaluate as hf_evaluate

    code_eval = hf_evaluate.load("code_eval")
    print(
        f"Evaluating {len(predictions)} samples with timeout={args.timeout}s, "
        f"num_workers={args.num_workers} ..."
    )
    pass_at_k_results, detailed_results = code_eval.compute(
        references=references,
        predictions=predictions,
        k=[1],
        num_workers=args.num_workers,
        timeout=args.timeout,
    )

    # Compute per-sample scores
    per_sample_pass = []
    for i in range(len(predictions)):
        task_results = detailed_results.get(i, [])
        if task_results:
            task_results.sort()
            passed = any(r[1]["passed"] for r in task_results)
        else:
            passed = False
        per_sample_pass.append(1.0 if passed else 0.0)

    mean_pass = sum(per_sample_pass) / len(per_sample_pass)
    stderr = math.sqrt(mean_pass * (1 - mean_pass) / len(per_sample_pass))

    # Compute original pass@1 for comparison
    valid_original = [s for s in original_scores if s is not None]
    orig_mean = sum(valid_original) / len(valid_original) if valid_original else None

    # Dump updated JSONL with HumanEval+ pass@1 scores
    task_id_to_pass = dict(zip(task_ids, per_sample_pass))
    samples_out_path = samples_path.parent / "samples_humaneval_plus.jsonl"
    with open(samples_out_path, "w") as f:
        for sample in samples:
            task_id = sample["doc"]["task_id"]
            if task_id in task_id_to_pass:
                sample = {**sample, "pass@1": task_id_to_pass[task_id]}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved updated samples to: {samples_out_path}")

    # Save results
    out_path = samples_path.parent / "results_humaneval_plus.json"
    results = {
        "results": {
            "humaneval_plus": {
                "alias": "humaneval_plus",
                "pass@1,create_test": mean_pass,
                "pass@1_stderr,create_test": stderr,
            }
        },
        "n_samples": len(per_sample_pass),
        "source_jsonl": str(samples_path),
        "original_pass@1": orig_mean,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("HumanEval+ Re-evaluation Results")
    print(f"{'=' * 60}")
    print(f"Samples evaluated: {len(per_sample_pass)}")
    if orig_mean is not None:
        print(f"Original HumanEval pass@1: {orig_mean:.4f}")
    print(f"HumanEval+ pass@1:         {mean_pass:.4f} +/- {stderr:.4f}")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
