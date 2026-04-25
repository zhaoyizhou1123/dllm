"""Combine partial HumanEval evaluation results from chunked SLURM runs.

Usage:
    python scripts/llada21/combine_humaneval_chunks.py \
        --base_dir results/llada21_humaneval_len512/confidence0.5_block_gibbs_edit_step50_early_exit2_postedit
"""

import json
import glob
import math
import sys
from pathlib import Path
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Combine chunked HumanEval results")
    parser.add_argument("--base_dir", required=True, help="Parent dir containing chunk_*/ subdirs")
    parser.add_argument("--total_samples", type=int, default=164, help="Expected number of samples")
    args = parser.parse_args()

    base = Path(args.base_dir)
    chunk_dirs = sorted(base.glob("chunk_*"))
    if not chunk_dirs:
        print(f"ERROR: No chunk_*/ directories found in {base}", file=sys.stderr)
        sys.exit(1)

    all_samples = {}
    for chunk_dir in chunk_dirs:
        pattern = str(chunk_dir / "inclusionAI__LLaDA2.1-mini" / "samples_humaneval_instruct_llada_*.jsonl")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"ERROR: No samples JSONL found in {chunk_dir}", file=sys.stderr)
            sys.exit(1)
        # Take the newest file if multiple exist
        samples_file = files[-1]
        with open(samples_file) as f:
            for line in f:
                sample = json.loads(line)
                doc_id = sample["doc_id"]
                if doc_id in all_samples:
                    print(f"WARNING: Duplicate doc_id {doc_id} in {chunk_dir.name}, using latest", file=sys.stderr)
                all_samples[doc_id] = sample
        print(f"  {chunk_dir.name}: loaded {samples_file}")

    # Validate coverage
    expected = set(range(args.total_samples))
    actual = set(all_samples.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        print(f"ERROR: Missing doc_ids: {missing}", file=sys.stderr)
        sys.exit(1)
    if extra:
        print(f"WARNING: Extra doc_ids beyond expected range: {extra}", file=sys.stderr)

    # Compute pass@1
    scores = [all_samples[i]["pass@1"] for i in range(args.total_samples)]
    mean_pass = sum(scores) / len(scores)
    stderr = math.sqrt(mean_pass * (1 - mean_pass) / len(scores))

    # Write combined outputs
    out_dir = base / "combined"
    out_dir.mkdir(exist_ok=True)

    combined_samples_path = out_dir / "samples_humaneval_instruct_llada_combined.jsonl"
    with open(combined_samples_path, "w") as f:
        for i in range(args.total_samples):
            f.write(json.dumps(all_samples[i], ensure_ascii=False) + "\n")

    results = {
        "results": {
            "humaneval_instruct_llada": {
                "alias": "humaneval_instruct_llada",
                "pass@1,create_test": mean_pass,
                "pass@1_stderr,create_test": stderr,
            }
        },
        "n_samples": args.total_samples,
        "n_chunks": len(chunk_dirs),
    }
    results_path = out_dir / "results_combined.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCombined {args.total_samples} samples from {len(chunk_dirs)} chunks")
    print(f"pass@1 = {mean_pass:.4f} +/- {stderr:.4f}")
    print(f"Results: {results_path}")
    print(f"Samples: {combined_samples_path}")


if __name__ == "__main__":
    main()
