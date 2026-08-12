#!/usr/bin/env python3
"""Download the HumanEval+ (EvalPlus) test set and stage it as a local parquet.

RUN THIS ON A MACHINE WITH INTERNET (huggingface.co reachable) -- the compute/login
cluster has HF blocked. It writes ./test.parquet, which you then copy to the cluster at:

    /home/zhaoyiz/personal/dllm/local_data/humanevalplus/test.parquet

Requirements:
    pip install "datasets>=2.0" pyarrow

Usage:
    python download_humanevalplus.py                 # writes ./test.parquet
    python download_humanevalplus.py --out /some/dir/test.parquet
"""

import sys
from argparse import ArgumentParser

REQUIRED_COLUMNS = {"task_id", "entry_point", "test"}
EXPECTED_ROWS = 164
TARGET_ON_CLUSTER = "/home/zhaoyiz/personal/dllm/local_data/humanevalplus/test.parquet"


def main():
    ap = ArgumentParser(description="Stage evalplus/humanevalplus test split as parquet")
    ap.add_argument("--out", default="test.parquet", help="output parquet path (default: ./test.parquet)")
    ap.add_argument("--dataset", default="evalplus/humanevalplus", help="HF dataset id")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: `pip install datasets pyarrow` first.")

    print(f"Downloading {args.dataset} [{args.split}] from Hugging Face ...")
    ds = load_dataset(args.dataset, split=args.split)

    print(f"  rows:    {ds.num_rows}")
    print(f"  columns: {ds.column_names}")

    # --- Validate ---
    missing = REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        sys.exit(f"ERROR: dataset missing required columns: {sorted(missing)}")
    if ds.num_rows != EXPECTED_ROWS:
        print(f"  WARNING: expected {EXPECTED_ROWS} rows, got {ds.num_rows}", file=sys.stderr)

    row0 = ds[0]
    print("\n  sample row:")
    print(f"    task_id     = {row0['task_id']!r}")
    print(f"    entry_point = {row0['entry_point']!r}")
    tlen = len(row0["test"]) if isinstance(row0["test"], str) else "n/a"
    print(f"    test        = <{tlen} chars of test harness>")

    # --- Write parquet ---
    ds.to_parquet(args.out)
    print(f"\nWrote {ds.num_rows} rows -> {args.out}")
    print("\nNext: copy it to the cluster, e.g.")
    print(f"    scp {args.out} <cluster>:{TARGET_ON_CLUSTER}")
    print("(the target directory already exists on the cluster).")


if __name__ == "__main__":
    main()
