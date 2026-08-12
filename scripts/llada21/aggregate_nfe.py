#!/usr/bin/env python3
"""
Aggregate LLaDA2.1 HumanEval scaling runs into a single accuracy-vs-NFE table.

For each config subdirectory under a results root, this:
  * reads every `generations.jsonl` (including chunked subdirs) and computes the
    mean per-sample NFE (the `nfe` field written by NFE-instrumented samplers),
  * locates the lm-eval results JSON and extracts a pass@k metric,
  * parses the method + sweep point from the directory name.

Emits a CSV of (method, config, sweep_point, n_samples, mean_nfe, pass_at_1).

Usage:
    python dllm/scripts/llada21/aggregate_nfe.py \
        --root results/llada21_humaneval_len512 \
        --out results/llada21_humaneval_len512/scaling.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def mean_nfe(config_dir: Path) -> tuple[float | None, int]:
    """Mean NFE over all generations.jsonl entries under config_dir (recursive)."""
    total, n = 0, 0
    for gen in config_dir.rglob("generations.jsonl"):
        with open(gen, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("nfe") is not None:
                    total += int(rec["nfe"])
                    n += 1
    return (total / n if n else None), n


def mean_gen_time(config_dir: Path) -> float | None:
    """Mean per-sample generation wall time (seconds) over generations.jsonl."""
    total, n = 0.0, 0
    for gen in config_dir.rglob("generations.jsonl"):
        with open(gen, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("gen_time_s") is not None:
                    total += float(rec["gen_time_s"])
                    n += 1
    return (total / n if n else None)


def find_pass_at_1(config_dir: Path) -> float | None:
    """Best-effort extraction of a pass@k metric from an lm-eval results JSON."""
    candidates = list(config_dir.rglob("results*.json")) + list(config_dir.rglob("*results*.json"))
    for js in candidates:
        try:
            with open(js, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        results = data.get("results", data)
        if not isinstance(results, dict):
            continue
        for _task, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            for key, val in metrics.items():
                if "pass" in key.lower() and isinstance(val, (int, float)):
                    return float(val)
    return None


def parse_sweep(name: str) -> tuple[str, str]:
    """Infer (method, sweep_point) from a config dir name."""
    if name.startswith("remdm"):
        method = "ReMDM"
        m = re.search(r"step(\d+)", name)
    elif name.startswith("proseco"):
        method = "ProSeCo"
        m = re.search(r"corr(\d+)", name)
    elif name.startswith("confidence") or "gibbs" in name:
        method = "R2D"
        m = re.search(r"step(\d+)", name)
    elif name.startswith("block") or "mdm_block" in name:
        method = "Standard"
        m = re.search(r"step(\d+)", name)
    else:
        method = name.split("_")[0]
        m = re.search(r"(\d+)", name)
    return method, (m.group(1) if m else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Results root holding per-config subdirs")
    ap.add_argument("--out", default=None, help="CSV output path (default: stdout)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    rows = []
    for config_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        avg_nfe, n = mean_nfe(config_dir)
        if n == 0:
            continue  # not an NFE-instrumented run
        method, sweep = parse_sweep(config_dir.name)
        rows.append(
            {
                "method": method,
                "config": config_dir.name,
                "sweep_point": sweep,
                "n_samples": n,
                "mean_nfe": f"{avg_nfe:.2f}" if avg_nfe is not None else "",
                "mean_gen_time_s": (
                    f"{t:.2f}" if (t := mean_gen_time(config_dir)) is not None else ""
                ),
                "pass_at_1": (
                    f"{p:.4f}" if (p := find_pass_at_1(config_dir)) is not None else ""
                ),
            }
        )

    rows.sort(key=lambda r: (r["method"], float(r["mean_nfe"] or 0)))
    fieldnames = ["method", "config", "sweep_point", "n_samples", "mean_nfe", "mean_gen_time_s", "pass_at_1"]
    out = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if args.out:
        out.close()
        print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
