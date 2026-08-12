"""Aggregate the LLaDA2.1-mini HumanEval+ edit_step scaling sweep across random seeds.

R2D's correction remasks randomly (torch.rand), so each run is one draw. This script
combines the three seed replicates to put error bars on the accuracy-vs-NFE curve:
  - seed 1234 : original sweep dirs  confidence0.9_block_gibbs_edit_step{S}_early_exit2_postedit
  - seed 1    : ..._postedit_seed1
  - seed 2    : ..._postedit_seed2
edit_step=0 is deterministic (no random remask, greedy at temp=0) -> single seed, std=0.

Per edit_step it reads:
  - HumanEval+ pass@1 : combined/results_humaneval_plus.json  ('pass@1,create_test')
  - base HumanEval    : combined/results_combined.json        ('pass@1,create_test')
  - mean NFE          : chunk_*/generations.jsonl (chunk-local doc_id + chunk*chunk_size)
and aggregates mean / sample-std across the available seeds.

Outputs (under --results_root):
  - scaling_edit_step_seeds.csv  -- per edit_step: mean/std/min/max + per-seed values
  - scaling_edit_step_seeds.png  -- HumanEval+ pass@1 (mean +/- std over seeds) vs mean NFE

Usage:
    python scripts/llada21/scaling_curve_seeds.py \
        --results_root results/llada21_humaneval_len512 \
        --edit_steps 0 1 2 4 8 16 32 --threshold 0.9 --early_exit 2
"""

import csv
import json
import math
import statistics
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# (label, dir suffix) for each seed replicate.
SEEDS = [("1234", ""), ("1", "_seed1"), ("2", "_seed2")]


def config_dir(root, threshold, edit_step, early_exit, suffix):
    return root / (f"confidence{threshold}_block_gibbs_edit_step{edit_step}"
                   f"_early_exit{early_exit}_postedit{suffix}")


def read_pass1(results_json, task_key):
    if not results_json.exists():
        return None
    with open(results_json) as f:
        r = json.load(f)
    try:
        return float(r["results"][task_key]["pass@1,create_test"])
    except (KeyError, TypeError):
        return None


def mean_nfe(base, total_samples):
    """mean NFE per problem; generations.jsonl uses chunk-local doc_ids."""
    chunks = sorted(base.glob("chunk_*"))
    if not chunks:
        return float("nan")
    chunk_size = math.ceil(total_samples / len(chunks))
    vals = []
    for gen in base.glob("chunk_*/generations.jsonl"):
        with open(gen) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("nfe") is not None:
                    vals.append(int(r["nfe"]))
    return statistics.mean(vals) if vals else float("nan")


def main():
    ap = ArgumentParser(description="Aggregate edit_step HumanEval+ scaling across seeds")
    ap.add_argument("--results_root", default="results/llada21_humaneval_len512")
    ap.add_argument("--edit_steps", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--early_exit", type=int, default=2)
    ap.add_argument("--total_samples", type=int, default=164)
    args = ap.parse_args()

    root = Path(args.results_root)
    N = args.total_samples
    rows = []
    for es in args.edit_steps:
        plus, base, nfes, seeds_found = [], [], [], []
        for label, suffix in SEEDS:
            d = config_dir(root, args.threshold, es, args.early_exit, suffix)
            if not d.exists():
                continue
            p = read_pass1(d / "combined" / "results_humaneval_plus.json", "humaneval_plus")
            b = read_pass1(d / "combined" / "results_combined.json", "humaneval_instruct_llada")
            if p is None:
                continue
            plus.append(p)
            if b is not None:
                base.append(b)
            nfes.append(mean_nfe(d, N))
            seeds_found.append(label)

        if not plus:
            print(f"WARN edit_step={es}: no seeds found, skipping", file=sys.stderr)
            continue

        n = len(plus)
        p_mean = statistics.mean(plus)
        p_std = statistics.stdev(plus) if n > 1 else 0.0
        b_mean = statistics.mean(base) if base else float("nan")
        b_std = statistics.stdev(base) if len(base) > 1 else 0.0
        nfe_mean = statistics.mean([x for x in nfes if not math.isnan(x)]) if nfes else float("nan")

        rows.append({
            "edit_step": es,
            "n_seeds": n,
            "seeds": "|".join(seeds_found),
            "plus_mean": round(p_mean, 4),
            "plus_std": round(p_std, 4),
            "plus_solved_mean": round(p_mean * N, 1),
            "plus_solved_std": round(p_std * N, 1),
            "plus_min": round(min(plus), 4),
            "plus_max": round(max(plus), 4),
            "plus_per_seed": "|".join(f"{x:.4f}" for x in plus),
            "plus_solved_per_seed": "|".join(str(round(x * N)) for x in plus),
            "base_mean": round(b_mean, 4) if not math.isnan(b_mean) else "",
            "base_std": round(b_std, 4),
            "mean_nfe": round(nfe_mean, 1) if not math.isnan(nfe_mean) else "",
        })

    if not rows:
        print("ERROR: no configs found.", file=sys.stderr)
        sys.exit(1)

    csv_path = root / "scaling_edit_step_seeds.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Console table
    print(f"\n{'edit_step':>9} {'n':>2} {'plus mean':>9} {'+/-std':>7} "
          f"{'solved':>7} {'min-max':>13} {'mean_nfe':>9}")
    for r in rows:
        rng = f"{round(r['plus_min']*N)}-{round(r['plus_max']*N)}"
        print(f"{r['edit_step']:>9} {r['n_seeds']:>2} {r['plus_mean']:>9.4f} "
              f"{r['plus_std']:>7.4f} {r['plus_solved_mean']:>7.1f} {rng:>13} "
              f"{str(r['mean_nfe']):>9}")

    # Plot: HumanEval+ mean +/- std over seeds vs mean NFE
    valid = [r for r in rows if r["mean_nfe"] != ""]
    if valid:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        xs = [r["mean_nfe"] for r in valid]
        ys = [r["plus_mean"] for r in valid]
        es_std = [r["plus_std"] for r in valid]
        ax.errorbar(xs, ys, yerr=es_std, marker="o", capsize=4, lw=1.8,
                    color="#d62728", label="HumanEval+ (mean $\\pm$ std over seeds)")
        # scatter individual seed points for transparency
        for r in valid:
            per = [float(v) for v in r["plus_per_seed"].split("|")]
            ax.scatter([r["mean_nfe"]] * len(per), per, s=14, color="#d62728",
                       alpha=0.35, zorder=3)
            ax.annotate(f"e={r['edit_step']} (n={r['n_seeds']})",
                        (r["mean_nfe"], r["plus_mean"]),
                        textcoords="offset points", xytext=(6, 8), fontsize=8,
                        color="#d62728")
        ax.set_xscale("log")
        ax.set_xlabel("Mean NFE per problem (model forward passes)")
        ax.set_ylabel("HumanEval+ pass@1")
        ax.set_title("LLaDA2.1-mini (16B MoE) — random-remask correction scaling\n"
                     "HumanEval+ across seeds (edit_step sweep; threshold=0.9, early_exit=2)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(loc="lower right")
        fig.tight_layout()
        png_path = root / "scaling_edit_step_seeds.png"
        fig.savefig(png_path, dpi=150)
        print(f"\nWrote {csv_path}\nWrote {png_path}")
    else:
        print(f"\nWrote {csv_path}\n(No NFE data; skipped PNG.)", file=sys.stderr)


if __name__ == "__main__":
    main()
