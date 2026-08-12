"""Build the LLaDA2.1-mini HumanEval / HumanEval+ accuracy-vs-NFE scaling curve.

For each edit_step config produced by scripts/llada21/conf_block_edit_sweep_local.sh, this
joins per-problem accuracy with per-problem NFE (model forward passes) by doc_id, for both:
  - base HumanEval  : combined/samples_humaneval_instruct_llada_combined.jsonl  (pass@1)
  - HumanEval+      : combined/samples_humaneval_plus.jsonl                      (pass@1)
NFE is metric-independent (same generations), read from chunk_*/generations.jsonl with the
chunk offset applied (generations use chunk-local doc_ids; samples use the global index).

Outputs (under --results_root):
  - scaling_edit_step.csv  -- one row per edit_step (base + plus pass@1, solved, mean/median NFE)
  - scaling_edit_step.png  -- pass@1 (y) vs mean NFE (x, log), HumanEval+ and base overlaid

Usage:
    python scripts/llada21/scaling_curve.py \
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


def config_dir(root: Path, threshold: float, edit_step: int, early_exit: int) -> Path:
    return root / f"confidence{threshold}_block_gibbs_edit_step{edit_step}_early_exit{early_exit}_postedit"


def load_pass1_from_file(path: Path) -> dict[int, float]:
    """doc_id -> pass@1 from a combined samples jsonl (global doc_ids)."""
    scores: dict[int, float] = {}
    if not path.exists():
        return scores
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            scores[int(s["doc_id"])] = float(s["pass@1"])
    return scores


def load_nfe(base: Path, chunk_size: int) -> dict[int, int]:
    """global doc_id -> nfe. generations.jsonl uses chunk-local doc_ids; the runner assigns
    chunk c the range [c*chunk_size, ...], so global_id = c*chunk_size + local_id."""
    nfe: dict[int, int] = {}
    for gen in sorted(base.glob("chunk_*/generations.jsonl")):
        chunk_idx = int(gen.parent.name.split("_")[1])
        offset = chunk_idx * chunk_size
        with open(gen) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("nfe") is not None:
                    nfe[offset + int(r["doc_id"])] = int(r["nfe"])
    return nfe


def agg(scores: dict[int, float]):
    if not scores:
        return float("nan"), 0, float("nan")
    n = len(scores)
    mean = sum(scores.values()) / n
    return mean, round(mean * n), math.sqrt(mean * (1 - mean) / n)


def main():
    ap = ArgumentParser(description="Build edit_step accuracy-vs-NFE scaling curve (base + plus)")
    ap.add_argument("--results_root", default="results/llada21_humaneval_len512")
    ap.add_argument("--edit_steps", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--early_exit", type=int, default=2)
    ap.add_argument("--total_samples", type=int, default=164)
    args = ap.parse_args()

    root = Path(args.results_root)
    rows = []
    for es in args.edit_steps:
        base = config_dir(root, args.threshold, es, args.early_exit)
        if not base.exists():
            print(f"WARN edit_step={es}: {base} missing, skipping", file=sys.stderr)
            continue
        combined = base / "combined"

        base_scores = load_pass1_from_file(combined / "samples_humaneval_instruct_llada_combined.jsonl")
        plus_scores = load_pass1_from_file(combined / "samples_humaneval_plus.jsonl")

        n_chunks = len(list(base.glob("chunk_*")))
        chunk_size = math.ceil(args.total_samples / n_chunks) if n_chunks else args.total_samples
        nfe = load_nfe(base, chunk_size)
        nfe_vals = [nfe[d] for d in base_scores if d in nfe]
        mean_nfe = statistics.mean(nfe_vals) if nfe_vals else float("nan")
        median_nfe = statistics.median(nfe_vals) if nfe_vals else float("nan")

        b_mean, b_solved, b_se = agg(base_scores)
        p_mean, p_solved, p_se = agg(plus_scores)
        if not plus_scores:
            print(f"WARN edit_step={es}: no HumanEval+ scores yet "
                  f"(samples_humaneval_plus.jsonl missing)", file=sys.stderr)

        rows.append({
            "edit_step": es,
            "base_pass@1": round(b_mean, 4),
            "base_solved": b_solved,
            "plus_pass@1": round(p_mean, 4) if not math.isnan(p_mean) else "",
            "plus_solved": p_solved if plus_scores else "",
            "mean_nfe": round(mean_nfe, 1),
            "median_nfe": median_nfe,
            "base_stderr": round(b_se, 4),
            "plus_stderr": round(p_se, 4) if not math.isnan(p_se) else "",
            "n_scored": len(base_scores),
        })

    if not rows:
        print("ERROR: no configs found; run the sweep first.", file=sys.stderr)
        sys.exit(1)

    # --- CSV ---
    csv_path = root / "scaling_edit_step.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- Console table ---
    print(f"\n{'edit_step':>9} {'base p@1':>9} {'base':>7} {'plus p@1':>9} {'plus':>7} {'mean_nfe':>9}")
    for r in rows:
        plus_p = f"{r['plus_pass@1']:.4f}" if r["plus_pass@1"] != "" else "-"
        plus_s = f"{r['plus_solved']}/164" if r["plus_solved"] != "" else "-"
        base_s = f"{r['base_solved']}/164"
        print(f"{r['edit_step']:>9} {r['base_pass@1']:>9.4f} {base_s:>7} "
              f"{plus_p:>9} {plus_s:>7} {r['mean_nfe']:>9.1f}")

    # --- Plot: HumanEval+ (primary) + base HumanEval (secondary) vs mean NFE ---
    valid = [r for r in rows if not math.isnan(r["mean_nfe"])]
    if valid:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        xs = [r["mean_nfe"] for r in valid]

        plus_rows = [r for r in valid if r["plus_pass@1"] != ""]
        if plus_rows:
            ax.errorbar([r["mean_nfe"] for r in plus_rows], [r["plus_pass@1"] for r in plus_rows],
                        yerr=[r["plus_stderr"] for r in plus_rows], marker="o", capsize=3, lw=1.8,
                        color="#d62728", label="HumanEval+ (primary)")
            for r in plus_rows:
                ax.annotate(f"e={r['edit_step']}", (r["mean_nfe"], r["plus_pass@1"]),
                            textcoords="offset points", xytext=(5, -12), fontsize=8, color="#d62728")

        ax.errorbar(xs, [r["base_pass@1"] for r in valid], yerr=[r["base_stderr"] for r in valid],
                    marker="s", capsize=3, lw=1.5, ls="--", color="#1f77b4", alpha=0.85,
                    label="base HumanEval")

        ax.set_xscale("log")
        ax.set_xlabel("Mean NFE per problem (model forward passes)")
        ax.set_ylabel("pass@1")
        ax.set_title("LLaDA2.1-mini (16B MoE) — random-remask correction scaling\n"
                     "(edit_step sweep; threshold=0.9, early_exit=2)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(loc="lower right")
        fig.tight_layout()
        png_path = root / "scaling_edit_step.png"
        fig.savefig(png_path, dpi=150)
        print(f"\nWrote {csv_path}\nWrote {png_path}")
    else:
        print(f"\nWrote {csv_path}\n(No NFE data; skipped PNG.)", file=sys.stderr)


if __name__ == "__main__":
    main()
