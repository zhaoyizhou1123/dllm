"""Inspect lm-eval samples_*.jsonl files: count non-EOS generated tokens per doc.

Useful for diffusion runs where the canvas is always exactly `max_new_tokens`
long — counting non-EOS tokens tells you how much real content the model
produced before degenerating into EOS padding (vs. filling the canvas).

Usage:
    python dllm/tools/inspect_eval_samples.py \\
        results/.../samples_humaneval_instruct_llada_*.jsonl \\
        [--base_model GSAI-ML/LLaDA-8B-Base] \\
        [--show all|fail|pass] \\
        [--quiet]   # only aggregate stats, no per-doc table

Prints a per-doc table of raw / filtered non-EOS token counts and pass@1, plus
aggregate mean/median/max across docs.
"""

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def count_non_eos(text: str, tok, eos_id: int) -> int:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return sum(1 for t in ids if t != eos_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples_path", help="Path to a samples_*.jsonl file produced by lm-eval --log_samples")
    parser.add_argument("--base_model", default="GSAI-ML/LLaDA-8B-Base", help="Tokenizer source")
    parser.add_argument("--show", choices=["all", "pass", "fail"], default="all", help="Filter rows in the per-doc table")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-doc table; show aggregates only")
    args = parser.parse_args()

    path = Path(args.samples_path)
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    eos = tok.eos_token_id

    raw_counts, filt_counts, passes = [], [], []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n_raw = count_non_eos(d["resps"][0][0], tok, eos)
        n_filt = count_non_eos(d["filtered_resps"][0][0], tok, eos)
        p = float(d.get("pass@1", float("nan")))
        raw_counts.append(n_raw)
        filt_counts.append(n_filt)
        passes.append(p)
        rows.append((d["doc_id"], n_raw, n_filt, p))

    if not args.quiet:
        print(f"{'doc':>4} {'raw_nonEOS':>10} {'filt_nonEOS':>11}  pass@1")
        for doc, n_raw, n_filt, p in rows:
            if args.show == "pass" and p < 1.0:
                continue
            if args.show == "fail" and p >= 1.0:
                continue
            print(f"{doc:>4} {n_raw:>10} {n_filt:>11}  {p}")

    n = len(rows)
    if n == 0:
        return
    print()
    print(f"=== Aggregates over {n} docs ===")
    print(f"raw_nonEOS  mean={statistics.mean(raw_counts):.1f}  median={statistics.median(raw_counts):.1f}  max={max(raw_counts)}")
    print(f"filt_nonEOS mean={statistics.mean(filt_counts):.1f}  median={statistics.median(filt_counts):.1f}  max={max(filt_counts)}")
    print(f"pass@1      mean={statistics.mean(passes):.3f}  ({sum(1 for p in passes if p >= 1.0)}/{n})")


if __name__ == "__main__":
    main()
