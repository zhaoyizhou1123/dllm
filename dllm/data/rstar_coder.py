"""
rStar-Coder dataset pre-tokenization and loading for LLaDA SFT.

Storage format (identical to mdm_correction/data/rstar_coder.py):
  labels.bin       uint32  [N, max_len]       — token IDs, right-padded with EOS
  prompt_mask.bin  uint8   [N, max_len/8]     — packed bits (bitorder="little")
                                                True = prompt + separator tokens
  meta.json

Sequence layout:
  ids = prompt_ids + sep_ids + answer_ids
  if len(ids) >= max_len:
      ids = ids[:max_len-1] + [EOS]
  else:
      ids = ids + [EOS] * (max_len - len(ids))

For ``microsoft/rStar-Coder`` ``synthetic_sft``:
  - prompt  → ``question`` field
  - answer  → ``code`` field  (reasoning-free; ``response`` is ignored)

Pre-tokenized binary already exists at /projects/bgqz/zzhou24/data/rstar_coder/.

Usage (load pre-tokenized data):
    from dllm.data.rstar_coder import split_rstar_coder
    train_data, val_data = split_rstar_coder("/projects/bgqz/zzhou24/data/rstar_coder")

Usage (re-tokenize from HuggingFace):
    python -m dllm.data.rstar_coder --out_dir /path/to/out [--limit 1000]
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Reasoning-strip helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REASON_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> and <reasoning>...</reasoning> blocks."""
    text = _THINK_RE.sub("", text)
    text = _REASON_RE.sub("", text)
    return text.strip()


def _get_prompt_and_answer(ex: dict) -> tuple[str, str]:
    """Extract prompt and answer from a dataset example with robust field detection."""
    prompt = (
        ex.get("input")
        or ex.get("problem")
        or ex.get("question")
        or ex.get("instruction")
        or ""
    )
    answer = (
        ex.get("output")
        or ex.get("solution")
        or ex.get("code")
        or ex.get("answer")
        or ""
    )
    answer = strip_reasoning(answer)
    return prompt.strip(), answer.strip()


# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------

def pretokenize_rstar_coder(
    out_dir: str,
    hf_dataset_name: str = "microsoft/rStar-Coder",
    hf_config: str = "synthetic_sft",
    tokenizer_name: str = "GSAI-ML/LLaDA-8B-Base",
    max_len: int = 2048,
    sep: str = "\n",
    batch_size: int = 1024,
    streaming: bool = True,
    limit: int | None = None,
) -> None:
    """
    Download rStar-Coder, strip reasoning traces, tokenise, and save as memmaps.

    This only needs to be run once; the result is already available at
    /projects/bgqz/zzhou24/data/rstar_coder/.
    """
    from datasets import load_dataset, load_dataset_builder
    from transformers import AutoTokenizer

    if max_len % 8 != 0:
        raise ValueError(f"max_len={max_len} must be divisible by 8 for packbits.")

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading tokenizer from {tokenizer_name} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=True)
    eos_id = tok.eos_token_id or 151643

    sep_ids = tok(sep, add_special_tokens=False).input_ids
    sep_len = len(sep_ids)

    if streaming:
        try:
            builder = load_dataset_builder(hf_dataset_name, hf_config)
            N_total = builder.info.splits["train"].num_examples
        except Exception:
            N_total = 2_000_000
        ds = load_dataset(hf_dataset_name, hf_config, split="train", streaming=True)
    else:
        ds = load_dataset(hf_dataset_name, hf_config, split="train", streaming=False)
        N_total = len(ds)

    if limit is not None:
        N_total = min(N_total, int(limit))

    mask_bytes = max_len // 8
    labels_path = os.path.join(out_dir, "labels.bin")
    mask_path = os.path.join(out_dir, "prompt_mask.bin")
    meta_path = os.path.join(out_dir, "meta.json")

    print(f"Allocating memmaps for up to {N_total:,} examples ...")
    labels_mm = np.memmap(labels_path, mode="w+", dtype=np.uint32, shape=(N_total, max_len))
    mask_mm = np.memmap(mask_path, mode="w+", dtype=np.uint8, shape=(N_total, mask_bytes))

    def batched(it, n):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) == n:
                yield buf
                buf = []
        if buf:
            yield buf

    def pack_mask(mask_bool_1d: np.ndarray) -> np.ndarray:
        return np.packbits(mask_bool_1d.astype(np.uint8), axis=-1, bitorder="little")

    written = skipped = 0
    pbar = tqdm(total=N_total, desc=f"Pretokenizing {hf_dataset_name} -> {out_dir}")

    for batch in batched(ds, batch_size):
        if written >= N_total:
            break
        remaining = N_total - written
        if len(batch) > remaining:
            batch = batch[:remaining]

        prompts, answers = [], []
        for ex in batch:
            p, a = _get_prompt_and_answer(ex)
            if not p and not a:
                skipped += 1
                continue
            prompts.append(p)
            answers.append(a)

        if not prompts:
            continue

        p_ids_batch = tok(prompts, add_special_tokens=False).input_ids
        a_ids_batch = tok(answers, add_special_tokens=False).input_ids

        for p_ids, a_ids in zip(p_ids_batch, a_ids_batch):
            if written >= N_total:
                break

            raw_ids = p_ids + sep_ids + a_ids
            prompt_len_raw = len(p_ids) + sep_len
            pm = np.zeros(max_len, dtype=np.bool_)

            if len(raw_ids) >= max_len:
                ids = raw_ids[: max_len - 1] + [eos_id]
                prompt_boundary = min(prompt_len_raw, max_len - 1)
            else:
                ids = raw_ids + [eos_id] * (max_len - len(raw_ids))
                prompt_boundary = min(prompt_len_raw, max_len)

            if prompt_boundary > 0:
                pm[:prompt_boundary] = True

            labels_mm[written, :] = np.asarray(ids, dtype=np.uint32)
            mask_mm[written, :] = pack_mask(pm)
            written += 1
            pbar.update(1)

    pbar.close()

    if written < N_total:
        print(f"Trimming memmaps from {N_total:,} to {written:,} rows ...")
        labels_mm.flush()
        mask_mm.flush()
        del labels_mm, mask_mm

        old_labels = np.memmap(labels_path, mode="r", dtype=np.uint32, shape=(N_total, max_len))
        old_mask = np.memmap(mask_path, mode="r", dtype=np.uint8, shape=(N_total, mask_bytes))
        new_labels = np.memmap(labels_path + ".tmp", mode="w+", dtype=np.uint32, shape=(written, max_len))
        new_mask = np.memmap(mask_path + ".tmp", mode="w+", dtype=np.uint8, shape=(written, mask_bytes))
        new_labels[:] = old_labels[:written]
        new_mask[:] = old_mask[:written]
        new_labels.flush()
        new_mask.flush()
        del old_labels, old_mask, new_labels, new_mask
        os.replace(labels_path + ".tmp", labels_path)
        os.replace(mask_path + ".tmp", mask_path)
    else:
        labels_mm.flush()
        mask_mm.flush()

    meta = {
        "hf_dataset": hf_dataset_name,
        "split": "train",
        "tokenizer": tokenizer_name,
        "max_len": max_len,
        "sep": sep,
        "eos_id": int(eos_id),
        "num_examples": int(written),
        "skipped": int(skipped),
        "labels_dtype": "uint32",
        "prompt_mask_packed": True,
        "prompt_mask_bitorder": "little",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Wrote {written:,} examples (skipped {skipped:,} empty).")


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class RStarCoderDataset(Dataset):
    """
    Memory-mapped loader for pre-tokenized rStar-Coder data.

    Returns:
        {
            "labels":      LongTensor[max_len],   # token IDs (prompt + code + EOS)
            "prompt_mask": BoolTensor[max_len],   # True = prompt / separator token
        }

    Compatible with the PhasedMaskingEdit pool which expects this exact format.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, "meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.max_len = int(self.meta["max_len"])
        self.N = int(self.meta["num_examples"])
        self.bitorder = self.meta.get("prompt_mask_bitorder", "little")

        self.labels_path = os.path.join(data_dir, "labels.bin")
        self.mask_path = os.path.join(data_dir, "prompt_mask.bin")

        self._labels_mm: np.memmap | None = None
        self._mask_mm: np.memmap | None = None

    def _open_memmaps(self):
        if self._labels_mm is None:
            self._labels_mm = np.memmap(
                self.labels_path, mode="r", dtype=np.uint32, shape=(self.N, self.max_len)
            )
            self._mask_mm = np.memmap(
                self.mask_path, mode="r", dtype=np.uint8, shape=(self.N, self.max_len // 8)
            )

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        self._open_memmaps()
        labels = torch.from_numpy(self._labels_mm[idx].astype(np.int64))
        packed = self._mask_mm[idx]
        mask = np.unpackbits(packed, bitorder=self.bitorder)[: self.max_len].astype(np.bool_)
        prompt_mask = torch.from_numpy(mask)
        return {"labels": labels, "prompt_mask": prompt_mask}


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_rstar_coder(
    data_dir: str,
    val_ratio: float = 0.02,
    seed: int = 2025,
):
    """
    Load RStarCoderDataset and split into train / val subsets.

    Args:
        data_dir:  path to the pre-tokenized binary directory
        val_ratio: fraction of data to hold out for validation (default 2 %)
        seed:      random seed for reproducibility

    Returns:
        (train_subset, val_subset)
    """
    dataset = RStarCoderDataset(data_dir)
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=g)


# ---------------------------------------------------------------------------
# CLI entry point for pre-tokenization
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize rStar-Coder dataset")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--hf_dataset", type=str, default="microsoft/rStar-Coder")
    parser.add_argument("--hf_config", type=str, default="synthetic_sft")
    parser.add_argument("--tokenizer", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--sep", type=str, default="\n")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--no_streaming", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pretokenize_rstar_coder(
        out_dir=args.out_dir,
        hf_dataset_name=args.hf_dataset,
        hf_config=args.hf_config,
        tokenizer_name=args.tokenizer,
        max_len=args.max_len,
        sep=args.sep,
        batch_size=args.batch_size,
        streaming=not args.no_streaming,
        limit=args.limit,
    )
