"""Convert an `accelerator.save_state()` FSDP checkpoint to HF `save_pretrained` format.

Examples/llada/sft_proseco.py saves checkpoints via `accelerator.save_state(ckpt_dir)`,
which (with `fsdp_state_dict_type: FULL_STATE_DICT`, see scripts/accelerate_configs/fsdp2.yaml)
produces a directory containing:
  - pytorch_model_fsdp.bin   (full unsharded state dict, rank 0)
  - optimizer.bin
  - random_states_*.pkl

This format is not loadable by `from_pretrained`. This script rebuilds an HF-loadable
directory by instantiating the base model skeleton, loading the state dict into it,
and calling `save_pretrained` (plus saving the tokenizer).

Usage:
    python dllm/tools/convert_fsdp_checkpoint.py \\
        --checkpoint_dir checkpoints/progressive_edit_rstar/step_500 \\
        [--base_model GSAI-ML/LLaDA-8B-Base] \\
        [--output_dir checkpoints/progressive_edit_rstar/step_500_hf]

Conversion is idempotent: if the output directory already contains a `config.json`,
the script exits without redoing work.
"""

import argparse
import os
import sys

import torch

from dllm.utils import get_model, get_tokenizer
from dllm.utils.configs import ModelArguments


def is_fsdp_checkpoint(path: str) -> bool:
    """A directory is an FSDP-state checkpoint iff it has the FSDP weights file
    and not a HuggingFace `config.json`."""
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "pytorch_model_fsdp.bin"))
        and not os.path.isfile(os.path.join(path, "config.json"))
    )


def convert_fsdp_to_hf(
    checkpoint_dir: str,
    base_model: str = "GSAI-ML/LLaDA-8B-Base",
    output_dir: str | None = None,
) -> str:
    """Convert a single FSDP checkpoint dir to HF format. Returns the output path."""
    checkpoint_dir = os.path.abspath(checkpoint_dir.rstrip("/"))
    if output_dir is None:
        output_dir = checkpoint_dir + "_hf"
    output_dir = os.path.abspath(output_dir)

    if os.path.isfile(os.path.join(output_dir, "config.json")):
        print(f"[convert_fsdp_checkpoint] Already converted: {output_dir}", flush=True)
        return output_dir

    sd_path = os.path.join(checkpoint_dir, "pytorch_model_fsdp.bin")
    if not os.path.isfile(sd_path):
        raise FileNotFoundError(
            f"No pytorch_model_fsdp.bin in {checkpoint_dir} — not an FSDP checkpoint?"
        )

    m_args = ModelArguments(model_name_or_path=base_model)
    print(f"[convert_fsdp_checkpoint] Loading base skeleton: {base_model}", flush=True)
    model = get_model(model_args=m_args)

    print(f"[convert_fsdp_checkpoint] Loading state dict: {sd_path}", flush=True)
    sd = torch.load(sd_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[convert_fsdp_checkpoint] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
    if unexpected:
        print(f"[convert_fsdp_checkpoint] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[convert_fsdp_checkpoint] Saving HF checkpoint to {output_dir}", flush=True)
    model.save_pretrained(output_dir)
    get_tokenizer(model_args=m_args).save_pretrained(output_dir)
    print(f"[convert_fsdp_checkpoint] Done: {output_dir}", flush=True)
    return output_dir


def resolve_checkpoint(path: str, base_model: str = "GSAI-ML/LLaDA-8B-Base") -> str:
    """Return a path that `from_pretrained` can load.

    If `path` is an FSDP checkpoint dir, convert it to `<path>_hf` (idempotent) and
    return that. Otherwise return `path` unchanged (HF repo id, or already-HF dir).
    """
    if is_fsdp_checkpoint(path):
        return convert_fsdp_to_hf(path, base_model=base_model)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_dir", required=True, help="Path to FSDP checkpoint dir (contains pytorch_model_fsdp.bin)")
    parser.add_argument("--base_model", default="GSAI-ML/LLaDA-8B-Base", help="Base model whose architecture/tokenizer to use")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: <checkpoint_dir>_hf)")
    parser.add_argument("--print_only", action="store_true", help="Print resolved output path without converting")
    args = parser.parse_args()

    if args.print_only:
        out = (args.output_dir or args.checkpoint_dir.rstrip("/") + "_hf")
        print(out)
        return

    out = convert_fsdp_to_hf(args.checkpoint_dir, base_model=args.base_model, output_dir=args.output_dir)
    print(out)


if __name__ == "__main__":
    main()
