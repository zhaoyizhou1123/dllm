"""
Evaluate a standard-MDM LLaDA-8B checkpoint on HumanEval using Modal (single H100).

Uses standard bidirectional block diffusion (MDLMSampler, no Gibbs correction).

One-time setup:
    pip install modal
    modal setup

Run:
    modal run scripts/modal_eval_standard.py

Custom checkpoint:
    modal run scripts/modal_eval_standard.py --checkpoint-path /checkpoints/standard_rstar/checkpoint-1500
"""

import modal

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("dllm-llada-eval-standard")

# ---------------------------------------------------------------------------
# Image: CUDA devel base + all deps + lm-evaluation-harness from submodule
# ---------------------------------------------------------------------------

dllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ninja-build")
    .pip_install(
        "torch==2.7.0+cu126",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "transformers==4.57.0",
        "accelerate==1.11.0",
        "deepspeed==0.18.0",
        "peft==0.17.1",
        "datasets==4.2.0",
        "sentencepiece==0.2.0",
        "torchmetrics",
        "tyro",
        "wandb",
        "omegaconf",
        "tqdm",
        "matplotlib",
        "rich",
        "packaging",
        "ninja",
        "wheel",
        "setuptools",
        "huggingface_hub",
    )
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .add_local_dir(
        "lm-evaluation-harness",
        remote_path="/root/lm-evaluation-harness",
        copy=True,
        ignore=[".git", "__pycache__", "*.pyc", "*.egg-info"],
    )
    .run_commands("cd /root/lm-evaluation-harness && pip install -e .")
    .add_local_dir(
        ".",
        remote_path="/root/dllm",
        copy=False,
        ignore=[
            ".git",
            ".pytest_cache",
            ".claude",
            "wandb",
            ".logs",
            ".models",
            "results",
            "checkpoints",
            "__pycache__",
            "*.egg-info",
            "*.pyc",
            "slurm",
            "lm-evaluation-harness",
        ],
    )
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------

data_vol = modal.Volume.from_name("dllm-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("dllm-checkpoints", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dllm-hf-cache", create_if_missing=True)
eval_vol = modal.Volume.from_name("dllm-eval-results", create_if_missing=True)

VOLUMES = {
    "/data": data_vol,
    "/checkpoints": ckpt_vol,
    "/hf-cache": hf_cache_vol,
    "/eval_results": eval_vol,
}

# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------


@app.function(
    image=dllm_image,
    gpu="H100:1",
    timeout=2 * 3600,
    volumes=VOLUMES,
)
def evaluate(
    checkpoint_path: str = "/checkpoints/standard_rstar/checkpoint-3000",
    max_new_tokens: int = 256,
    steps: int = 256,
    block_size: int = 256,
    temperature: float = 0.0,
    batch_size: int = 1,
    output_dir: str = "",
):
    import os
    import subprocess
    import sys

    if not output_dir:
        output_dir = f"/eval_results/humaneval_maxlen{max_new_tokens}_block{block_size}_notemplate/standard_steps{steps}"

    os.chdir("/root/dllm")
    sys.path.insert(0, "/root/dllm")
    os.environ["PYTHONPATH"] = "/root/dllm"
    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "True"

    # Auto-convert non-HF checkpoints to HF format
    if os.path.isdir(checkpoint_path) and not os.path.isfile(
        os.path.join(checkpoint_path, "config.json")
    ):
        from dllm.tools.convert_fsdp_checkpoint import convert_fsdp_to_hf

        print(f"No config.json found in {checkpoint_path}, converting to HF format...")
        checkpoint_path = convert_fsdp_to_hf(checkpoint_path)
        print(f"Using converted checkpoint: {checkpoint_path}")
        ckpt_vol.commit()

    os.makedirs(output_dir, exist_ok=True)

    model_args = (
        f"pretrained={checkpoint_path},"
        f"max_new_tokens={max_new_tokens},"
        f"steps={steps},"
        f"block_size={block_size},"
        f"temperature={temperature},"
        f"eos_early_stop=True,"
        f"cfg_scale=0.0,"
        f"remasking=low_confidence,"
        f"begin_suppress_tokens=[]"
    )

    cmd = [
        "python",
        "dllm/pipelines/llada/eval.py",
        "--tasks",
        "humaneval",
        "--num_fewshot",
        "0",
        "--model",
        "llada",
        "--batch_size",
        str(batch_size),
        "--model_args",
        model_args,
        "--confirm_run_unsafe_code",
        "--output_path",
        output_dir,
        "--log_samples",
    ]

    print(f"Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/root/dllm", env=os.environ)

    # Persist results to volume
    eval_vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    checkpoint_path: str = "/checkpoints/mask_proseco_rstar/step_600",
    max_new_tokens: int = 256,
    steps: int = 256,
    block_size: int = 32,
    temperature: float = 0.0,
    batch_size: int = 1,
    output_dir: str = "",
):
    if not output_dir:
        output_dir = f"/eval_results/humaneval_maxlen{max_new_tokens}_block{block_size}/standard_steps{steps}"
    evaluate.remote(
        checkpoint_path=checkpoint_path,
        max_new_tokens=max_new_tokens,
        steps=steps,
        block_size=block_size,
        temperature=temperature,
        batch_size=batch_size,
        output_dir=output_dir,
    )
