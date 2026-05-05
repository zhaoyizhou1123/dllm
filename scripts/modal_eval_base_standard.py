"""
Evaluate pretrained LLaDA-8B-Base (before SFT) on HumanEval using Modal (single H100).

Uses standard bidirectional block diffusion (MDLMSampler, no Gibbs correction).

One-time setup:
    pip install modal
    modal setup

Run:
    modal run scripts/modal_eval_base_standard.py
"""

import modal

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("dllm-llada-eval-base-standard")

# ---------------------------------------------------------------------------
# Image: CUDA devel base + all deps + lm-evaluation-harness
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

hf_cache_vol = modal.Volume.from_name("dllm-hf-cache", create_if_missing=True)
eval_vol = modal.Volume.from_name("dllm-eval-results", create_if_missing=True)

VOLUMES = {
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
    model_name: str = "GSAI-ML/LLaDA-8B-Base",
    max_new_tokens: int = 256,
    steps: int = 256,
    block_size: int = 256,
    temperature: float = 0.0,
    batch_size: int = 1,
    output_dir: str = "",
):
    import os
    import subprocess

    if not output_dir:
        output_dir = f"/eval_results/humaneval_maxlen{max_new_tokens}_block{block_size}/base_standard_steps{steps}"

    os.chdir("/root/dllm")
    os.environ["PYTHONPATH"] = "/root/dllm"
    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "True"

    os.makedirs(output_dir, exist_ok=True)

    model_args = (
        f"pretrained={model_name},"
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
    model_name: str = "GSAI-ML/LLaDA-8B-Base",
    max_new_tokens: int = 256,
    steps: int = 256,
    block_size: int = 32,
    temperature: float = 0.0,
    batch_size: int = 1,
    output_dir: str = "",
):
    if not output_dir:
        output_dir = f"/eval_results/humaneval_maxlen{max_new_tokens}_block{block_size}/base_standard_steps{steps}"
    evaluate.remote(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        steps=steps,
        block_size=block_size,
        temperature=temperature,
        batch_size=batch_size,
        output_dir=output_dir,
    )
