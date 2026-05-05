"""
Run standard MDM SFT training for LLaDA-8B on Modal with H100 GPUs.

One-time setup:
    pip install modal
    modal setup
    modal secret create wandb-secret WANDB_API_KEY=<your-key>
    modal volume put dllm-data /local/path/to/rstar_coder/ rstar_coder/

Train:
    modal run scripts/modal_train_standard.py

Resume from checkpoint:
    modal run scripts/modal_train_standard.py --resume /checkpoints/standard_rstar/checkpoint-500

Disable flash attention:
    modal run scripts/modal_train_standard.py --no-flash-attention
"""

import modal

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("dllm-llada-standard")

# ---------------------------------------------------------------------------
# Image: CUDA devel base (needed for flash-attn compilation) + all deps
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
        "lm_eval",
    )
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .add_local_dir(
        ".",
        remote_path="/root/dllm",
        copy=True,
        ignore=[
            ".git",
            ".pytest_cache",
            ".claude",
            "wandb",
            ".logs",
            ".models",
            "results",
            "checkpoints",
            "lm-evaluation-harness",
            "__pycache__",
            "*.egg-info",
            "*.pyc",
            "slurm",
        ],
    )
    .run_commands("cd /root/dllm && pip install -e .")
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------

data_vol = modal.Volume.from_name("dllm-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("dllm-checkpoints", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("dllm-hf-cache", create_if_missing=True)

VOLUMES = {
    "/data": data_vol,
    "/checkpoints": ckpt_vol,
    "/hf-cache": hf_cache_vol,
}

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

NGPUS = 8
# Global batch = NGPUS x bs 4 x grad_accum
# 4 GPUs: 4 x 4 x 8 = 128 | 8 GPUs: 8 x 4 x 4 = 128
GRAD_ACCUM = 128 // (NGPUS * 4)


@app.function(
    image=dllm_image,
    gpu=f"H100:{NGPUS}",
    timeout=24 * 3600,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    data_dir: str = "/data/rstar_coder/rstar_coder"",
    output_dir: str = "/checkpoints/standard_rstar",
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base",
    resume_from_checkpoint: str | None = None,
    flash_attention: bool = True,
):
    import json
    import os
    import subprocess

    os.chdir("/root/dllm")
    os.environ["PYTHONPATH"] = "/root/dllm"
    os.environ["HF_HOME"] = "/hf-cache"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["WANDB_PROJECT"] = "llada-sft"

    # -- Patch model config for flash attention if requested -------------------
    if flash_attention:
        from huggingface_hub import snapshot_download

        local_model_path = snapshot_download(
            model_name_or_path,
            cache_dir="/hf-cache",
        )
        config_path = os.path.join(local_model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        if not config.get("flash_attention", False):
            config["flash_attention"] = True
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Patched flash_attention=true in {config_path}")
        model_name_or_path = local_model_path

    # -- Build accelerate launch command ---------------------------------------
    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        "scripts/accelerate_configs/modal_fsdp.yaml",
        "--num_processes",
        str(NGPUS),
        "--num_machines",
        "1",
        "--machine_rank",
        "0",
        "--main_process_port",
        "29500",
        "examples/llada/sft.py",
        "--data_dir",
        data_dir,
        "--group_by_length",
        "false",
        "--remove_unused_columns",
        "false",
        "--model_name_or_path",
        model_name_or_path,
        "--output_dir",
        output_dir,
        "--per_device_train_batch_size",
        "4",
        "--gradient_accumulation_steps",
        str(GRAD_ACCUM),
        "--gradient_checkpointing",
        "true",
        "--num_train_epochs",
        "6",
        "--learning_rate",
        "2e-5",
        "--warmup_steps",
        "1000",
        "--logging_steps",
        "50",
        "--eval_steps",
        "500",
        "--save_steps",
        "500",
        "--save_total_limit",
        "1",
        "--run_name",
        "dllm-llada-standard",
        "--report_to",
        "wandb",
    ]

    if resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint])

    print(f"Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/root/dllm", env=os.environ)

    # Persist checkpoints to volume
    ckpt_vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    data_dir: str = "/data/rstar_coder/rstar_coder",
    output_dir: str = "/checkpoints/standard_rstar",
    resume: str | None = None,
    no_flash_attention: bool = False,
):
    train.remote(
        data_dir=data_dir,
        output_dir=output_dir,
        resume_from_checkpoint=resume,
        flash_attention=not no_flash_attention,
    )
