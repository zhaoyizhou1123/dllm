"""
ProSeCo SFT for LLaDA-8B-Base on rStar-Coder (arXiv:2602.11590).

Two forward passes per training step:
  1. Standard MDM loss: randomly mask response tokens, predict originals.
     Loss = Σ CE(logits[masked], x0[masked]) / num_mask / B
  2. Self-correction loss: greedily decode pass-1 logits, use as "corrupt"
     input, train model to correct back to original.
     Loss = Σ CE(corr_logits[response], x0[response]) / num_mask / B

Both losses contribute gradients in a single Accelerate accumulate() block,
so no manual no_sync() management is needed.

Local (single GPU):
    accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/llada/sft_proseco.py --output_dir .models/proseco

Multi-GPU (8 GPUs, FSDP):
    accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \\
        examples/llada/sft_proseco.py --output_dir .models/proseco

Multi-node (2 nodes × 8 GPUs, Slurm):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \\
        --accelerate_config fsdp --script_path examples/llada/sft_proseco.py \\
        --output_dir .models/proseco
"""

import json
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import dllm
from dllm.data.rstar_coder import split_rstar_coder

logger = dllm.utils.get_default_logger(__name__)

# ---------------------------------------------------------------------------
# Constants for LLaDA-8B-Base
# ---------------------------------------------------------------------------
MASK_ID = 126336   # <|mdm_mask|>
EOS_ID  = 126081   # <|endoftext|>


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    attn_implementation: str = None


@dataclass
class DataArguments:
    data_dir: str = field(
        default="/projects/bgqz/zzhou24/data/rstar_coder",
        metadata={"help": "Path to pre-tokenized rStar-Coder binary directory."},
    )
    val_ratio: float = 0.02


@dataclass
class TrainingArguments:
    output_dir: str = field(default=".models/LLaDA-8B-Base/rstar_coder_proseco")

    # Training duration (max_steps overrides num_train_epochs if > 0)
    num_train_epochs: int = 6
    max_steps: int = -1

    # Optimisation
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Batching
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    dataloader_num_workers: int = 4

    # Precision
    bf16: bool = True

    # ProSeCo-specific
    min_t: float = field(
        default=0.1,
        metadata={"help": "Minimum masking fraction. LLaDA SFT paper uses 0.1."},
    )
    proseco_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the self-correction loss."},
    )

    # Memory / speed
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing to save memory at the cost of ~30% slower training."},
    )

    # Logging / checkpointing
    seed: int = 42
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 5
    resume_from_checkpoint: Optional[str] = None
    report_to: str = "wandb"


# ---------------------------------------------------------------------------
# MDM masking helpers
# ---------------------------------------------------------------------------

def randomly_mask(
    x0: torch.Tensor,
    prompt_mask: torch.Tensor,
    mask_id: int,
    min_t: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly mask response tokens.

    Samples num_mask ~ Uniform(min_t * L_eff, L_eff] for each sequence,
    then masks exactly num_mask randomly-chosen response positions.

    Returns:
        noised:    [B, L]  masked input (prompt tokens unchanged)
        mask_idx:  [B, L]  bool — True at masked positions
        num_mask:  [B, L]  float — same value broadcast across all positions per seq
    """
    device = x0.device
    B, L = x0.shape
    L_eff = (~prompt_mask).sum(dim=1, keepdim=True).float()  # [B, 1]

    frac = min_t + torch.rand(B, 1, device=device) * (1.0 - min_t)
    num_mask_per_seq = torch.ceil(frac * L_eff).long().clamp(min=1)  # [B, 1]

    # Assign random scores to response positions; inf for prompt so they're never selected
    scores = torch.rand(B, L, device=device).masked_fill(prompt_mask, float("inf"))
    order = scores.argsort(dim=1).argsort(dim=1)  # rank-order
    mask_idx = (order < num_mask_per_seq) & (~prompt_mask)  # [B, L]

    noised = torch.where(mask_idx, torch.tensor(mask_id, device=device), x0)
    num_mask_f = num_mask_per_seq.float().expand_as(mask_idx)  # [B, L]
    return noised, mask_idx, num_mask_f


# ---------------------------------------------------------------------------
# Validation loss (standard MDM, same for all strategies)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_val_loss(model, val_loader, mask_id: int, min_t: float, accelerator) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_count = torch.tensor(0, device=accelerator.device)

    for batch in val_loader:
        x0 = batch["labels"]
        prompt_mask = batch["prompt_mask"]
        B = x0.shape[0]

        noised, mask_idx, num_mask_f = randomly_mask(x0, prompt_mask, mask_id, min_t)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(input_ids=noised).logits
        ce = F.cross_entropy(
            logits.transpose(1, 2), x0, reduction="none"
        )  # [B, L]
        loss = (ce * mask_idx.float() / num_mask_f.clamp_min(1)).sum() / B
        total_loss += loss * B
        total_count += B

    total_loss = accelerator.reduce(total_loss, reduction="sum")
    total_count = accelerator.reduce(total_count, reduction="sum")
    model.train()
    return (total_loss / total_count.clamp_min(1)).item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16" if training_args.bf16 else "no",
        log_with=training_args.report_to if training_args.report_to != "none" else None,
        project_dir=training_args.output_dir,
    )
    set_seed(training_args.seed)

    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)

    # ---- Model ---------------------------------------------------------------
    with accelerator.local_main_process_first():
        model = dllm.utils.get_model(model_args=model_args)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---- Data ----------------------------------------------------------------
    with accelerator.local_main_process_first():
        train_data, val_data = split_rstar_coder(
            data_args.data_dir,
            val_ratio=data_args.val_ratio,
            seed=training_args.seed,
        )

    train_loader = DataLoader(
        train_data,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=False,
        drop_last=False,
    )

    # ---- Optimizer & scheduler -----------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        fused=torch.cuda.is_available(),
    )

    steps_per_epoch = math.ceil(len(train_loader) / training_args.gradient_accumulation_steps)
    total_steps = (
        training_args.max_steps
        if training_args.max_steps > 0
        else training_args.num_train_epochs * steps_per_epoch
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Accelerate prepare --------------------------------------------------
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # ---- WandB init ----------------------------------------------------------
    if accelerator.is_main_process and training_args.report_to == "wandb":
        accelerator.init_trackers(
            project_name="llada-sft",
            config={
                "strategy": "proseco",
                "model": model_args.model_name_or_path,
                "data_dir": data_args.data_dir,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "min_t": training_args.min_t,
                "proseco_weight": training_args.proseco_weight,
                "total_steps": total_steps,
            },
            init_kwargs={"wandb": {"name": "dllm-llada-proseco"}},
        )

    # ---- Resume --------------------------------------------------------------
    global_step = 0
    start_step = 0
    if training_args.resume_from_checkpoint is not None:
        accelerator.load_state(training_args.resume_from_checkpoint)
        # Recover global_step from checkpoint dir name (step_NNNN)
        ckpt_name = os.path.basename(training_args.resume_from_checkpoint)
        if ckpt_name.startswith("step_"):
            start_step = int(ckpt_name[5:])
            global_step = start_step
        logger.info(f"Resumed from checkpoint: {training_args.resume_from_checkpoint} (step {start_step})")

    # ---- Training loop -------------------------------------------------------
    logger.info(
        f"Starting ProSeCo training: total_steps={total_steps}, "
        f"per_device_bs={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, "
        f"world_size={accelerator.num_processes}"
    )

    model.train()
    data_iter = iter(train_loader)

    pbar = tqdm(
        total=total_steps,
        initial=global_step,
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x0 = batch["labels"]                # [B, L]
        prompt_mask = batch["prompt_mask"]  # [B, L]
        B = x0.shape[0]

        with accelerator.accumulate(model):
            # ---- Pass 1: MDM loss ----
            noised, mask_idx, num_mask_f = randomly_mask(
                x0, prompt_mask, MASK_ID, training_args.min_t
            )
            logits = model(input_ids=noised).logits  # [B, L, V]

            ce_mdm = F.cross_entropy(
                logits.transpose(1, 2), x0, reduction="none"
            )  # [B, L]
            mdm_loss = (ce_mdm * mask_idx.float() / num_mask_f.clamp_min(1)).sum() / B
            accelerator.backward(mdm_loss)

            # ---- Pass 2: ProSeCo self-correction ----
            with torch.no_grad():
                pred = logits.detach().argmax(dim=-1)   # [B, L]
            # Keep prompt tokens; replace response with model's greedy prediction
            correction_input = torch.where(prompt_mask, x0, pred)

            corr_logits = model(input_ids=correction_input).logits  # [B, L, V]
            response_mask = ~prompt_mask  # [B, L]
            ce_corr = F.cross_entropy(
                corr_logits.transpose(1, 2), x0, reduction="none"
            )  # [B, L]
            correction_loss = (ce_corr * response_mask.float() / num_mask_f.clamp_min(1)).sum() / B
            accelerator.backward(training_args.proseco_weight * correction_loss)

            # ---- Optimizer step (only on last accumulation micro-step) ----
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.update(1)
                pbar.set_postfix(
                    mdm=f"{mdm_loss.item():.4f}",
                    corr=f"{correction_loss.item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                # ---- Logging ----
                if global_step % training_args.logging_steps == 0:
                    logs = {
                        "train/mdm_loss": mdm_loss.item(),
                        "train/correction_loss": correction_loss.item(),
                        "train/total_loss": (mdm_loss + training_args.proseco_weight * correction_loss).item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    }
                    accelerator.log(logs, step=global_step)
                    if accelerator.is_main_process:
                        logger.info(
                            f"step {global_step}/{total_steps}  "
                            f"mdm={mdm_loss.item():.4f}  corr={correction_loss.item():.4f}  "
                            f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                        )

                # ---- Evaluation ----
                if global_step % training_args.eval_steps == 0:
                    val_loss = compute_val_loss(model, val_loader, MASK_ID, training_args.min_t, accelerator)
                    accelerator.log({"val/loss": val_loss}, step=global_step)
                    if accelerator.is_main_process:
                        logger.info(f"step {global_step}  val_loss={val_loss:.4f}")

                # ---- Checkpointing ----
                if global_step % training_args.save_steps == 0:
                    ckpt_dir = os.path.join(training_args.output_dir, f"step_{global_step}")
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        logger.info(f"Checkpoint saved: {ckpt_dir}")
                        # Remove oldest checkpoints beyond save_total_limit
                        if training_args.save_total_limit > 0:
                            ckpts = sorted(
                                [d for d in os.listdir(training_args.output_dir)
                                 if d.startswith("step_") and os.path.isdir(
                                     os.path.join(training_args.output_dir, d))],
                                key=lambda x: int(x[5:]),
                            )
                            for old in ckpts[: max(0, len(ckpts) - training_args.save_total_limit)]:
                                shutil.rmtree(os.path.join(training_args.output_dir, old))
                                logger.info(f"Removed old checkpoint: {old}")

                if global_step >= total_steps:
                    break

    pbar.close()

    # ---- Final save ----------------------------------------------------------
    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(final_dir)
        logger.info(f"Final model saved to: {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    train()
