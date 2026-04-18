"""
Progressive Edit (PUMA-style) SFT for LLaDA-8B-Base on rStar-Coder.

Trains LLaDA-8B-Base using a PhasedMaskingEdit pool that progressively reveals
tokens using the model's own confidence scores.  The model learns to fill in
masked positions while iterating over all response positions.

Loss = Σ_b Σ_{l ∈ response} CE(logits[b,l], x0[b,l]) / L_eff[b] / B

K-schedule support: K can be increased per epoch, e.g.:
  [[12, 0], [18, 2], [24, 4], [30, 5]]   (format: [K, epoch])

Local (single GPU):
    accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/llada/sft_progressive_edit.py --output_dir .models/prog_edit

Multi-GPU (8 GPUs, FSDP):
    accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \\
        examples/llada/sft_progressive_edit.py --output_dir .models/prog_edit

Multi-node (2 nodes × 8 GPUs, Slurm):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \\
        --accelerate_config fsdp --script_path examples/llada/sft_progressive_edit.py \\
        --output_dir .models/prog_edit
"""

import json
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
from dllm.core.trainers.pool import PhasedMaskingEdit, mdm_edit_loss_fn_from_logits
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
    output_dir: str = field(default=".models/LLaDA-8B-Base/rstar_coder_progressive_edit")

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

    # Progressive edit-specific
    K: int = field(
        default=12,
        metadata={"help": "Number of masking phases (intervals). Doubled each K-schedule step."},
    )
    mode: str = field(
        default="standard",
        metadata={"help": "Pool mode: 'standard' or 'confidence_collapse'."},
    )
    confidence_threshold: float = field(
        default=0.9,
        metadata={"help": "Confidence threshold for confidence_collapse mode."},
    )
    k_schedule: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "JSON list of [K, epoch] pairs to update K at epoch boundaries. "
                "E.g. '[[12,0],[18,2],[24,4],[30,5]]'."
            )
        },
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
# Validation loss (standard MDM, same for all strategies)
# ---------------------------------------------------------------------------

def _randomly_mask_for_val(
    x0: torch.Tensor,
    prompt_mask: torch.Tensor,
    mask_id: int,
    min_t: float = 0.1,
) -> tuple:
    device = x0.device
    B, L = x0.shape
    L_eff = (~prompt_mask).sum(dim=1, keepdim=True).float()
    frac = min_t + torch.rand(B, 1, device=device) * (1.0 - min_t)
    num_mask_per_seq = torch.ceil(frac * L_eff).long().clamp(min=1)
    scores = torch.rand(B, L, device=device).masked_fill(prompt_mask, float("inf"))
    order = scores.argsort(dim=1).argsort(dim=1)
    mask_idx = (order < num_mask_per_seq) & (~prompt_mask)
    noised = torch.where(mask_idx, torch.tensor(mask_id, device=device), x0)
    num_mask_f = num_mask_per_seq.float().expand_as(mask_idx)
    return noised, mask_idx, num_mask_f


@torch.no_grad()
def compute_val_loss(model, val_loader, mask_id: int, accelerator) -> float:
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_count = torch.tensor(0, device=accelerator.device)

    for batch in val_loader:
        x0 = batch["labels"]
        prompt_mask = batch["prompt_mask"]
        B = x0.shape[0]

        noised, mask_idx, num_mask_f = _randomly_mask_for_val(x0, prompt_mask, mask_id)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(input_ids=noised).logits
        ce = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")
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

    # ---- K-schedule ----------------------------------------------------------
    k_schedule: List[Tuple[int, int]] = []
    if training_args.k_schedule is not None:
        k_schedule = json.loads(training_args.k_schedule)  # [[K, epoch], ...]
        k_schedule.sort(key=lambda x: x[1])  # ascending by epoch
    next_k_idx = 0

    # Apply initial K from k_schedule if epoch=0 entry exists
    initial_K = training_args.K
    if k_schedule and k_schedule[0][1] == 0:
        initial_K = k_schedule[0][0]
        next_k_idx = 1

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

    # ---- Optimizer -----------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        fused=torch.cuda.is_available(),
    )

    # ---- Accelerate prepare --------------------------------------------------
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # ---- Scheduler (after prepare so len(train_loader) is per-GPU) -----------
    # For progressive_edit each batch goes through K phases.  K may change per
    # epoch according to k_schedule, so compute total micro-steps accordingly.
    loader_len = len(train_loader)

    def _total_optimizer_steps() -> int:
        if training_args.max_steps > 0:
            return training_args.max_steps
        cur_k, ki = initial_K, 0
        total_micro = 0
        for e in range(training_args.num_train_epochs):
            while ki < len(k_schedule) and k_schedule[ki][1] <= e:
                cur_k = k_schedule[ki][0]
                ki += 1
            total_micro += loader_len * cur_k
        return math.ceil(total_micro / training_args.gradient_accumulation_steps)

    total_steps = _total_optimizer_steps()
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Pool ----------------------------------------------------------------
    # Pool is per-process; each rank maintains its own pool fed by its shard of
    # the DataLoader.  Pool state is NOT checkpointed (acceptable: re-initialises
    # from dataset on resume).
    pool = PhasedMaskingEdit(
        train_loader=train_loader,
        batch_size=training_args.per_device_train_batch_size,
        mask_id=MASK_ID,
        K=initial_K,
        device=accelerator.device,
        L=2048,
        mode=training_args.mode,
        confidence_threshold=training_args.confidence_threshold,
    )

    # ---- WandB init ----------------------------------------------------------
    if accelerator.is_main_process and training_args.report_to == "wandb":
        accelerator.init_trackers(
            project_name="llada-sft",
            config={
                "strategy": "progressive_edit",
                "model": model_args.model_name_or_path,
                "data_dir": data_args.data_dir,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "K": initial_K,
                "mode": training_args.mode,
                "k_schedule": training_args.k_schedule,
                "total_steps": total_steps,
            },
            init_kwargs={"wandb": {"name": "dllm-llada-progressive_edit"}},
        )

    # ---- Resume --------------------------------------------------------------
    global_step = 0
    start_epoch = 0
    start_micro = 0
    if training_args.resume_from_checkpoint is not None:
        accelerator.load_state(training_args.resume_from_checkpoint)
        ckpt_name = os.path.basename(training_args.resume_from_checkpoint)
        if ckpt_name.startswith("step_"):
            global_step = int(ckpt_name[5:])
        logger.info(f"Resumed from checkpoint: {training_args.resume_from_checkpoint} (step {global_step})")
        # Compute start_epoch and start_micro from global_step
        remaining = global_step * training_args.gradient_accumulation_steps
        cur_k, ki = initial_K, 0
        for e in range(training_args.num_train_epochs):
            while ki < len(k_schedule) and k_schedule[ki][1] <= e:
                cur_k = k_schedule[ki][0]
                ki += 1
            micro_this = loader_len * cur_k
            if remaining < micro_this:
                start_epoch = e
                start_micro = remaining
                break
            remaining -= micro_this
        # Fast-forward k_schedule to start_epoch
        while next_k_idx < len(k_schedule) and k_schedule[next_k_idx][1] <= start_epoch:
            pool.update_k(k_schedule[next_k_idx][0])
            next_k_idx += 1

    # ---- Training loop -------------------------------------------------------
    logger.info(
        f"Starting Progressive Edit training: total_steps={total_steps}, "
        f"per_device_bs={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, "
        f"K={initial_K}, world_size={accelerator.num_processes}"
    )

    model.train()
    num_epochs = training_args.num_train_epochs

    done = False
    for epoch in range(start_epoch, num_epochs):
        # ---- K-schedule (epoch-level) ----
        while next_k_idx < len(k_schedule) and k_schedule[next_k_idx][1] <= epoch:
            new_k = k_schedule[next_k_idx][0]
            pool.update_k(new_k)
            if accelerator.is_main_process:
                logger.info(f"epoch {epoch}: K → {new_k}")
            next_k_idx += 1

        pool.reset_loader_iter()
        micro_steps_this_epoch = loader_len * pool.K
        steps_this_epoch = math.ceil(micro_steps_this_epoch / training_args.gradient_accumulation_steps)
        _start = start_micro if epoch == start_epoch else 0
        _initial_step = math.ceil(_start / training_args.gradient_accumulation_steps)

        pbar = tqdm(
            total=steps_this_epoch,
            initial=_initial_step,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        step_in_epoch = _initial_step

        for micro in range(_start, micro_steps_this_epoch):
            with accelerator.accumulate(model):
                # The pool provides xt (partially masked) and x0 (ground truth)
                xt = pool.xt          # [B, L]
                x0 = pool.x0          # [B, L]
                prompt_mask = pool.state["prompt_mask"]  # [B, L]

                logits = model(input_ids=xt).logits  # [B, L, V]

                loss = mdm_edit_loss_fn_from_logits(
                    logits=logits,
                    x0=x0,
                    xt=xt,
                    mask_id=MASK_ID,
                    prompt_mask=prompt_mask,
                )
                accelerator.backward(loss)

                # ---- Pool advance (always, every micro-step) ----
                pool.update_from_logits(logits.detach())

                # ---- Optimizer step (only on last accumulation micro-step) ----
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    step_in_epoch += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        K=pool.K, lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    )

                    # ---- Logging ----
                    if global_step % training_args.logging_steps == 0:
                        logs = {
                            "train/loss": loss.item(),
                            "train/epoch": epoch,
                            "train/step_in_epoch": step_in_epoch,
                            "train/current_k": pool.K,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                        }
                        accelerator.log(logs, step=global_step)
                        if accelerator.is_main_process:
                            logger.info(
                                f"epoch {epoch + 1}/{num_epochs}  "
                                f"step {step_in_epoch}/{steps_this_epoch}  "
                                f"global_step {global_step}/{total_steps}  "
                                f"loss={loss.item():.4f}  K={pool.K}  "
                                f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                            )

                    # ---- Evaluation ----
                    if global_step % training_args.eval_steps == 0:
                        val_loss = compute_val_loss(model, val_loader, MASK_ID, accelerator)
                        accelerator.log({"val/loss": val_loss}, step=global_step)
                        if accelerator.is_main_process:
                            logger.info(f"step {global_step}  val_loss={val_loss:.4f}")

                    # ---- Checkpointing ----
                    if global_step % training_args.save_steps == 0:
                        ckpt_dir = os.path.join(training_args.output_dir, f"step_{global_step}")
                        accelerator.save_state(ckpt_dir)
                        if accelerator.is_main_process:
                            logger.info(f"Checkpoint saved: {ckpt_dir}")
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
                        done = True
                        break

            if done:
                break

        pbar.close()

        if done:
            break

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
