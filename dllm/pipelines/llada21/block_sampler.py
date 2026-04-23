"""
Standard block diffusion sampler for LLaDA2.1-MoE.

Uses LLaDA2.1's correct infrastructure (block-causal attention, mask-filled canvas,
position_ids) with a fixed, deterministic unmasking schedule (pure topk by confidence,
no adaptive threshold gating, no iterative editing).

Supports batch_size > 1 (equal-length prompts required) because the schedule is
deterministic and does not depend on per-sequence confidence.

Run:
    python dllm/pipelines/llada21/eval.py \
        --tasks humaneval_instruct_llada --num_fewshot 0 \
        --model llada21_block --apply_chat_template \
        --batch_size 4 \
        --model_args "pretrained=inclusionAI/LLaDA2.1-mini,max_new_tokens=256,steps_per_block=32,block_size=32,temperature=0.0,eos_early_stop=True" \
        --confirm_run_unsafe_code
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.llada2.sampler import even_transfer_schedule, sample_tokens


@dataclass
class LLaDA21BlockSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: Optional[int] = None
    block_size: int = 32
    steps_per_block: int = 32
    steps: Optional[int] = None  # total steps; derives steps_per_block if set
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    remasking: str = "low_confidence"  # "low_confidence" or "random"
    schedule_type: str = "even"  # "even" or "scheduler"
    stochastic_transfer: bool = False  # only for schedule_type="scheduler"
    eos_early_stop: bool = False
    cfg_scale: float = 0.0
    suppress_tokens: Optional[list[int]] = None


@dataclass
class LLaDA21BlockSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDA21BlockSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Standard block diffusion sampler for LLaDA2.1.

        Uses block-causal attention and mask-filled canvas (matching LLaDA2.1 training)
        with a fixed unmasking schedule and pure topk confidence selection.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
                    All prompts must have equal length.
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = LLaDA21BlockSamplerConfig()

        block_size = kwargs.get("block_size", config.block_size)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        steps_per_block = kwargs.get("steps_per_block", config.steps_per_block)
        steps = kwargs.get("steps", config.steps)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        remasking = kwargs.get("remasking", config.remasking)
        schedule_type = kwargs.get("schedule_type", config.schedule_type)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        eos_early_stop = kwargs.get("eos_early_stop", config.eos_early_stop)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        return_dict = kwargs.get("return_dict", config.return_dict)

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Normalize inputs -----
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]
        if len(set(prompt_lens)) != 1:
            raise ValueError(
                "LLaDA21BlockSampler expects all prompts to have the same length."
            )

        prompt_len = prompt_lens[0]
        B = len(inputs)

        if max_new_tokens:
            max_length = max_new_tokens + prompt_len
        else:
            max_new_tokens = max_length - prompt_len

        num_blocks = (max_length + block_size - 1) // block_size
        total_len = num_blocks * block_size
        prompt_blocks = prompt_len // block_size
        num_gen_blocks = num_blocks - prompt_blocks

        # Resolve steps_per_block from total steps if provided
        if steps is not None:
            steps_per_block = max(1, steps // num_gen_blocks)

        # ----- Block-causal attention (log-transformed additive mask) -----
        block_mask = torch.tril(
            torch.ones(num_blocks, num_blocks, device=self.model.device)
        )
        block_attn = (
            (
                block_mask.repeat_interleave(block_size, dim=0)
                .repeat_interleave(block_size, dim=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            .log()
            .to(torch.bfloat16)
        )

        position_ids = torch.arange(total_len, device=self.model.device).unsqueeze(0)

        # ----- Canvas initialized with masks, prompts at front -----
        x = torch.full(
            (B, total_len),
            mask_id,
            dtype=torch.long,
            device=self.model.device,
        )
        for i, p in enumerate(inputs):
            x[i, :prompt_len] = p

        # Track unmasked (prompt) positions for CFG
        if cfg_scale > 0.0:
            unmasked_index = torch.zeros(
                (B, total_len), dtype=torch.bool, device=self.model.device
            )
            unmasked_index[:, :prompt_len] = True

        histories = [x.clone()] if return_dict else None

        # ----- Block loop -----
        for blk in range(prompt_blocks, num_blocks):
            window_end = (blk + 1) * block_size
            cur_attn = block_attn[:, :, :window_end, :window_end]
            cur_pos = position_ids[:, :window_end]

            # Build per-block mask index for scheduler-based schedule
            block_start = blk * block_size
            block_slice = x[:, block_start:window_end]
            active_in_block = block_slice == mask_id  # [B, block_size]

            # Compute schedule for this block
            if schedule_type == "scheduler":
                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index=active_in_block,
                    steps=steps_per_block,
                    scheduler=self.scheduler,
                    stochastic=stochastic_transfer,
                )
                effective_steps = num_transfer_tokens.size(1)
            else:
                transfer_schedule = even_transfer_schedule(block_size, steps_per_block)
                effective_steps = steps_per_block

            for step_idx in range(effective_steps):
                # Check if any masks remain in this block
                block_slice = x[:, block_start:window_end]
                active_mask = block_slice == mask_id
                if not active_mask.any():
                    break

                # ----- Forward pass -----
                if cfg_scale > 0.0:
                    un_x = x[:, :window_end].clone()
                    un_x[unmasked_index[:, :window_end]] = mask_id
                    x_ = torch.cat([x[:, :window_end], un_x], dim=0)
                    logits = self.model(
                        x_,
                        attention_mask=cur_attn.expand(2, -1, -1, -1),
                        position_ids=cur_pos.expand(2, -1),
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x[:, :window_end],
                        attention_mask=cur_attn,
                        position_ids=cur_pos,
                    ).logits

                logits_block = logits[:, -block_size:, :]

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits_block[:, :, token_id] = -torch.inf

                # ----- Sample tokens and get confidence -----
                tokens, probs = sample_tokens(
                    logits_block, temperature=temperature, top_k=top_k, top_p=top_p
                )

                # Confidence for topk selection
                if remasking == "low_confidence":
                    confidence = probs
                elif remasking == "random":
                    confidence = torch.rand_like(probs)
                else:
                    raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

                # Determine k for this step
                if schedule_type == "scheduler":
                    # Per-sample k from scheduler
                    per_sample_k = num_transfer_tokens[:, step_idx]  # [B]
                else:
                    k_this_step = int(transfer_schedule[step_idx].item())

                # ----- Pure topk selection -----
                transfer_index = torch.zeros_like(active_mask, dtype=torch.bool)
                for b in range(B):
                    conf = torch.where(
                        active_mask[b],
                        confidence[b],
                        torch.full_like(confidence[b], -float("inf")),
                    )
                    num_available = active_mask[b].sum().item()
                    if schedule_type == "scheduler":
                        k = int(per_sample_k[b].item())
                    else:
                        k = k_this_step
                    actual_k = min(k, num_available)
                    if actual_k > 0:
                        _, idx = torch.topk(conf, k=actual_k)
                        transfer_index[b, idx] = True

                # Commit predictions
                x[:, block_start:window_end][transfer_index] = tokens[transfer_index]

                if histories is not None:
                    histories.append(x.clone())

            # ----- EOS early stopping between blocks -----
            if eos_early_stop and eos_id is not None:
                generated_part = x[:, prompt_len:window_end]
                has_no_masks = (generated_part == mask_id).sum(dim=1) == 0
                has_eos = (generated_part == eos_id).any(dim=1)
                if (has_no_masks & has_eos).all():
                    break

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDA21BlockSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError
