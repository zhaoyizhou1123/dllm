"""
Confidence-based block diffusion sampler for LLaDA2.1 with Gibbs correction.

Run:
    python dllm/pipelines/llada21/eval.py \
        --tasks humaneval_instruct_llada --num_fewshot 0 \
        --model llada21_confidence_block --apply_chat_template \
        --batch_size 1 \
        --model_args "pretrained=inclusionAI/LLaDA2.1-mini,max_new_tokens=256,block_size=32,threshold=0.9,min_transfer=1,temperature=0.0,eos_early_stop=True,edit_freq=1,edit_step=10,edit_strategy=gibbs_standard,remasking_strategy=random" \
        --confirm_run_unsafe_code
"""

from dataclasses import dataclass
from typing import Optional

import torch

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.pipelines.llada2.sampler import sample_tokens
from dllm.pipelines.llada21.gibbs_block_sampler import (
    block_forward,
    gibbs_correct,
)


@dataclass
class LLaDA21ConfidenceBlockSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: Optional[int] = None
    block_size: int = 32
    threshold: float = 0.9
    min_transfer: int = 1
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    eos_early_stop: bool = False
    cfg_scale: float = 0.0
    suppress_tokens: Optional[list[int]] = None
    edit_freq: int = -1
    edit_step: int = 0
    edit_start: int = 0
    edit_strategy: str = "gibbs_standard"
    remasking_strategy: str = "random"
    keep_original_mask: bool = True
    early_exit_number: int = 5
    skip_blocks: int = 0


@dataclass
class LLaDA21ConfidenceBlockSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDA21ConfidenceBlockSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = LLaDA21ConfidenceBlockSamplerConfig()

        block_size = kwargs.get("block_size", config.block_size)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        threshold = float(kwargs.get("threshold", config.threshold))
        min_transfer = int(kwargs.get("min_transfer", config.min_transfer))
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        eos_early_stop = kwargs.get("eos_early_stop", config.eos_early_stop)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        return_dict = kwargs.get("return_dict", config.return_dict)
        edit_freq = int(kwargs.get("edit_freq", config.edit_freq))
        edit_step = int(kwargs.get("edit_step", config.edit_step))
        edit_start = int(kwargs.get("edit_start", config.edit_start))
        edit_strategy = kwargs.get("edit_strategy", config.edit_strategy)
        remasking_strategy = kwargs.get("remasking_strategy", config.remasking_strategy)
        keep_original_mask = bool(kwargs.get("keep_original_mask", config.keep_original_mask))
        early_exit_number = int(kwargs.get("early_exit_number", config.early_exit_number))
        skip_blocks = int(kwargs.get("skip_blocks", config.skip_blocks))

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]
        if len(set(prompt_lens)) != 1:
            raise ValueError(
                "LLaDA21ConfidenceBlockSampler expects all prompts to have the same length."
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

        x = torch.full(
            (B, total_len), mask_id, dtype=torch.long, device=self.model.device,
        )
        for i, p in enumerate(inputs):
            x[i, :prompt_len] = p

        if cfg_scale > 0.0:
            unmasked_index = torch.zeros(
                (B, total_len), dtype=torch.bool, device=self.model.device
            )
            unmasked_index[:, :prompt_len] = True
        else:
            unmasked_index = None

        histories = [x.clone()] if return_dict else None
        global_step = 0

        for blk in range(prompt_blocks, num_blocks):
            blk_start = blk * block_size
            window_end = (blk + 1) * block_size
            cur_attn = block_attn[:, :, :window_end, :window_end]
            cur_pos = position_ids[:, :window_end]

            prompt_mask_in_block = torch.zeros(
                block_size, dtype=torch.bool, device=self.model.device
            )
            if blk_start < prompt_len:
                prompt_mask_in_block[: min(prompt_len - blk_start, block_size)] = True

            def forward_fn(_we=window_end, _ca=cur_attn, _cp=cur_pos):
                if cfg_scale > 0.0:
                    unmasked_index[:, blk_start:_we] = (
                        x[:, blk_start:_we] != mask_id
                    )
                return block_forward(
                    self.model, x, _we, block_size, _ca, _cp,
                    cfg_scale, unmasked_index, suppress_tokens, mask_id,
                )

            for gen_step in range(block_size):
                block_slice = x[:, blk_start:window_end]
                active_mask = block_slice == mask_id
                if not active_mask.any():
                    break

                logits_block = forward_fn()

                if (
                    edit_freq > 0
                    and (global_step + 1) % edit_freq == 0
                    and gen_step >= edit_start
                    and edit_step > 0
                    and blk - prompt_blocks >= skip_blocks
                ):
                    logits_block = gibbs_correct(
                        x, blk_start, window_end, logits_block,
                        prompt_mask_in_block, mask_id, edit_step,
                        edit_strategy, remasking_strategy,
                        keep_original_mask, early_exit_number,
                        forward_fn,
                    )
                    block_slice = x[:, blk_start:window_end]
                    active_mask = block_slice == mask_id
                    if not active_mask.any():
                        break

                tokens, probs = sample_tokens(
                    logits_block, temperature=temperature,
                    top_k=top_k, top_p=top_p,
                )

                transfer_index = torch.zeros_like(active_mask, dtype=torch.bool)
                for b in range(B):
                    conf = torch.where(
                        active_mask[b], probs[b],
                        torch.full_like(probs[b], -float("inf")),
                    )
                    high_conf = (conf > threshold) & active_mask[b]
                    if high_conf.sum().item() >= min_transfer:
                        transfer_index[b] = high_conf
                    else:
                        num_available = active_mask[b].sum().item()
                        k = min(min_transfer, num_available)
                        if k > 0:
                            _, idx = torch.topk(conf, k=k)
                            transfer_index[b, idx] = True

                x[:, blk_start:window_end][transfer_index] = tokens[transfer_index]

                if histories is not None:
                    histories.append(x.clone())

                global_step += 1

            # -- Post-editing: refine non-prompt tokens without remasking --
            if edit_step > 0:
                non_prompt = ~prompt_mask_in_block.unsqueeze(0).expand(B, -1)
                prev_block = None
                consecutive_unchanged = 0

                for _edit_iter in range(edit_step):
                    logits_block = forward_fn()
                    tokens, _probs = sample_tokens(
                        logits_block, temperature=temperature,
                        top_k=top_k, top_p=top_p,
                    )

                    block_slice = x[:, blk_start:window_end]
                    new_block = torch.where(non_prompt, tokens, block_slice)
                    x[:, blk_start:window_end] = new_block

                    if histories is not None:
                        histories.append(x.clone())

                    if early_exit_number > 0:
                        if prev_block is not None and torch.equal(new_block, prev_block):
                            consecutive_unchanged += 1
                            if consecutive_unchanged >= early_exit_number:
                                break
                        else:
                            consecutive_unchanged = 0
                        prev_block = new_block.clone()

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
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
