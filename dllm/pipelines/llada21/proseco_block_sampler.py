"""
ProSeCo inference-only baseline for LLaDA2.1 block diffusion.

Block-local port of `mdm_multi_proseco_sampling` (`mdm_correction/sampling.py`).
ProSeCo inference is the "no-remask (t=0)" corner of the remask-denoise family: each
step it (a) unmasks a few tokens by confidence like standard decoding, and (b)
re-predicts *all* already-revealed (non-prompt) tokens with the model's fresh argmax
— but it NEVER remasks. It runs on any pretrained denoiser (uses the same model, no
separate corrector head), which is exactly why it is a valid inference-only baseline.

LLaDA2.1 is semi-autoregressive block diffusion, so the procedure is applied
block-locally (same adaptation as the R2D and ReMDM block samplers): unmask + full
re-prediction happen within the current block, then `correction_step` extra
re-prediction passes over the fully-revealed block scale test-time compute.

Run:
    python dllm/pipelines/llada21/eval.py \
        --tasks humaneval_instruct_llada --num_fewshot 0 \
        --model llada21_proseco --apply_chat_template \
        --batch_size 1 \
        --model_args "pretrained=inclusionAI/LLaDA2.1-mini,max_new_tokens=512,block_size=32,unmasking_num=1,correction_step=8,temperature=0.0" \
        --confirm_run_unsafe_code
"""

from dataclasses import dataclass
from typing import Optional

import torch

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.pipelines.llada2.sampler import sample_tokens
from dllm.pipelines.llada21.gibbs_block_sampler import block_forward


@dataclass
class LLaDA21ProSeCoBlockSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: Optional[int] = None
    block_size: int = 32
    unmasking_num: int = 1
    correction_step: int = 0  # extra full re-prediction passes per block (swept)
    confidence: str = "top_k"  # "top_k" (max prob) | "random"
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    eos_early_stop: bool = False
    cfg_scale: float = 0.0
    suppress_tokens: Optional[list[int]] = None


@dataclass
class LLaDA21ProSeCoBlockSampler(BaseSampler):
    supports_nfe = True

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDA21ProSeCoBlockSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = LLaDA21ProSeCoBlockSamplerConfig()

        block_size = kwargs.get("block_size", config.block_size)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        unmasking_num = int(kwargs.get("unmasking_num", config.unmasking_num))
        correction_step = int(kwargs.get("correction_step", config.correction_step))
        confidence = kwargs.get("confidence", config.confidence)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        eos_early_stop = kwargs.get("eos_early_stop", config.eos_early_stop)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        return_dict = kwargs.get("return_dict", config.return_dict)
        return_histories = kwargs.get("return_histories", return_dict)

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
                "LLaDA21ProSeCoBlockSampler expects all prompts to have the same length."
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

        histories = [x.clone()] if return_histories else None
        nfe_counter = [0]

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
            valid_block = ~prompt_mask_in_block.unsqueeze(0).expand(B, -1)  # [B, Lb]

            def forward_fn(_we=window_end, _ca=cur_attn, _cp=cur_pos):
                nfe_counter[0] += 1
                if cfg_scale > 0.0:
                    unmasked_index[:, blk_start:_we] = x[:, blk_start:_we] != mask_id
                return block_forward(
                    self.model, x, _we, block_size, _ca, _cp,
                    cfg_scale, unmasked_index, suppress_tokens, mask_id,
                )

            if not ((x[:, blk_start:window_end] == mask_id) & valid_block).any():
                continue

            # Reveal all block positions (block_size // unmasking_num steps) then
            # run `correction_step` extra full re-prediction passes.
            num_iter = max(1, block_size // max(unmasking_num, 1)) + correction_step

            for _i in range(num_iter):
                block = x[:, blk_start:window_end]
                active_mask = (block == mask_id) & valid_block  # still-masked response tokens

                logits_block = forward_fn()  # [B, Lb, V]
                tokens, probs = sample_tokens(
                    logits_block, temperature=temperature, top_k=top_k, top_p=top_p
                )

                if confidence == "top_k":
                    conf_all = probs
                elif confidence == "random":
                    conf_all = torch.rand_like(probs)
                else:
                    raise NotImplementedError(f"Unknown confidence strategy: {confidence}")

                # Select unmasking_num masked positions to newly reveal (per sample).
                update_mask = torch.zeros_like(active_mask, dtype=torch.bool)
                for b in range(B):
                    conf = torch.where(
                        active_mask[b], conf_all[b],
                        torch.full_like(conf_all[b], -float("inf")),
                    )
                    num_available = active_mask[b].sum().item()
                    k = min(unmasking_num, num_available)
                    if k > 0:
                        _, idx = torch.topk(conf, k=k)
                        update_mask[b, idx] = True

                # ProSeCo: re-predict every already-revealed response token + newly
                # revealed ones; never remask.
                reveal_indices = (~active_mask) & valid_block
                write = update_mask | reveal_indices
                x[:, blk_start:window_end] = torch.where(write, tokens, block)

                if histories is not None:
                    histories.append(x.clone())

            # Safety fill (should be a no-op: num_iter reveals the whole block).
            for _ in range(block_size):
                block = x[:, blk_start:window_end]
                residual = (block == mask_id) & valid_block
                if not residual.any():
                    break
                logits_block = forward_fn()
                pred = torch.argmax(logits_block, dim=-1)
                x[:, blk_start:window_end] = torch.where(residual, pred, block)
                if histories is not None:
                    histories.append(x.clone())

            if eos_early_stop and eos_id is not None:
                generated_part = x[:, prompt_len:window_end]
                has_no_masks = (generated_part == mask_id).sum(dim=1) == 0
                has_eos = (generated_part == eos_id).any(dim=1)
                if (has_no_masks & has_eos).all():
                    break

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories, nfe=[nfe_counter[0]] * B)

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
