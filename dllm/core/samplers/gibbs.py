"""
Gibbs-style correction sampler for masked diffusion LMs.

Ported from ~/projects/mdm_correction/sampling.py. Runs the standard MDLM
outer unmasking loop and, every `edit_freq` steps, triggers an inner Gibbs
correction that (re)masks a subset of tokens, re-forwards the model, and
reveals again — iteratively escaping local optima during decoding.

With `edit_freq=-1` (default) the correction path is disabled and this
sampler produces identical output to ``MDLMSampler``.

Three variants, selected via ``edit_strategy``:
  - ``gibbs_standard``  — inner reveal touches *only currently-masked* tokens.
  - ``gibbs_edit``      — inner reveal touches *all non-prompt* tokens; final
                           step restores the original mask pattern in ``xt``.
  - ``gibbs_edit_v2``   — inner reveal touches all non-prompt tokens; final
                           step remasks `num_masks` positions.

Run: used via the lm-eval harness, e.g.::

    accelerate launch dllm/pipelines/llada/eval.py \\
        --tasks humaneval_instruct_llada --num_fewshot 0 \\
        --model llada_gibbs --apply_chat_template \\
        --model_args "pretrained=<ckpt>,edit_freq=1,edit_step=10,\\
edit_strategy=gibbs_edit,remasking_strategy=random,..."
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class GibbsSamplerConfig(MDLMSamplerConfig):
    """MDLMSamplerConfig + Gibbs-correction fields."""

    edit_freq: int = -1  # run Gibbs every k outer steps; -1 disables
    edit_step: int = 0  # S: inner Gibbs iterations per trigger
    edit_start: int = 0  # first outer step index eligible for Gibbs
    edit_strategy: str = "gibbs_standard"  # "gibbs_standard" | "gibbs_edit" | "gibbs_edit_v2"
    remasking_strategy: str = "random"  # "random" | "low_confidence"
    threshold: float | None = None  # skip Gibbs if avg response confidence exceeds this
    keep_original_mask: bool = True  # final remask of gibbs_standard/gibbs_edit uses input-x mask pattern; False → remasking_strategy


@dataclass
class GibbsSampler(MDLMSampler):
    """MDLM sampler with periodic Gibbs correction."""

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: GibbsSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = GibbsSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        # Gibbs-specific
        edit_freq = int(kwargs.get("edit_freq", config.edit_freq))
        edit_step = int(kwargs.get("edit_step", config.edit_step))
        edit_start = int(kwargs.get("edit_start", config.edit_start))
        edit_strategy = kwargs.get("edit_strategy", config.edit_strategy)
        remasking_strategy = kwargs.get("remasking_strategy", config.remasking_strategy)
        threshold = kwargs.get("threshold", config.threshold)
        keep_original_mask = bool(
            kwargs.get("keep_original_mask", config.keep_original_mask)
        )

        assert 1 <= block_size
        assert 1 <= steps
        assert edit_strategy in ("gibbs_standard", "gibbs_edit", "gibbs_edit_v2")
        assert remasking_strategy in ("random", "low_confidence")

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping (mirrors MDLMSampler.sample) -----
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Canvas + attention mask -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = mask_id
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # prompt_mask: True outside the per-sample generation region.
        # Gibbs must not remask prompt positions nor right-pad EOS.
        prompt_mask = torch.ones((B, T), dtype=torch.bool, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            gen_end = min(pl + max_new_tokens, T)
            prompt_mask[i, pl:gen_end] = False

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block schedule -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        global_step = 0  # counter across all blocks (drives edit_freq)

        for b in range(num_blocks):
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = x[j, start:end] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for i in range(effective_steps):
                mask_index = x == mask_id

                # ----- Forward (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf
                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # ----- Gibbs correction hook -----
                if (
                    edit_freq > 0
                    and (global_step + 1) % edit_freq == 0
                    and global_step >= edit_start
                    and edit_step > 0
                ):
                    x, logits = self._gibbs_correct(
                        x=x,
                        logits=logits,
                        prompt_mask=prompt_mask,
                        attention_mask=attention_mask,
                        mask_id=mask_id,
                        edit_step=edit_step,
                        edit_strategy=edit_strategy,
                        remasking_strategy=remasking_strategy,
                        threshold=threshold,
                        suppress_tokens=suppress_tokens,
                        begin_suppress_tokens=begin_suppress_tokens,
                        right_shift_logits=right_shift_logits,
                        keep_original_mask=keep_original_mask,
                    )
                    mask_index = x == mask_id  # Gibbs may have remasked new positions

                # ----- Argmax + confidence + topk commit -----
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )
                else:
                    raise NotImplementedError(remasking)

                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

                global_step += 1

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    # ─────────────────────── Gibbs correction helpers ───────────────────────

    @torch.no_grad()
    def _gibbs_forward(
        self,
        yt: torch.Tensor,
        attention_mask: torch.Tensor,
        suppress_tokens,
        right_shift_logits: bool,
    ) -> torch.Tensor:
        """Forward used inside Gibbs; matches outer-loop suppress_tokens + right_shift parity."""
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            logits = self.model(yt, attention_mask=attention_mask).logits
        if suppress_tokens is not None and len(suppress_tokens) > 0:
            for token_id in suppress_tokens:
                logits[:, :, token_id] = -torch.inf
        if right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits

    @staticmethod
    def _compute_remask(
        B: int,
        L: int,
        num_masks: torch.Tensor,
        k_max: int,
        prompt_mask: torch.Tensor,
        logits: torch.Tensor,
        remasking_strategy: str,
        begin_suppress_tokens,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a (B, L) bool remask mask selecting `num_masks[b]` positions per sample."""
        if remasking_strategy == "low_confidence":
            _logits = logits
            if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                _logits = logits.clone()
                for tid in begin_suppress_tokens:
                    _logits[:, :, tid] = -torch.inf
            conf = F.softmax(_logits, dim=-1).max(dim=-1).values  # (B, L)
            remask_scores = torch.where(
                ~prompt_mask,
                -conf,
                torch.full((B, L), -float("inf"), device=device),
            )
        else:  # "random"
            remask_scores = torch.where(
                ~prompt_mask,
                torch.rand(B, L, device=device),
                torch.full((B, L), -float("inf"), device=device),
            )
        _, top_indices = torch.topk(remask_scores, k=k_max, dim=-1)  # (B, k_max)
        valid = torch.arange(k_max, device=device).unsqueeze(0) < num_masks.unsqueeze(1)
        remask = torch.zeros(B, L, dtype=torch.bool, device=device)
        remask.scatter_(1, top_indices, valid)
        return remask

    @torch.no_grad()
    def _gibbs_correct(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        prompt_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_id: int,
        edit_step: int,
        edit_strategy: str,
        remasking_strategy: str,
        threshold: float | None,
        suppress_tokens,
        begin_suppress_tokens,
        right_shift_logits: bool,
        keep_original_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to the correct Gibbs variant and return (x_new, logits_new)."""
        # Optional threshold early-exit: if the response region is already
        # confident, shrink edit_step to 1 (cheap one-shot correction).
        if threshold is not None:
            p = F.softmax(logits, dim=-1)
            token_prob = p.gather(-1, x.unsqueeze(-1)).squeeze(-1)
            response_mask = ~prompt_mask
            if response_mask.any():
                avg_conf = token_prob[response_mask].mean().item()
            else:
                avg_conf = 1.0
            if avg_conf > threshold:
                edit_step = 1
            del p, token_prob

        if edit_strategy == "gibbs_standard":
            return self._gibbs_standard(
                x, logits, prompt_mask, attention_mask, mask_id,
                edit_step, remasking_strategy,
                suppress_tokens, begin_suppress_tokens, right_shift_logits,
                keep_original_mask,
            )
        if edit_strategy == "gibbs_edit":
            return self._gibbs_edit_v1(
                x, logits, prompt_mask, attention_mask, mask_id,
                edit_step, remasking_strategy,
                suppress_tokens, begin_suppress_tokens, right_shift_logits,
                keep_original_mask,
            )
        if edit_strategy == "gibbs_edit_v2":
            return self._gibbs_edit_v2(
                x, logits, prompt_mask, attention_mask, mask_id,
                edit_step, remasking_strategy,
                suppress_tokens, begin_suppress_tokens, right_shift_logits,
            )
        raise ValueError(f"Unknown edit_strategy: {edit_strategy}")

    @torch.no_grad()
    def _gibbs_standard(
        self,
        x, logits, prompt_mask, attention_mask, mask_id,
        edit_step, remasking_strategy,
        suppress_tokens, begin_suppress_tokens, right_shift_logits,
        keep_original_mask,
    ):
        """Ported from mdm_gibbs_standard_sampling. Inner reveal touches only masked positions."""
        B, L = x.shape
        device = x.device
        original_mask = x == mask_id
        num_masks = original_mask.sum(dim=-1)

        new_pred = torch.argmax(logits, dim=-1)
        yt = torch.where(original_mask, new_pred, x)

        for _ in range(edit_step):
            k_max = int(num_masks.max().item())
            if k_max > 0:
                if remasking_strategy == "low_confidence":
                    # Need fresh confidence on the current yt
                    logits = self._gibbs_forward(
                        yt, attention_mask, suppress_tokens, right_shift_logits
                    )
                    new_pred = torch.argmax(logits, dim=-1)
                    yt = torch.where(yt == mask_id, new_pred, yt)
                remask = self._compute_remask(
                    B, L, num_masks, k_max, prompt_mask,
                    logits, remasking_strategy, begin_suppress_tokens, device,
                )
                yt = yt.clone()
                yt[remask] = mask_id

            logits = self._gibbs_forward(
                yt, attention_mask, suppress_tokens, right_shift_logits
            )
            new_pred = torch.argmax(logits, dim=-1)
            yt = torch.where(yt == mask_id, new_pred, yt)

        # Final: fresh forward, full reveal, then remask.
        logits = self._gibbs_forward(
            yt, attention_mask, suppress_tokens, right_shift_logits
        )
        new_pred = torch.argmax(logits, dim=-1)
        yt = torch.where(yt == mask_id, new_pred, yt)
        if keep_original_mask:
            if original_mask.any():
                yt = yt.clone()
                yt[original_mask] = mask_id
        else:
            k_max = int(num_masks.max().item())
            if k_max > 0:
                remask = self._compute_remask(
                    B, L, num_masks, k_max, prompt_mask,
                    logits, remasking_strategy, begin_suppress_tokens, device,
                )
                yt = yt.clone()
                yt[remask] = mask_id
        return yt, logits

    @torch.no_grad()
    def _gibbs_edit_v1(
        self,
        x, logits, prompt_mask, attention_mask, mask_id,
        edit_step, remasking_strategy,
        suppress_tokens, begin_suppress_tokens, right_shift_logits,
        keep_original_mask,
    ):
        """Ported from mdm_gibbs_edit_sampling. Final step restores the original mask pattern."""
        B, L = x.shape
        device = x.device
        unmask_indices = x != mask_id  # frozen snapshot of originally-unmasked positions
        num_masks = (x == mask_id).sum(dim=-1)

        new_pred = torch.argmax(logits, dim=-1)
        yt = torch.where(~prompt_mask, new_pred, x)

        for _ in range(edit_step):
            k_max = int(num_masks.max().item())
            if k_max > 0:
                remask = self._compute_remask(
                    B, L, num_masks, k_max, prompt_mask,
                    logits, remasking_strategy, begin_suppress_tokens, device,
                )
                yt = yt.clone()
                yt[remask] = mask_id

            logits = self._gibbs_forward(
                yt, attention_mask, suppress_tokens, right_shift_logits
            )
            new_pred = torch.argmax(logits, dim=-1)
            yt = torch.where(~prompt_mask, new_pred, x)

        if keep_original_mask:
            # Restore original mask pattern: originally unmasked positions take yt's
            # latest prediction; originally masked positions stay as mask_id.
            xt_out = torch.where(unmask_indices, yt, x)
        else:
            # yt already holds predictions for all non-prompt positions; pick
            # num_masks remask positions via remasking_strategy.
            k_max = int(num_masks.max().item())
            if k_max > 0:
                remask = self._compute_remask(
                    B, L, num_masks, k_max, prompt_mask,
                    logits, remasking_strategy, begin_suppress_tokens, device,
                )
                yt = yt.clone()
                yt[remask] = mask_id
            xt_out = yt
        return xt_out, logits

    @torch.no_grad()
    def _gibbs_edit_v2(
        self,
        x, logits, prompt_mask, attention_mask, mask_id,
        edit_step, remasking_strategy,
        suppress_tokens, begin_suppress_tokens, right_shift_logits,
    ):
        """Ported from mdm_gibbs_edit_sampling_v2. Final step remasks num_masks positions on yt."""
        B, L = x.shape
        device = x.device
        num_masks = (x == mask_id).sum(dim=-1)

        new_pred = torch.argmax(logits, dim=-1)
        yt = torch.where(~prompt_mask, new_pred, x)

        for _ in range(edit_step):
            k_max = int(num_masks.max().item())
            if k_max > 0:
                remask = self._compute_remask(
                    B, L, num_masks, k_max, prompt_mask,
                    logits, remasking_strategy, begin_suppress_tokens, device,
                )
                yt = yt.clone()
                yt[remask] = mask_id

            logits = self._gibbs_forward(
                yt, attention_mask, suppress_tokens, right_shift_logits
            )
            new_pred = torch.argmax(logits, dim=-1)
            yt = torch.where(~prompt_mask, new_pred, x)

        # Final: reveal all non-prompt, then remask num_masks positions.
        new_pred = torch.argmax(logits, dim=-1)
        yt = torch.where(~prompt_mask, new_pred, x)
        k_max = int(num_masks.max().item())
        if k_max > 0:
            remask = self._compute_remask(
                B, L, num_masks, k_max, prompt_mask,
                logits, remasking_strategy, begin_suppress_tokens, device,
            )
            yt = yt.clone()
            yt[remask] = mask_id
        return yt, logits
