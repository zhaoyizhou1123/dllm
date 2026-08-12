"""
ReMDM inference-only baseline for LLaDA2.1 block diffusion.

Block-local port of the discrete nested-loop ReMDM sampler
(`mdm_correction/sampling_remdm_discrete.py`). LLaDA2.1 is semi-autoregressive
block diffusion with a block-causal attention mask, so the ReMDM remask-denoise
kernel is applied *within the current block* — the same block-local adaptation used
by the R2D `confidence_block` / `gibbs_block` samplers. This keeps ReMDM, R2D, and
ProSeCo on equal footing (all block-local) for a fair accuracy-vs-NFE comparison on
the same frozen LLaDA2.1-mini denoiser.

Per block: an OUTER loop over decoding steps (each with its own (alpha_t, alpha_s)
transition from a discrete time grid) wraps an INNER correction loop of `edit_step`
iterations at the SAME noise level, with per-sample early exit when consecutive x0
predictions converge. ReMDM's defining property — remasking already-revealed tokens
(sigma > 0) — is preserved; that is what distinguishes it from R2D.

Reference: ReMDM, arXiv:2503.00307 (Kuleshov group). Selection routing / variants
mirror `mdm_correction/sampling_remdm.py`.

Run:
    python dllm/pipelines/llada21/eval.py \
        --tasks humaneval_instruct_llada --num_fewshot 0 \
        --model llada21_remdm --apply_chat_template \
        --batch_size 1 \
        --model_args "pretrained=inclusionAI/LLaDA2.1-mini,max_new_tokens=512,block_size=32,variant=cap,eta=0.4,edit_step=8,early_exit_number=5,temperature=0.0" \
        --confirm_run_unsafe_code
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.pipelines.llada21.gibbs_block_sampler import block_forward


# ── ReMDM math helpers (ported from mdm_correction/sampling_remdm.py) ──────────

NEG_INF = -1e9


def _subs_p_x0(
    logits: torch.Tensor, block: torch.Tensor, mask_id: int
) -> torch.Tensor:
    """SUBS carry-over parameterization -> p_x0 (probabilities).

    Mirrors official ReMDM `_subs_parameterization` (kuleshov-group/remdm
    diffusion.py:264-280); validated block-for-block in
    `mdm_correction/sampling_remdm_faithful.py::_subs_p_x0`:
      * mask logit -> -inf  (p_x0 places zero mass on the mask token), and
      * carry-over: already-decoded positions (block != mask) are forced one-hot
        on the current token, so a revealed token can only STAY (prob 1-sigma) or
        be REMASKED (prob sigma) -- it is never re-predicted from the full vocab.
    `logits` is expected already divided by the posterior temperature.
    """
    logits = logits.float().clone()
    logits[..., mask_id] = NEG_INF
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked = block != mask_id
    logits[unmasked] = NEG_INF
    logits[unmasked, block[unmasked]] = 0.0
    return logits.exp()


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    """Categorical sampling with fp64 Gumbel noise (matches ReMDM upstream)."""
    categorical_probs = categorical_probs.to(torch.float64)
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _sigma_max(alpha_t: torch.Tensor, alpha_s: torch.Tensor) -> torch.Tensor:
    sigma_max = torch.ones_like(alpha_t)
    active = alpha_t > 0
    sigma_max[active] = torch.minimum(
        torch.ones_like(alpha_t[active]),
        (1.0 - alpha_s[active]) / alpha_t[active],
    )
    return sigma_max


def _apply_nucleus(p_x0: torch.Tensor, nucleus_p: float) -> torch.Tensor:
    """Top-p filter on p_x0 (disabled when nucleus_p >= 1.0)."""
    if nucleus_p >= 1.0:
        return p_x0
    sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cumulative_probs <= nucleus_p
    top_p_mask[..., 0] = True
    nucleus_probs = sorted_probs * top_p_mask
    nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)


@dataclass
class LLaDA21ReMDMBlockSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: Optional[int] = None
    block_size: int = 32
    unmasking_num: int = 1
    # ReMDM knobs
    variant: str = "cap"  # "cap" | "rescale" | "markovian" | "conf"
    eta: float = 0.4
    edit_step: int = 0  # inner correction iterations per outer step (swept)
    early_exit_number: int = 5
    time_eps: float = 1e-5
    nucleus_p: float = 1.0
    conf_base_variant: str = "repo"
    freeze_unmasked: bool = False
    # SUBS carry-over (official ReMDM parameterization). Default False preserves the
    # original raw-softmax "resample-all" behavior of in-flight runs; True switches
    # to the official carry-over p_x0 (decoded tokens sticky, mask logit -inf).
    carry_over: bool = False
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    eos_early_stop: bool = False
    cfg_scale: float = 0.0
    suppress_tokens: Optional[list[int]] = None


@dataclass
class LLaDA21ReMDMBlockSampler(BaseSampler):
    supports_nfe = True

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDA21ReMDMBlockSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = LLaDA21ReMDMBlockSamplerConfig()

        block_size = kwargs.get("block_size", config.block_size)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        unmasking_num = int(kwargs.get("unmasking_num", config.unmasking_num))
        variant = kwargs.get("variant", config.variant)
        eta = float(kwargs.get("eta", config.eta))
        edit_step = int(kwargs.get("edit_step", config.edit_step))
        early_exit_number = int(kwargs.get("early_exit_number", config.early_exit_number))
        time_eps = float(kwargs.get("time_eps", config.time_eps))
        nucleus_p = float(kwargs.get("nucleus_p", config.nucleus_p))
        conf_base_variant = kwargs.get("conf_base_variant", config.conf_base_variant)
        freeze_unmasked = bool(kwargs.get("freeze_unmasked", config.freeze_unmasked))
        carry_over = bool(kwargs.get("carry_over", config.carry_over))
        temperature = float(kwargs.get("temperature", config.temperature))
        eos_early_stop = kwargs.get("eos_early_stop", config.eos_early_stop)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        return_dict = kwargs.get("return_dict", config.return_dict)
        return_histories = kwargs.get("return_histories", return_dict)

        posterior_temp = 1.0 if temperature <= 0.0 else temperature

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
                "LLaDA21ReMDMBlockSampler expects all prompts to have the same length."
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

        vocab_hint = None  # inferred from first logits

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

            # If this block has no masked (generatable) positions, skip it.
            if not ((x[:, blk_start:window_end] == mask_id) & valid_block).any():
                continue

            # Discrete time grid for this block's outer loop.
            num_outer = max(1, block_size // max(unmasking_num, 1))
            ts = torch.linspace(
                1.0, time_eps, num_outer + 1,
                device=self.model.device, dtype=torch.float32,
            )

            conf_scores = None
            if variant == "conf":
                conf_scores = torch.full(
                    (B, block_size), float("-inf"), device=self.model.device
                )

            for outer in range(num_outer):
                alpha_t = torch.full((B,), 1.0 - ts[outer].item(),
                                     device=self.model.device, dtype=torch.float32)
                alpha_s = torch.full((B,), 1.0 - ts[outer + 1].item(),
                                     device=self.model.device, dtype=torch.float32)
                sigma_max_val = _sigma_max(alpha_t, alpha_s)

                prev_x0 = None
                frozen = torch.zeros(B, dtype=torch.bool, device=self.model.device)
                consecutive = torch.zeros(B, dtype=torch.long, device=self.model.device)

                for _inner in range(1 + edit_step):
                    block = x[:, blk_start:window_end]
                    mask_indices = (block == mask_id) & valid_block
                    if not mask_indices.any():
                        break

                    logits_block = forward_fn().float()  # [B, Lb, V]
                    if vocab_hint is None:
                        vocab_hint = logits_block.shape[-1]

                    p_x0 = F.softmax(logits_block / posterior_temp, dim=-1)
                    if carry_over:
                        p_x0 = _subs_p_x0(logits_block / posterior_temp, block, mask_id)
                    p_x0 = _apply_nucleus(p_x0, nucleus_p)

                    if early_exit_number > 0:
                        curr_x0 = torch.argmax(p_x0, dim=-1)
                        if prev_x0 is not None:
                            unchanged = ((curr_x0 == prev_x0) | ~valid_block).all(dim=1)
                            consecutive = torch.where(
                                unchanged & ~frozen, consecutive + 1,
                                torch.where(frozen, consecutive, torch.zeros_like(consecutive)),
                            )
                            newly = (consecutive >= early_exit_number) & ~frozen
                            frozen = frozen | newly
                            if frozen.all():
                                break
                        prev_x0 = curr_x0

                    # ── sigma (remask intensity) ──
                    if variant == "cap":
                        sigma = torch.minimum(torch.full_like(alpha_t, eta), sigma_max_val)[:, None]
                    elif variant == "rescale":
                        sigma = (eta * sigma_max_val)[:, None]
                    elif variant == "markovian":
                        sigma = (1.0 - alpha_s)[:, None]
                    elif variant == "conf":
                        decoded_mask = (~mask_indices) & valid_block
                        eta_weights = torch.zeros_like(conf_scores)
                        if decoded_mask.any():
                            safe_conf = conf_scores.masked_fill(~decoded_mask, float("-inf"))
                            has_decoded = decoded_mask.any(dim=-1)
                            eta_weights[has_decoded] = safe_conf[has_decoded].softmax(dim=-1)
                        if conf_base_variant == "repo":
                            sigma_base = sigma_max_val
                        elif conf_base_variant == "cap":
                            sigma_base = torch.minimum(torch.full_like(alpha_t, eta), sigma_max_val)
                        elif conf_base_variant == "rescale":
                            sigma_base = eta * sigma_max_val
                        else:
                            raise ValueError(f"Unknown conf_base_variant: {conf_base_variant!r}")
                        sigma = eta_weights * sigma_base[:, None]
                    else:
                        raise ValueError(
                            f"Unknown ReMDM variant: {variant!r}. "
                            "Expected: cap / rescale / markovian / conf"
                        )

                    sigma_tokens = sigma if sigma.ndim == 2 else sigma.expand(B, block_size)
                    if sigma_tokens.shape[1] == 1:
                        sigma_tokens = sigma_tokens.expand(B, block_size)
                    sigma_tokens = torch.where(
                        valid_block, sigma_tokens, torch.zeros_like(sigma_tokens)
                    )

                    # ── ReMDM approximate posterior q(x_s | x_t, x_0) over the block ──
                    alpha_t_exp = alpha_t[:, None, None]
                    alpha_s_exp = alpha_s[:, None, None]
                    sigma_exp = sigma_tokens[:, :, None]

                    q_xs = p_x0 * (1.0 - sigma_exp)
                    q_xs[..., mask_id] = sigma_tokens

                    denom = (1.0 - alpha_t_exp).clamp(min=1e-6)
                    q_xs_masked = p_x0 * ((alpha_s_exp - (1.0 - sigma_exp) * alpha_t_exp) / denom)
                    q_xs_masked[..., mask_id] = (
                        (1.0 - alpha_s[:, None]) - sigma_tokens * alpha_t[:, None]
                    ) / denom.squeeze(-1)

                    q_xs = torch.where(mask_indices.unsqueeze(-1), q_xs_masked, q_xs)
                    q_xs = torch.where(
                        valid_block.unsqueeze(-1),
                        q_xs,
                        F.one_hot(block, num_classes=logits_block.shape[-1]).float(),
                    )
                    q_xs = q_xs.clamp(min=0.0)
                    q_xs = q_xs / q_xs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

                    xs = q_xs.argmax(dim=-1) if temperature <= 0.0 else _sample_categorical(q_xs)

                    if freeze_unmasked:
                        keep = (~mask_indices) & valid_block
                        xs = torch.where(keep, block, xs)
                    if frozen.any():
                        xs = torch.where(frozen[:, None].expand_as(xs), block, xs)

                    if variant == "conf":
                        unmask_mask = mask_indices & (xs != mask_id)
                        bi = torch.arange(B, device=xs.device)[:, None]
                        fi = torch.arange(block_size, device=xs.device)[None, :]
                        conf_values = -p_x0[bi, fi, xs]
                        conf_scores = torch.where(unmask_mask, conf_values, conf_scores)
                        remask_mask = ((block != mask_id) & valid_block) & (xs == mask_id)
                        conf_scores = torch.where(
                            remask_mask, torch.full_like(conf_scores, float("-inf")), conf_scores
                        )

                    new_block = torch.where(valid_block, xs, block)
                    x[:, blk_start:window_end] = new_block
                    if histories is not None:
                        histories.append(x.clone())

                if not ((x[:, blk_start:window_end] == mask_id) & valid_block).any():
                    break

            # ── Safety fill: commit any residual masks by argmax (bounded) ──
            for _ in range(block_size):
                block = x[:, blk_start:window_end]
                residual = (block == mask_id) & valid_block
                if not residual.any():
                    break
                logits_block = forward_fn().float()
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
