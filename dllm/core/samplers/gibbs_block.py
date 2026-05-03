"""
Fixed-schedule block diffusion sampler with Gibbs correction for LLaDA 1.x.

Adapted from ``dllm.pipelines.llada21.gibbs_block_sampler`` for models that
use bidirectional full-sequence attention (no ``position_ids``, 2-D
``attention_mask``).  The core algorithm is identical: within each block,
unmask ``unmasking_num`` tokens per step (fixed schedule), and optionally
apply Gibbs correction every ``edit_freq`` steps.

Run: used via the lm-eval harness, e.g.::

    python dllm/pipelines/llada/eval.py \\
        --tasks humaneval_instruct_llada --num_fewshot 0 \\
        --model llada_gibbs_block --apply_chat_template \\
        --batch_size 1 \\
        --model_args "pretrained=kuleshov-group/proseco-llada-sft,\\
max_new_tokens=256,block_size=32,unmasking_num=1,temperature=0.0,\\
cfg_scale=0.0,suppress_tokens=[126081],begin_suppress_tokens=[],\\
edit_freq=1,edit_step=10,edit_strategy=gibbs_edit,remasking_strategy=random" \\
        --confirm_run_unsafe_code
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.pipelines.llada2.sampler import sample_tokens


# ────────────────────────── helpers ──────────────────────────

def block_forward(model, x, blk_start, block_size, attention_mask,
                  cfg_scale, unmasked_index, suppress_tokens, mask_id):
    """Full-sequence forward for LLaDA 1.x, returning logits for the current block."""
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[unmasked_index] = mask_id
        x_ = torch.cat([x, un_x], dim=0)
        logits = model(
            x_, attention_mask=attention_mask.repeat(2, 1),
        ).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        logits = model(x, attention_mask=attention_mask).logits
    logits_block = logits[:, blk_start:blk_start + block_size, :]
    if suppress_tokens is not None and len(suppress_tokens) > 0:
        for token_id in suppress_tokens:
            logits_block[:, :, token_id] = -torch.inf
    return logits_block


# ── Gibbs correction routines (model-agnostic, operate on block slices) ──
# These are identical to the ones in llada21/gibbs_block_sampler.py.

def compute_remask(B, block_size, num_masks, k_max, prompt_mask_in_block,
                   logits_block, remasking_strategy, device):
    non_prompt = ~prompt_mask_in_block.unsqueeze(0).expand(B, -1)
    if remasking_strategy == "low_confidence":
        conf = F.softmax(logits_block, dim=-1).max(dim=-1).values
        remask_scores = torch.where(
            non_prompt, -conf,
            torch.full((B, block_size), -float("inf"), device=device),
        )
    else:
        remask_scores = torch.where(
            non_prompt,
            torch.rand(B, block_size, device=device),
            torch.full((B, block_size), -float("inf"), device=device),
        )
    _, top_indices = torch.topk(remask_scores, k=k_max, dim=-1)
    valid = torch.arange(k_max, device=device).unsqueeze(0) < num_masks.unsqueeze(1)
    remask = torch.zeros(B, block_size, dtype=torch.bool, device=device)
    remask.scatter_(1, top_indices, valid)
    return remask


@torch.no_grad()
def gibbs_standard(x, block_start, window_end, logits_block,
                   prompt_mask_in_block, mask_id, edit_step,
                   remasking_strategy, keep_original_mask,
                   early_exit_number, forward_fn):
    B = x.shape[0]
    block_size = window_end - block_start
    device = x.device

    block_slice = x[:, block_start:window_end]
    original_mask = block_slice == mask_id
    num_masks = original_mask.sum(dim=-1)

    new_pred = torch.argmax(logits_block, dim=-1)
    yt = torch.where(original_mask, new_pred, block_slice)
    prev_yt = None
    consecutive_count = 0

    for _ in range(edit_step):
        k_max = int(num_masks.max().item())
        if k_max > 0:
            if remasking_strategy == "low_confidence":
                x[:, block_start:window_end] = yt
                logits_block = forward_fn()
                new_pred = torch.argmax(logits_block, dim=-1)
                yt = torch.where(yt == mask_id, new_pred, yt)
            remask = compute_remask(
                B, block_size, num_masks, k_max, prompt_mask_in_block,
                logits_block, remasking_strategy, device,
            )
            yt = yt.clone()
            yt[remask] = mask_id

        x[:, block_start:window_end] = yt
        logits_block = forward_fn()
        new_pred = torch.argmax(logits_block, dim=-1)
        yt = torch.where(yt == mask_id, new_pred, yt)

        if early_exit_number > 0:
            if prev_yt is not None and torch.equal(yt, prev_yt):
                consecutive_count += 1
                if consecutive_count >= early_exit_number:
                    break
            else:
                consecutive_count = 0
            prev_yt = yt.clone()

    x[:, block_start:window_end] = yt
    logits_block = forward_fn()
    new_pred = torch.argmax(logits_block, dim=-1)
    yt = torch.where(yt == mask_id, new_pred, yt)

    if keep_original_mask:
        if original_mask.any():
            yt = yt.clone()
            yt[original_mask] = mask_id
    else:
        k_max = int(num_masks.max().item())
        if k_max > 0:
            remask = compute_remask(
                B, block_size, num_masks, k_max, prompt_mask_in_block,
                logits_block, remasking_strategy, device,
            )
            yt = yt.clone()
            yt[remask] = mask_id

    x[:, block_start:window_end] = yt
    logits_block = forward_fn()
    return logits_block


@torch.no_grad()
def gibbs_edit_v1(x, block_start, window_end, logits_block,
                  prompt_mask_in_block, mask_id, edit_step,
                  remasking_strategy, keep_original_mask,
                  early_exit_number, forward_fn):
    B = x.shape[0]
    block_size = window_end - block_start
    device = x.device

    orig_block = x[:, block_start:window_end].clone()
    unmask_indices = orig_block != mask_id
    num_masks = (orig_block == mask_id).sum(dim=-1)
    non_prompt = ~prompt_mask_in_block.unsqueeze(0).expand(B, -1)

    new_pred = torch.argmax(logits_block, dim=-1)
    yt = torch.where(non_prompt, new_pred, orig_block)
    prev_yt = None
    consecutive_count = 0

    for _ in range(edit_step):
        k_max = int(num_masks.max().item())
        if k_max > 0:
            remask = compute_remask(
                B, block_size, num_masks, k_max, prompt_mask_in_block,
                logits_block, remasking_strategy, device,
            )
            yt = yt.clone()
            yt[remask] = mask_id

        x[:, block_start:window_end] = yt
        logits_block = forward_fn()
        new_pred = torch.argmax(logits_block, dim=-1)
        yt = torch.where(non_prompt, new_pred, orig_block)

        if early_exit_number > 0:
            if prev_yt is not None and torch.equal(yt, prev_yt):
                consecutive_count += 1
                if consecutive_count >= early_exit_number:
                    break
            else:
                consecutive_count = 0
            prev_yt = yt.clone()

    if keep_original_mask:
        xt_out = torch.where(unmask_indices, yt, orig_block)
    else:
        k_max = int(num_masks.max().item())
        if k_max > 0:
            remask = compute_remask(
                B, block_size, num_masks, k_max, prompt_mask_in_block,
                logits_block, remasking_strategy, device,
            )
            yt = yt.clone()
            yt[remask] = mask_id
        xt_out = yt

    x[:, block_start:window_end] = xt_out
    logits_block = forward_fn()
    return logits_block


@torch.no_grad()
def gibbs_edit_v2(x, block_start, window_end, logits_block,
                  prompt_mask_in_block, mask_id, edit_step,
                  remasking_strategy, early_exit_number, forward_fn):
    B = x.shape[0]
    block_size = window_end - block_start
    device = x.device

    orig_block = x[:, block_start:window_end].clone()
    num_masks = (orig_block == mask_id).sum(dim=-1)
    non_prompt = ~prompt_mask_in_block.unsqueeze(0).expand(B, -1)

    new_pred = torch.argmax(logits_block, dim=-1)
    yt = torch.where(non_prompt, new_pred, orig_block)
    prev_yt = None
    consecutive_count = 0

    for _ in range(edit_step):
        k_max = int(num_masks.max().item())
        if k_max > 0:
            remask = compute_remask(
                B, block_size, num_masks, k_max, prompt_mask_in_block,
                logits_block, remasking_strategy, device,
            )
            yt = yt.clone()
            yt[remask] = mask_id

        x[:, block_start:window_end] = yt
        logits_block = forward_fn()
        new_pred = torch.argmax(logits_block, dim=-1)
        yt = torch.where(non_prompt, new_pred, orig_block)

        if early_exit_number > 0:
            if prev_yt is not None and torch.equal(yt, prev_yt):
                consecutive_count += 1
                if consecutive_count >= early_exit_number:
                    break
            else:
                consecutive_count = 0
            prev_yt = yt.clone()

    new_pred = torch.argmax(logits_block, dim=-1)
    yt = torch.where(non_prompt, new_pred, orig_block)
    k_max = int(num_masks.max().item())
    if k_max > 0:
        remask = compute_remask(
            B, block_size, num_masks, k_max, prompt_mask_in_block,
            logits_block, remasking_strategy, device,
        )
        yt = yt.clone()
        yt[remask] = mask_id

    x[:, block_start:window_end] = yt
    logits_block = forward_fn()
    return logits_block


@torch.no_grad()
def proseco_correct(x, active_start, block_start, window_end,
                    logits_block, mask_id, edit_step, active_forward_fn):
    """Proseco-style mask-free fixed-point correction over all generated tokens.

    Corrects across all past blocks by iterating over the full active region
    (``active_start:window_end``). Only committed (originally non-masked)
    positions are updated; original masks are restored.

    Args:
        x: Full sequence tensor (B, T), modified in-place.
        active_start: Start of the generated region (prompt boundary).
        block_start: Start of the current block.
        window_end: End of the current block.
        logits_block: Initial logits for the current block (B, block_size, V).
        mask_id: The mask token id.
        edit_step: Maximum number of fixed-point iterations.
        active_forward_fn: Returns logits for ``active_start:window_end``.
    """
    B = x.shape[0]
    block_offset = block_start - active_start

    orig_active = x[:, active_start:window_end].clone()
    original_mask = (orig_active == mask_id)

    # Initial fill: only current block has masks; fill with argmax
    new_pred = torch.argmax(logits_block, dim=-1)
    yt = orig_active.clone()
    block_mask = original_mask[:, block_offset:]
    yt[:, block_offset:][block_mask] = new_pred[block_mask]
    x[:, active_start:window_end] = yt

    # Fixed-point iteration over full active region
    logits_active = None
    for _ in range(edit_step):
        logits_active = active_forward_fn()
        corrected = torch.argmax(logits_active, dim=-1)
        if torch.equal(corrected, yt):
            break
        yt = corrected
        x[:, active_start:window_end] = yt

    # Merge: restore original masks, update committed positions
    final = orig_active.clone()
    committed = ~original_mask
    final[committed] = yt[committed]
    x[:, active_start:window_end] = final

    if logits_active is not None:
        return logits_active[:, block_offset:, :]
    return logits_block


def gibbs_correct(x, block_start, window_end, logits_block,
                  prompt_mask_in_block, mask_id, edit_step,
                  edit_strategy, remasking_strategy,
                  keep_original_mask, early_exit_number, forward_fn):
    if edit_strategy == "gibbs_standard":
        return gibbs_standard(
            x, block_start, window_end, logits_block,
            prompt_mask_in_block, mask_id, edit_step,
            remasking_strategy, keep_original_mask,
            early_exit_number, forward_fn,
        )
    if edit_strategy == "gibbs_edit":
        return gibbs_edit_v1(
            x, block_start, window_end, logits_block,
            prompt_mask_in_block, mask_id, edit_step,
            remasking_strategy, keep_original_mask,
            early_exit_number, forward_fn,
        )
    if edit_strategy == "gibbs_edit_v2":
        return gibbs_edit_v2(
            x, block_start, window_end, logits_block,
            prompt_mask_in_block, mask_id, edit_step,
            remasking_strategy, early_exit_number, forward_fn,
        )
    raise ValueError(f"Unknown edit_strategy: {edit_strategy}")


# ────────────────────────── config + sampler ──────────────────────────

@dataclass
class GibbsBlockSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: Optional[int] = None
    block_size: int = 32
    unmasking_num: int = 1
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    remasking: str = "low_confidence"
    eos_early_stop: bool = False
    cfg_scale: float = 0.0
    suppress_tokens: Optional[list[int]] = None
    begin_suppress_tokens: Optional[list[int]] = None
    edit_freq: int = -1
    edit_step: int = 0
    edit_start: int = 0
    edit_strategy: str = "gibbs_standard"
    remasking_strategy: str = "random"
    keep_original_mask: bool = True
    early_exit_number: int = 5
    skip_blocks: int = 0


@dataclass
class GibbsBlockSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: GibbsBlockSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = GibbsBlockSamplerConfig()

        block_size = kwargs.get("block_size", config.block_size)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        unmasking_num = int(kwargs.get("unmasking_num", config.unmasking_num))
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        remasking = kwargs.get("remasking", config.remasking)
        eos_early_stop = kwargs.get("eos_early_stop", config.eos_early_stop)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
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

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        num_blocks = (max_new_tokens + block_size - 1) // block_size

        # ── Canvas: EOS-padded, with prompt + mask tail ──
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, :prompt_lens[i]] = p
            x[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens] = mask_id

        # 2-D attention mask (B, T)
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        if cfg_scale > 0.0:
            unmasked_index = (x != mask_id) & attention_mask.bool()
        else:
            unmasked_index = None

        histories = [x.clone()] if return_dict else None
        global_step = 0

        for blk in range(num_blocks):
            # Block boundaries relative to the longest prompt
            max_prompt = max(prompt_lens)
            blk_start = max_prompt + blk * block_size
            window_end = min(blk_start + block_size, T)
            actual_block_size = window_end - blk_start

            prompt_mask_in_block = torch.zeros(
                actual_block_size, dtype=torch.bool, device=self.model.device
            )
            # If prompt extends into this block, mark those positions
            for i in range(B):
                if prompt_lens[i] > blk_start:
                    end = min(prompt_lens[i] - blk_start, actual_block_size)
                    prompt_mask_in_block[:end] = True

            def forward_fn(_bs=blk_start, _abs=actual_block_size):
                if cfg_scale > 0.0:
                    unmasked_index[:, _bs:_bs + _abs] = (
                        x[:, _bs:_bs + _abs] != mask_id
                    )
                return block_forward(
                    self.model, x, _bs, _abs, attention_mask,
                    cfg_scale, unmasked_index, suppress_tokens, mask_id,
                )

            for gen_step in range(actual_block_size):
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
                    and blk >= skip_blocks
                ):
                    if edit_strategy == "proseco_eval":
                        _as = max_prompt
                        _asize = window_end - _as

                        def active_fwd(_as=_as, _asize=_asize):
                            if cfg_scale > 0.0:
                                unmasked_index[:, _as:_as + _asize] = (
                                    x[:, _as:_as + _asize] != mask_id
                                )
                            return block_forward(
                                self.model, x, _as, _asize, attention_mask,
                                cfg_scale, unmasked_index, suppress_tokens,
                                mask_id,
                            )

                        logits_block = proseco_correct(
                            x, max_prompt, blk_start, window_end,
                            logits_block, mask_id, edit_step, active_fwd,
                        )
                    else:
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

                if remasking == "low_confidence":
                    confidence = probs
                elif remasking == "random":
                    confidence = torch.rand_like(probs)
                else:
                    raise NotImplementedError(f"Unknown remasking: {remasking}")

                transfer_index = torch.zeros_like(active_mask, dtype=torch.bool)
                for b in range(B):
                    conf = torch.where(
                        active_mask[b], confidence[b],
                        torch.full_like(confidence[b], -float("inf")),
                    )
                    num_available = active_mask[b].sum().item()
                    k = min(unmasking_num, num_available)
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
                # Check if all samples have completed (no masks + has EOS)
                generated_part = x[:, max_prompt:window_end]
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
