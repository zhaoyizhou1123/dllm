"""
Progressive masking pool for PUMA-style training.

Ported from ~/projects/mdm_correction/progressive.py (lines 1-537).

Provides:
  - mdm_edit_loss_fn_from_logits: memory-efficient edit loss over all response positions
  - PhasedMasking:     base pool with ground-truth token reveal
  - PhasedMaskingEdit: edit variant that reveals model-predicted tokens

References:
  PUMA: arXiv:2602.10314
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def mdm_edit_loss_fn_from_logits(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    mask_id: int,
    prompt_mask: torch.Tensor,
    arm_init: bool = False,
) -> torch.Tensor:
    """
    Memory-efficient edit loss over ALL response positions.

    Computes CE without materialising the full [B, L, V] log_probs tensor
    (which costs ~4 GB for LLaDA-8B).  Uses F.cross_entropy with ignore_index
    to skip prompt positions.

    Loss = (Σ_b  Σ_{l ∈ response} CE(logits[b,l], x0[b,l]) / L_eff[b]) / B

    Args:
        logits:      [B, L, V]  raw model output
        x0:          [B, L]     ground-truth token IDs
        xt:          [B, L]     current (partially masked) input — not used in
                                the loss itself, kept for API symmetry with the
                                log_probs variant
        mask_id:     int        mask token ID (unused here, kept for symmetry)
        prompt_mask: [B, L]     True = prompt token (excluded from loss)
        arm_init:    bool       if True, shift off the first token (for AR init)
    """
    B, L, V = logits.shape

    if arm_init:
        x0 = x0[:, 1:]
        logits = logits[:, :-1, :]
        prompt_mask = prompt_mask[:, 1:]
        L = L - 1

    L_eff = (L - prompt_mask.sum(dim=1)).float().clamp_min(1)  # [B]

    targets = x0.masked_fill(prompt_mask, -100)
    nll = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(B, L)  # [B, L]; prompt positions are 0 (ignored)

    per_seq_loss = nll.sum(dim=1)  # [B]
    return (per_seq_loss / L_eff).sum() / B


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------

def build_intervals(K: int) -> List[Tuple[float, float]]:
    """K ratio intervals spanning [0, 1]: [(0/K, 1/K), ..., ((K-1)/K, 1)]."""
    if K <= 0:
        return []
    return [(j / K, (j + 1) / K) for j in range(K)]


def phase_initialize(B: int, K: int, device: torch.device) -> torch.Tensor:
    """Initialise phase tensor so each phase 0..K-1 appears roughly equally."""
    base = torch.arange(K, device=device)
    repeats = math.ceil(B / K)
    phases = base.repeat(repeats)[:B]
    return phases[torch.randperm(B, device=device)]


def unmask_from_scores(
    scores: torch.Tensor,
    num_unmask: torch.Tensor,
    x0: torch.Tensor,
    xt_format: torch.Tensor,
) -> torch.Tensor:
    """
    Reveal the top-k highest-confidence masked positions.

    Args:
        scores:     [B, L]  confidence score per position (lower = masked)
        num_unmask: [B]     how many tokens to reveal per sequence
        x0:         [B, L]  source of the token values to insert
        xt_format:  [B, L]  current xt (cloned and updated in-place)
    """
    B = scores.shape[0]
    k_max = int(num_unmask.max().item())
    new_xt = xt_format.clone()
    if k_max > 0:
        _, topk_idx = scores.topk(k=k_max, dim=1, largest=True)  # [B, k_max]
        arange_k = torch.arange(k_max, device=scores.device).unsqueeze(0).expand(B, k_max)
        unmask_idx = arange_k < num_unmask.unsqueeze(1)  # [B, k_max]
        rows = torch.arange(B, device=scores.device).unsqueeze(1).expand(B, k_max)
        flat_r = rows[unmask_idx]
        flat_c = topk_idx[unmask_idx]
        new_xt[flat_r, flat_c] = x0[flat_r, flat_c]
    return new_xt


# ---------------------------------------------------------------------------
# PhasedMasking (base class)
# ---------------------------------------------------------------------------

class PhasedMasking:
    """
    Progressive unmasking pool.

    Maintains a batch of partially-unmasked sequences.  Each sequence cycles
    through K phases; at each phase a fraction of tokens is revealed based on
    model confidence.  When a sequence completes all K phases it is replaced
    with a fresh sample from the DataLoader.

    Batch dict format expected from the DataLoader::

        {"labels": LongTensor[L], "prompt_mask": BoolTensor[L]}

    Prompt tokens are fixed and never masked/revealed.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        batch_size: int,
        mask_id: int,
        K: int,
        device: torch.device,
        L: int,
        mode: str = "standard",
        confidence_threshold: Optional[float] = None,
    ):
        assert mode in ("standard", "confidence_collapse"), f"invalid mode: {mode!r}"
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.mask_id = mask_id
        self.K = K
        self.device = device
        self.L = L
        self.mode = mode
        self.confidence_threshold = confidence_threshold

        self.intervals = build_intervals(K)
        self.lower = torch.tensor([a for a, _ in self.intervals], device=device, dtype=torch.float)
        self.upper = torch.tensor([b for _, b in self.intervals], device=device, dtype=torch.float)

        B = batch_size
        self.state = dict(
            t=0,
            phase=phase_initialize(B, K, device),
            prompt_mask=torch.zeros(B, L, dtype=torch.bool, device=device),
            L_eff=torch.zeros(B, dtype=torch.long, device=device),
            eos_mask=torch.zeros(B, L, dtype=torch.bool, device=device),
            L_eos=torch.zeros(B, dtype=torch.long, device=device),
        )

        self._reset_iter()
        self._initialize_pool()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def current_batch(self) -> torch.Tensor:
        """Return the current partially-masked input [B, L]."""
        return self.xt

    def reset_loader_iter(self):
        """Reset the internal DataLoader iterator (call at epoch start)."""
        self._reset_iter()

    def update_k(self, new_k: int):
        """Update K and rebuild the interval tensors."""
        self.K = new_k
        self.intervals = build_intervals(new_k)
        self.lower = torch.tensor([a for a, _ in self.intervals], device=self.device, dtype=torch.float)
        self.upper = torch.tensor([b for _, b in self.intervals], device=self.device, dtype=torch.float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_iter(self):
        self._iter = iter(self.train_loader)

    def _sample_ratio(self, stages: torch.Tensor) -> torch.Tensor:
        lo = self.lower.index_select(0, stages)
        hi = self.upper.index_select(0, stages)
        return lo + torch.rand_like(lo) * (hi - lo)

    def _sample_target_unmasked(self, ratio: torch.Tensor, L_eff: torch.Tensor) -> torch.Tensor:
        num_unmask = torch.round(ratio * L_eff.float()).long()
        return torch.minimum(num_unmask, (L_eff - 1).clamp_min(1))

    def calculate_phase(self, xt: torch.Tensor) -> torch.Tensor:
        current_unmask = (~self.state["prompt_mask"] & (xt != self.mask_id)).sum(dim=1).long()
        ratio = current_unmask.float() / self.state["L_eff"].clamp_min(1).float()
        boundaries = self.upper[:-1]
        return torch.bucketize(ratio, boundaries).clamp_(0, self.K - 1).long()

    @torch.no_grad()
    def _get_new_seq(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw n sequences from the DataLoader.

        Handles DataLoader exhaustion by reshuffling and restarting.
        Expects batch dicts with keys ``"labels"`` and ``"prompt_mask"``.
        """
        out, masks = [], []
        while len(out) < n:
            try:
                batch = next(self._iter)
            except StopIteration:
                print("Warning: train loader exhausted — reshuffling and restarting.")
                self._reset_iter()
                batch = next(self._iter)
            labels = batch["labels"]
            prompt_mask = batch["prompt_mask"]
            if labels.ndim == 1:
                out.append(labels)
                masks.append(prompt_mask)
            else:
                for t, m in zip(labels, prompt_mask):
                    out.append(t)
                    masks.append(m)
                    if len(out) >= n:
                        break
        x0 = torch.stack(out[:n], dim=0).to(self.device)
        pm = torch.stack(masks[:n], dim=0).to(self.device)
        return x0, pm

    @torch.no_grad()
    def _refill_pool(self, n: int, stages: torch.Tensor):
        L, device = self.L, self.device
        new_x0, new_masks = self._get_new_seq(n)
        new_L_eff = (~new_masks).sum(dim=1).long()
        ratio = self._sample_ratio(stages)
        u0 = self._sample_target_unmasked(ratio, new_L_eff)

        new_xt = torch.full_like(new_x0, self.mask_id)
        new_xt = torch.where(new_masks, new_x0, new_xt)

        k_max = int(u0.max().item())
        if k_max > 0:
            rand_score = torch.rand(n, L, device=device, dtype=torch.float)
            rand_score = torch.where(new_masks, torch.finfo(rand_score.dtype).min, rand_score)
            new_xt = unmask_from_scores(rand_score, u0, new_x0, new_xt)

        return new_x0, new_xt, new_masks, new_L_eff

    @torch.no_grad()
    def _initialize_pool(self):
        B = self.batch_size
        phases = self.state["phase"]
        new_x0, new_xt, new_masks, new_L_eff = self._refill_pool(B, phases)
        self.x0 = new_x0
        self.xt = new_xt
        self.state["prompt_mask"] = new_masks
        self.state["L_eff"] = new_L_eff

    @torch.no_grad()
    def update_with_logits(self, log_probs: torch.Tensor):
        """
        Advance pool state using log-probabilities [B, L, V].

        Reveals top-k tokens per sequence according to model confidence, then
        refills completed sequences from the DataLoader.
        """
        B, L, V = log_probs.shape
        device = self.device

        phase_next = (self.state["phase"] + 1) % self.K
        replace = (phase_next == 0)

        mask_idx = (self.xt == self.mask_id)
        assert not (mask_idx & self.state["prompt_mask"]).any()

        ratio = self._sample_ratio(phase_next)
        num_unmask = self._sample_target_unmasked(ratio, self.state["L_eff"])
        current_num_unmask = (~mask_idx & ~self.state["prompt_mask"]).sum(dim=1).long()
        to_reveal = (num_unmask - current_num_unmask).clamp_min(0)
        to_reveal = torch.where(replace, torch.zeros_like(to_reveal), to_reveal)

        xt = self.xt
        k_max = int(to_reveal.max().item())
        if k_max > 0:
            score_conf = torch.where(mask_idx, log_probs.max(dim=2)[0],
                                     torch.finfo(log_probs.dtype).min * torch.ones_like(log_probs[:, :, 0]))
            xt = unmask_from_scores(score_conf, to_reveal, self.x0, self.xt)

        if self.mode == "confidence_collapse":
            tau = math.log(self.confidence_threshold)
            p = log_probs.max(dim=2)[0]
            update_unmask = (p > tau) & (xt == self.mask_id) & (~self.state["prompt_mask"])
            xt = torch.where(update_unmask, self.x0, xt)
            phase_next = self.calculate_phase(xt)

        self.xt = xt
        self.state["phase"] = phase_next

        n_new = int(replace.sum().item())
        if n_new > 0:
            idx = replace.nonzero(as_tuple=False).squeeze(1)
            stages = torch.zeros(n_new, device=device, dtype=torch.long)
            new_x0, new_xt, new_masks, new_L_eff = self._refill_pool(n_new, stages)
            self.x0[idx] = new_x0
            self.xt[idx] = new_xt
            self.state["prompt_mask"][idx] = new_masks
            self.state["L_eff"][idx] = new_L_eff
            self.state["phase"][idx] = 0

        self.state["t"] += 1


# ---------------------------------------------------------------------------
# PhasedMaskingEdit
# ---------------------------------------------------------------------------

class PhasedMaskingEdit(PhasedMasking):
    """
    Edit variant of PhasedMasking.

    Key difference from the base class: when revealing tokens, uses the
    *model's predicted tokens* (argmax of logits) rather than the ground-truth
    x0.  This implements the "edit" curriculum from the PUMA paper.

    Provides two update methods:

    * ``update_with_logits(log_probs)``  — accepts full log-prob tensor [B,L,V]
    * ``update_from_logits(logits)``     — memory-efficient; accepts raw logits
                                           [B,L,V], avoids storing [B,L,V] softmax
    """

    @torch.no_grad()
    def update_with_logits(self, log_probs: torch.Tensor):
        """Advance pool using log-probabilities [B, L, V]."""
        B, L, V = log_probs.shape
        device = self.device

        phase_next = (self.state["phase"] + 1) % self.K
        replace = (phase_next == 0)

        mask_idx = (self.xt == self.mask_id)
        assert not (mask_idx & self.state["prompt_mask"]).any()

        ratio = self._sample_ratio(phase_next)
        num_unmask = self._sample_target_unmasked(ratio, self.state["L_eff"])
        current_num_unmask = (~mask_idx & ~self.state["prompt_mask"]).sum(dim=1).long()
        to_reveal = (num_unmask - current_num_unmask).clamp_min(0)
        to_reveal = torch.where(replace, torch.zeros_like(to_reveal), to_reveal)

        xt = self.xt
        k_max = int(to_reveal.max().item())
        if k_max > 0:
            score_conf = torch.where(mask_idx, log_probs.max(dim=2)[0],
                                     torch.finfo(log_probs.dtype).min * torch.ones_like(log_probs[:, :, 0]))
            pred_token = log_probs.argmax(dim=2)
            xt = unmask_from_scores(score_conf, to_reveal, pred_token, self.xt)

        if self.mode == "confidence_collapse":
            tau = math.log(self.confidence_threshold)
            p = log_probs.max(dim=2)[0]
            update_unmask = (p > tau) & (xt == self.mask_id) & (~self.state["prompt_mask"])
            xt = torch.where(update_unmask, self.x0, xt)
            phase_next = self.calculate_phase(xt)

        self.xt = xt
        self.state["phase"] = phase_next

        n_new = int(replace.sum().item())
        if n_new > 0:
            idx = replace.nonzero(as_tuple=False).squeeze(1)
            stages = torch.zeros(n_new, device=device, dtype=torch.long)
            new_x0, new_xt, new_masks, new_L_eff = self._refill_pool(n_new, stages)
            self.x0[idx] = new_x0
            self.xt[idx] = new_xt
            self.state["prompt_mask"][idx] = new_masks
            self.state["L_eff"][idx] = new_L_eff
            self.state["phase"][idx] = 0

        self.state["t"] += 1

    @torch.no_grad()
    def update_from_logits(self, logits: torch.Tensor):
        """
        Memory-efficient update using raw logits [B, L, V].

        Avoids materialising full log_probs by computing:
            max_log_conf[b, l] = logits.max(dim=-1) - logsumexp(logits, dim=-1)
        """
        pred_tokens = logits.argmax(dim=-1)                                    # [B, L]
        max_log_conf = logits.max(dim=-1)[0] - torch.logsumexp(logits, dim=-1)  # [B, L]

        device = self.device
        phase_next = (self.state["phase"] + 1) % self.K
        replace = (phase_next == 0)

        mask_idx = (self.xt == self.mask_id)
        assert not (mask_idx & self.state["prompt_mask"]).any()

        ratio = self._sample_ratio(phase_next)
        num_unmask = self._sample_target_unmasked(ratio, self.state["L_eff"])
        current_num_unmask = (~mask_idx & ~self.state["prompt_mask"]).sum(dim=1).long()
        to_reveal = (num_unmask - current_num_unmask).clamp_min(0)
        to_reveal = torch.where(replace, torch.zeros_like(to_reveal), to_reveal)

        xt = self.xt
        k_max = int(to_reveal.max().item())
        if k_max > 0:
            score_conf = torch.where(
                mask_idx, max_log_conf,
                torch.full_like(max_log_conf, torch.finfo(max_log_conf.dtype).min),
            )
            xt = unmask_from_scores(score_conf, to_reveal, pred_tokens, self.xt)

        if self.mode == "confidence_collapse":
            tau = math.log(self.confidence_threshold)
            update_unmask = (max_log_conf > tau) & (xt == self.mask_id) & (~self.state["prompt_mask"])
            xt = torch.where(update_unmask, self.x0, xt)
            phase_next = self.calculate_phase(xt)

        self.xt = xt
        self.state["phase"] = phase_next

        n_new = int(replace.sum().item())
        if n_new > 0:
            idx = replace.nonzero(as_tuple=False).squeeze(1)
            stages = torch.zeros(n_new, device=device, dtype=torch.long)
            new_x0, new_xt, new_masks, new_L_eff = self._refill_pool(n_new, stages)
            self.x0[idx] = new_x0
            self.xt[idx] = new_xt
            self.state["prompt_mask"][idx] = new_masks
            self.state["L_eff"][idx] = new_L_eff
            self.state["phase"][idx] = 0

        self.state["t"] += 1
