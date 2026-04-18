"""
Unit tests for ProSeCo loss logic.

Verifies:
  - randomly_mask output shapes and invariants
  - correction_input construction (prompt tokens fixed, response from argmax)
  - num_mask weighting consistency
  - mdm_loss + correction_loss shapes match expected

All tests run on CPU — no GPU needed.
"""

import math
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inline the helpers from sft_proseco.py so tests don't require accelerate
# ---------------------------------------------------------------------------

MASK_ID = 126336


def randomly_mask(
    x0: torch.Tensor,
    prompt_mask: torch.Tensor,
    mask_id: int,
    min_t: float,
):
    """Randomly mask response tokens (replica of sft_proseco.randomly_mask)."""
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


def compute_mdm_loss(logits, x0, mask_idx, num_mask_f, B):
    ce = F.cross_entropy(logits.transpose(1, 2), x0, reduction="none")
    return (ce * mask_idx.float() / num_mask_f.clamp_min(1)).sum() / B


def compute_correction_loss(corr_logits, x0, prompt_mask, num_mask_f, B):
    response_mask = ~prompt_mask
    ce = F.cross_entropy(corr_logits.transpose(1, 2), x0, reduction="none")
    return (ce * response_mask.float() / num_mask_f.clamp_min(1)).sum() / B


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

B, L, V = 4, 32, 100


def _make_batch(prompt_frac=0.25):
    x0 = torch.randint(0, V, (B, L))
    prompt_mask = torch.zeros(B, L, dtype=torch.bool)
    p_len = int(L * prompt_frac)
    prompt_mask[:, :p_len] = True
    return x0, prompt_mask


# ---------------------------------------------------------------------------
# randomly_mask
# ---------------------------------------------------------------------------

def test_randomly_mask_output_shapes():
    x0, pm = _make_batch()
    noised, mask_idx, num_mask_f = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    assert noised.shape == (B, L)
    assert mask_idx.shape == (B, L)
    assert num_mask_f.shape == (B, L)


def test_randomly_mask_prompt_tokens_unchanged():
    x0, pm = _make_batch()
    noised, mask_idx, _ = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    # Prompt positions must never be masked
    assert not mask_idx[pm].any(), "Prompt tokens should never be in mask_idx"
    assert (noised[pm] == x0[pm]).all(), "Prompt token values must be unchanged"


def test_randomly_mask_masked_positions_have_mask_id():
    x0, pm = _make_batch()
    noised, mask_idx, _ = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    assert (noised[mask_idx] == MASK_ID).all()


def test_randomly_mask_non_masked_response_unchanged():
    x0, pm = _make_batch()
    noised, mask_idx, _ = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    # Non-masked, non-prompt positions should keep original values
    non_masked_response = (~mask_idx) & (~pm)
    assert (noised[non_masked_response] == x0[non_masked_response]).all()


def test_randomly_mask_at_least_one_per_seq():
    x0, pm = _make_batch()
    _, mask_idx, _ = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    counts = mask_idx.sum(dim=1)
    assert (counts >= 1).all(), "Each sequence must have at least 1 masked token"


def test_randomly_mask_count_within_bounds():
    x0, pm = _make_batch()
    L_eff = (~pm).sum(dim=1).float()
    for _ in range(10):  # run several times due to randomness
        _, mask_idx, _ = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
        counts = mask_idx.sum(dim=1).float()
        # num_mask = ceil(frac * L_eff) with frac in [0.1, 1.0]
        # so counts must satisfy: ceil(0.1 * L_eff) <= counts <= L_eff
        assert (counts <= L_eff).all()
        assert (counts >= torch.ceil(0.1 * L_eff)).all()


def test_randomly_mask_min_t_equals_one_masks_all_response():
    x0, pm = _make_batch()
    _, mask_idx, _ = randomly_mask(x0, pm, MASK_ID, min_t=1.0)
    L_eff = (~pm).sum(dim=1)
    counts = mask_idx.sum(dim=1)
    assert (counts == L_eff).all(), "min_t=1 should mask all response tokens"


def test_randomly_mask_num_mask_f_broadcast():
    """num_mask_f should have the same value for all positions in a sequence."""
    x0, pm = _make_batch()
    _, _, num_mask_f = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    # All values in each row should be the same (broadcast from [B,1])
    for b in range(B):
        assert num_mask_f[b].unique().numel() == 1


# ---------------------------------------------------------------------------
# correction_input construction
# ---------------------------------------------------------------------------

def test_correction_input_prompt_uses_x0():
    """For prompt positions, correction_input must equal x0."""
    x0, pm = _make_batch()
    logits = torch.randn(B, L, V)
    pred = logits.detach().argmax(dim=-1)
    correction_input = torch.where(pm, x0, pred)
    assert (correction_input[pm] == x0[pm]).all()


def test_correction_input_response_uses_argmax():
    """For response positions, correction_input must equal argmax of logits."""
    x0, pm = _make_batch()
    logits = torch.randn(B, L, V)
    pred = logits.detach().argmax(dim=-1)
    correction_input = torch.where(pm, x0, pred)
    response_mask = ~pm
    assert (correction_input[response_mask] == pred[response_mask]).all()


def test_correction_input_no_gradient():
    """argmax on detached logits must produce no gradient."""
    x0, pm = _make_batch()
    logits = torch.randn(B, L, V, requires_grad=True)
    pred = logits.detach().argmax(dim=-1)
    # pred is a plain integer tensor, no grad_fn
    assert not pred.requires_grad


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def test_mdm_loss_scalar():
    x0, pm = _make_batch()
    noised, mask_idx, num_mask_f = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    logits = torch.randn(B, L, V)
    loss = compute_mdm_loss(logits, x0, mask_idx, num_mask_f, B)
    assert loss.shape == ()
    assert loss.item() > 0


def test_correction_loss_scalar():
    x0, pm = _make_batch()
    _, _, num_mask_f = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    corr_logits = torch.randn(B, L, V)
    loss = compute_correction_loss(corr_logits, x0, pm, num_mask_f, B)
    assert loss.shape == ()
    assert loss.item() > 0


def test_losses_nonnegative():
    x0, pm = _make_batch()
    noised, mask_idx, num_mask_f = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    logits = torch.randn(B, L, V)
    corr_logits = torch.randn(B, L, V)
    mdm_loss = compute_mdm_loss(logits, x0, mask_idx, num_mask_f, B)
    corr_loss = compute_correction_loss(corr_logits, x0, pm, num_mask_f, B)
    assert mdm_loss.item() >= 0
    assert corr_loss.item() >= 0


def test_total_loss_is_sum():
    """total_loss = mdm_loss + proseco_weight * correction_loss."""
    proseco_weight = 0.5
    x0, pm = _make_batch()
    noised, mask_idx, num_mask_f = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    logits = torch.randn(B, L, V)
    corr_logits = torch.randn(B, L, V)
    mdm = compute_mdm_loss(logits, x0, mask_idx, num_mask_f, B)
    corr = compute_correction_loss(corr_logits, x0, pm, num_mask_f, B)
    total = mdm + proseco_weight * corr
    expected = mdm.item() + proseco_weight * corr.item()
    assert abs(total.item() - expected) < 1e-6


def test_mdm_loss_zero_when_perfect():
    """MDM loss should be near 0 when logits perfectly predict masked tokens."""
    x0, pm = _make_batch()
    noised, mask_idx, num_mask_f = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    logits = torch.full((B, L, V), -1e9)
    for b in range(B):
        for l in range(L):
            logits[b, l, x0[b, l]] = 1e9
    loss = compute_mdm_loss(logits, x0, mask_idx, num_mask_f, B)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_correction_loss_only_on_response():
    """Correction loss is zero if response is empty (all prompt)."""
    x0 = torch.randint(0, V, (B, L))
    pm = torch.ones(B, L, dtype=torch.bool)  # all prompt
    _, _, num_mask_f = randomly_mask(x0, ~pm, MASK_ID, min_t=0.1)  # trivial
    corr_logits = torch.randn(B, L, V)
    # All positions are prompt → response_mask = all False
    ce = F.cross_entropy(corr_logits.transpose(1, 2), x0, reduction="none")
    response_mask = ~pm
    loss = (ce * response_mask.float() / num_mask_f.clamp_min(1)).sum() / B
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_mdm_backward():
    x0, pm = _make_batch()
    noised, mask_idx, num_mask_f = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    logits = torch.randn(B, L, V, requires_grad=True)
    loss = compute_mdm_loss(logits, x0, mask_idx, num_mask_f, B)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == (B, L, V)


def test_correction_backward():
    x0, pm = _make_batch()
    _, _, num_mask_f = randomly_mask(x0, pm, MASK_ID, min_t=0.1)
    corr_logits = torch.randn(B, L, V, requires_grad=True)
    loss = compute_correction_loss(corr_logits, x0, pm, num_mask_f, B)
    loss.backward()
    assert corr_logits.grad is not None
    assert corr_logits.grad.shape == (B, L, V)


def test_detached_logits_no_grad_in_correction_input():
    """correction_input is built from logits.detach(), so gradients don't
    flow from correction_loss back through pass-1 logits."""
    x0, pm = _make_batch()
    logits = torch.randn(B, L, V, requires_grad=True)
    pred = logits.detach().argmax(dim=-1)
    correction_input = torch.where(pm, x0, pred)
    # correction_input has no grad_fn
    assert not correction_input.requires_grad
