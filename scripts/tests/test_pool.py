"""
Unit tests for dllm.core.trainers.pool.

Tests run on CPU with tiny synthetic batches — no GPU queue needed.
"""

import importlib.util
import math
import os
import pytest
import torch
from torch.utils.data import DataLoader

# Import pool.py directly to avoid triggering dllm's full package init
# (which pulls in pipeline deps like lm_eval and functorch).
_pool_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "dllm", "core", "trainers", "pool.py")
)
_spec = importlib.util.spec_from_file_location("pool", _pool_path)
_pool_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pool_mod)

PhasedMaskingEdit = _pool_mod.PhasedMaskingEdit
PhasedMasking = _pool_mod.PhasedMasking
build_intervals = _pool_mod.build_intervals
mdm_edit_loss_fn_from_logits = _pool_mod.mdm_edit_loss_fn_from_logits
phase_initialize = _pool_mod.phase_initialize
unmask_from_scores = _pool_mod.unmask_from_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MASK_ID = 999
B, L, V = 4, 16, 50
DEVICE = torch.device("cpu")


def _make_loader(B=B, L=L, num_batches=8):
    """Synthetic DataLoader with fixed-length sequences."""
    N = B * num_batches
    labels = torch.randint(0, 40, (N, L))  # keep vocab < V
    prompt_mask = torch.zeros(N, L, dtype=torch.bool)
    prompt_mask[:, : L // 4] = True  # first quarter is prompt

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return N
        def __getitem__(self, i):
            return {"labels": labels[i], "prompt_mask": prompt_mask[i]}

    return DataLoader(_Dataset(), batch_size=B, shuffle=False, drop_last=True)


def _make_pool(K=4, mode="standard", loader=None):
    if loader is None:
        loader = _make_loader()
    return PhasedMaskingEdit(
        train_loader=loader,
        batch_size=B,
        mask_id=MASK_ID,
        K=K,
        device=DEVICE,
        L=L,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# build_intervals
# ---------------------------------------------------------------------------

def test_build_intervals_basic():
    ivs = build_intervals(4)
    assert len(ivs) == 4
    assert ivs[0] == (0.0, 0.25)
    assert abs(ivs[-1][1] - 1.0) < 1e-6


def test_build_intervals_empty():
    assert build_intervals(0) == []


def test_build_intervals_covers_unit():
    K = 7
    ivs = build_intervals(K)
    assert abs(ivs[0][0]) < 1e-9
    assert abs(ivs[-1][1] - 1.0) < 1e-9
    for i in range(len(ivs) - 1):
        assert abs(ivs[i][1] - ivs[i + 1][0]) < 1e-9


# ---------------------------------------------------------------------------
# phase_initialize
# ---------------------------------------------------------------------------

def test_phase_initialize_range():
    K = 5
    phases = phase_initialize(20, K, DEVICE)
    assert phases.shape == (20,)
    assert phases.min().item() >= 0
    assert phases.max().item() <= K - 1


def test_phase_initialize_roughly_uniform():
    K = 4
    N = 400
    phases = phase_initialize(N, K, DEVICE)
    counts = torch.bincount(phases, minlength=K)
    # Each bucket should have roughly N/K samples; allow ±25% tolerance
    expected = N / K
    for c in counts:
        assert abs(c.item() - expected) / expected < 0.25


# ---------------------------------------------------------------------------
# unmask_from_scores
# ---------------------------------------------------------------------------

def test_unmask_from_scores_reveals_correct_count():
    scores = torch.rand(B, L)
    num_unmask = torch.tensor([2, 3, 1, 4])
    x0 = torch.randint(0, 40, (B, L))
    xt = torch.full((B, L), MASK_ID)

    new_xt = unmask_from_scores(scores, num_unmask, x0, xt)

    for i in range(B):
        revealed = (new_xt[i] != MASK_ID).sum().item()
        assert revealed == num_unmask[i].item(), f"row {i}: {revealed} != {num_unmask[i]}"


def test_unmask_from_scores_zero_reveals_nothing():
    scores = torch.rand(B, L)
    num_unmask = torch.zeros(B, dtype=torch.long)
    x0 = torch.randint(0, 40, (B, L))
    xt = torch.full((B, L), MASK_ID)
    new_xt = unmask_from_scores(scores, num_unmask, x0, xt)
    assert (new_xt == MASK_ID).all()


# ---------------------------------------------------------------------------
# mdm_edit_loss_fn_from_logits
# ---------------------------------------------------------------------------

def test_loss_fn_shape():
    logits = torch.randn(B, L, V)
    x0 = torch.randint(0, V, (B, L))
    xt = torch.full((B, L), MASK_ID)
    prompt_mask = torch.zeros(B, L, dtype=torch.bool)
    prompt_mask[:, :4] = True
    loss = mdm_edit_loss_fn_from_logits(logits, x0, xt, MASK_ID, prompt_mask)
    assert loss.shape == ()  # scalar
    assert loss.item() > 0


def test_loss_fn_prompt_excluded():
    """With full prompt_mask the loss should be 0 (no response tokens)."""
    logits = torch.randn(B, L, V)
    x0 = torch.randint(0, V, (B, L))
    xt = torch.full((B, L), MASK_ID)
    # Mark all positions as prompt
    prompt_mask = torch.ones(B, L, dtype=torch.bool)
    loss = mdm_edit_loss_fn_from_logits(logits, x0, xt, MASK_ID, prompt_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_fn_perfect_prediction_is_zero():
    """When logits place all mass on correct token, CE should be near 0."""
    x0 = torch.randint(0, V, (B, L))
    logits = torch.full((B, L, V), -1e9)
    for b in range(B):
        for l in range(L):
            logits[b, l, x0[b, l]] = 1e9
    xt = torch.full((B, L), MASK_ID)
    prompt_mask = torch.zeros(B, L, dtype=torch.bool)
    loss = mdm_edit_loss_fn_from_logits(logits, x0, xt, MASK_ID, prompt_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_loss_fn_arm_init():
    """arm_init=True should shift off first token and not crash."""
    logits = torch.randn(B, L, V)
    x0 = torch.randint(0, V, (B, L))
    xt = torch.full((B, L), MASK_ID)
    prompt_mask = torch.zeros(B, L, dtype=torch.bool)
    loss = mdm_edit_loss_fn_from_logits(logits, x0, xt, MASK_ID, prompt_mask, arm_init=True)
    assert loss.shape == ()
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# PhasedMaskingEdit pool state transitions
# ---------------------------------------------------------------------------

def test_pool_initialization():
    pool = _make_pool(K=4)
    assert pool.xt.shape == (B, L)
    assert pool.x0.shape == (B, L)
    assert pool.state["prompt_mask"].shape == (B, L)
    assert pool.state["L_eff"].shape == (B,)
    # Prompt tokens should never be masked
    assert not (pool.xt[pool.state["prompt_mask"]] == MASK_ID).any()


def test_pool_prompt_tokens_unchanged():
    """Prompt tokens should always equal x0 (never masked)."""
    pool = _make_pool(K=4)
    pm = pool.state["prompt_mask"]
    assert (pool.xt[pm] == pool.x0[pm]).all()


def test_pool_update_from_logits_advances_state():
    pool = _make_pool(K=4)
    xt_before = pool.xt.clone()
    logits = torch.randn(B, L, V)
    pool.update_from_logits(logits)
    # t should have incremented
    assert pool.state["t"] == 1
    # At least some positions might have changed (stochastic, but highly likely)
    # We just check shape and prompt invariant
    pm = pool.state["prompt_mask"]
    assert (pool.xt[pm] == pool.x0[pm]).all()


def test_pool_update_from_logits_never_masks_prompt():
    pool = _make_pool(K=4)
    logits = torch.randn(B, L, V)
    for _ in range(10):
        pool.update_from_logits(logits)
        pm = pool.state["prompt_mask"]
        assert not (pool.xt[pm] == MASK_ID).any(), "Prompt tokens should never be masked"


def test_pool_refills_on_phase_completion():
    """After K steps each sequence should have been refilled at least once (probabilistic)."""
    K = 2
    pool = _make_pool(K=K)
    logits = torch.randn(B, L, V)
    # Run K+2 steps to force at least one refill
    for _ in range(K + 2):
        pool.update_from_logits(logits)
    # Pool should still have valid shapes
    assert pool.xt.shape == (B, L)


def test_pool_update_k():
    pool = _make_pool(K=4)
    pool.update_k(8)
    assert pool.K == 8
    assert len(pool.intervals) == 8
    # After update_k, intervals should still cover [0,1]
    assert abs(pool.intervals[0][0]) < 1e-9
    assert abs(pool.intervals[-1][1] - 1.0) < 1e-9


def test_pool_update_k_correctness():
    """update_k(new_k) should rebuild lower/upper tensors to match new K."""
    pool = _make_pool(K=4)
    pool.update_k(6)
    expected_lower = torch.tensor([i / 6 for i in range(6)])
    expected_upper = torch.tensor([(i + 1) / 6 for i in range(6)])
    assert torch.allclose(pool.lower, expected_lower, atol=1e-6)
    assert torch.allclose(pool.upper, expected_upper, atol=1e-6)


# ---------------------------------------------------------------------------
# Data format compatibility
# ---------------------------------------------------------------------------

def test_dataset_batch_format():
    """Verify the synthetic DataLoader produces the expected batch format."""
    loader = _make_loader(B=2, L=L, num_batches=2)
    batch = next(iter(loader))
    assert "labels" in batch
    assert "prompt_mask" in batch
    assert batch["labels"].shape == (2, L)
    assert batch["prompt_mask"].shape == (2, L)
    assert batch["labels"].dtype == torch.int64
    assert batch["prompt_mask"].dtype == torch.bool


def test_pool_with_single_item_batches():
    """Pool should work when DataLoader yields B=1 sequences."""
    loader = _make_loader(B=1, L=L, num_batches=16)
    pool = PhasedMaskingEdit(
        train_loader=loader,
        batch_size=1,
        mask_id=MASK_ID,
        K=4,
        device=DEVICE,
        L=L,
    )
    logits = torch.randn(1, L, V)
    pool.update_from_logits(logits)
    assert pool.xt.shape == (1, L)
