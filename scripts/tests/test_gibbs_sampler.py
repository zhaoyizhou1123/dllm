"""
Unit tests for GibbsSampler: parity with MDLMSampler when disabled, and smoke
coverage of the three edit_strategy variants.

Run from repo root:
  pytest scripts/tests/test_gibbs_sampler.py -v
"""

from types import SimpleNamespace

import pytest
import torch

from dllm.core.samplers import (
    GibbsSampler,
    GibbsSamplerConfig,
    MDLMSampler,
    MDLMSamplerConfig,
)


VOCAB = 17
MASK_ID = 2
BOS_ID = 3
EOS_ID = 4


class _FixedLogitsModel(torch.nn.Module):
    """Tiny model: logits deterministically depend on input_ids + a seed param.

    Generates reproducible (B, L, V) logits by mixing input token ids with a
    position-dependent bias. No real learning — just enough structure for the
    samplers to exercise their code paths.
    """

    def __init__(self, vocab=VOCAB, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.bias = torch.nn.Parameter(
            torch.randn(vocab, generator=g), requires_grad=False
        )
        self.pos_bias = torch.nn.Parameter(
            torch.randn(256, vocab, generator=g), requires_grad=False
        )
        self.embed = torch.nn.Parameter(
            torch.randn(vocab, vocab, generator=g), requires_grad=False
        )
        self.vocab = vocab

    @property
    def device(self):
        return self.bias.device

    def forward(self, input_ids, attention_mask=None):
        # (B, L) -> (B, L, V)
        e = self.embed[input_ids.clamp(min=0, max=self.vocab - 1)]  # (B, L, V)
        L = input_ids.shape[1]
        p = self.pos_bias[:L].unsqueeze(0)  # (1, L, V)
        logits = e + p + self.bias
        return SimpleNamespace(logits=logits)


def _make_tokenizer():
    return SimpleNamespace(
        mask_token_id=MASK_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=EOS_ID,
    )


def _fresh_sampler(cls):
    """Fresh model+tokenizer per call so repeated runs are independent."""
    model = _FixedLogitsModel(seed=42).eval()
    tok = _make_tokenizer()
    return cls(model=model, tokenizer=tok, scheduler=None)


def _prompts(B=2):
    return [
        torch.tensor([5, 6, 7], dtype=torch.long),
        torch.tensor([5, 6, 7, 8], dtype=torch.long),
    ][:B]


class TestGibbsMDLMParity:
    """With edit_freq=-1, GibbsSampler must reduce to MDLMSampler."""

    @pytest.mark.parametrize("remasking", ["low_confidence", "random"])
    def test_disabled_gibbs_matches_mdlm(self, remasking):
        prompts = _prompts()
        common = dict(
            max_new_tokens=8,
            block_size=8,
            steps=8,
            temperature=0.0,
            remasking=remasking,
        )

        torch.manual_seed(0)
        mdlm = _fresh_sampler(MDLMSampler)
        out_mdlm = mdlm.sample(
            prompts, config=MDLMSamplerConfig(**common), return_dict=False
        )

        torch.manual_seed(0)
        gibbs = _fresh_sampler(GibbsSampler)
        out_gibbs = gibbs.sample(
            prompts,
            config=GibbsSamplerConfig(edit_freq=-1, **common),
            return_dict=False,
        )

        assert torch.equal(out_mdlm, out_gibbs), (
            f"MDLM and disabled-Gibbs outputs diverged for remasking={remasking}"
        )


class TestGibbsVariants:
    """All three edit_strategy variants run end-to-end and fully unmask the canvas."""

    @pytest.mark.parametrize(
        "edit_strategy", ["gibbs_standard", "gibbs_edit", "gibbs_edit_v2"]
    )
    @pytest.mark.parametrize("remasking_strategy", ["random", "low_confidence"])
    @pytest.mark.parametrize("keep_original_mask", [True, False])
    def test_variant_runs_and_unmasks(
        self, edit_strategy, remasking_strategy, keep_original_mask
    ):
        prompts = _prompts()
        torch.manual_seed(1)
        gibbs = _fresh_sampler(GibbsSampler)
        cfg = GibbsSamplerConfig(
            max_new_tokens=6,
            block_size=6,
            steps=6,
            temperature=0.0,
            remasking="low_confidence",
            edit_freq=2,
            edit_step=2,
            edit_start=0,
            edit_strategy=edit_strategy,
            remasking_strategy=remasking_strategy,
            keep_original_mask=keep_original_mask,
        )
        out = gibbs.sample(prompts, config=cfg, return_dict=False)

        assert out.dtype == torch.long
        B, T = out.shape
        assert B == len(prompts)

        # Generation region must be fully revealed — the outer MDLM loop commits
        # every position by the last step, so no mask_id should remain there.
        max_pl = max(p.shape[0] for p in prompts)
        assert T == max_pl + cfg.max_new_tokens
        for i, p in enumerate(prompts):
            pl = p.shape[0]
            gen = out[i, pl : pl + cfg.max_new_tokens]
            assert (
                (gen != MASK_ID).all().item()
            ), f"mask_id leaked into generation region for variant={edit_strategy}"

    def test_edit_freq_zero_never_triggers(self):
        """edit_freq=-1 must match edit_freq=0 (both disable)."""
        prompts = _prompts()
        common = dict(
            max_new_tokens=4,
            block_size=4,
            steps=4,
            temperature=0.0,
            remasking="low_confidence",
            edit_step=3,
            edit_strategy="gibbs_standard",
            remasking_strategy="random",
        )

        torch.manual_seed(2)
        s = _fresh_sampler(GibbsSampler)
        out_a = s.sample(prompts, config=GibbsSamplerConfig(edit_freq=-1, **common))

        torch.manual_seed(2)
        s = _fresh_sampler(GibbsSampler)
        out_b = s.sample(prompts, config=GibbsSamplerConfig(edit_freq=0, **common))

        assert torch.equal(out_a, out_b)

    def test_keep_original_mask_preserves_pattern(self):
        """gibbs_standard with keep_original_mask=True must return a sequence whose
        masked positions equal the pre-correction masked set."""
        gibbs = _fresh_sampler(GibbsSampler)
        B, L = 2, 8
        x = torch.full((B, L), 9, dtype=torch.long)
        x[0, 3:6] = MASK_ID
        x[1, 2:7] = MASK_ID
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[:, :2] = True  # pretend first 2 tokens are prompt
        attention_mask = torch.ones(B, L, dtype=torch.long)

        original_mask = x == MASK_ID
        logits = gibbs.model(x).logits

        yt, _ = gibbs._gibbs_standard(
            x=x,
            logits=logits,
            prompt_mask=prompt_mask,
            attention_mask=attention_mask,
            mask_id=MASK_ID,
            edit_step=1,
            remasking_strategy="random",
            suppress_tokens=None,
            begin_suppress_tokens=None,
            right_shift_logits=False,
            keep_original_mask=True,
        )
        assert torch.equal(yt == MASK_ID, original_mask), (
            "keep_original_mask=True must preserve the input mask pattern in gibbs_standard final step"
        )

    def test_keep_original_mask_false_uses_strategy(self):
        """gibbs_edit with keep_original_mask=False must remask num_masks positions
        (possibly different from the original set)."""
        gibbs = _fresh_sampler(GibbsSampler)
        B, L = 2, 8
        x = torch.full((B, L), 9, dtype=torch.long)
        x[0, 3:6] = MASK_ID
        x[1, 2:7] = MASK_ID
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[:, :2] = True
        attention_mask = torch.ones(B, L, dtype=torch.long)

        num_masks = (x == MASK_ID).sum(dim=-1)
        logits = gibbs.model(x).logits

        yt, _ = gibbs._gibbs_edit_v1(
            x=x,
            logits=logits,
            prompt_mask=prompt_mask,
            attention_mask=attention_mask,
            mask_id=MASK_ID,
            edit_step=1,
            remasking_strategy="random",
            suppress_tokens=None,
            begin_suppress_tokens=None,
            right_shift_logits=False,
            keep_original_mask=False,
        )
        # Same total count of masks, and no masks inside prompt region.
        assert torch.equal((yt == MASK_ID).sum(dim=-1), num_masks)
        assert not (yt[:, :2] == MASK_ID).any()

    def test_invalid_edit_strategy_raises(self):
        prompts = _prompts(B=1)
        gibbs = _fresh_sampler(GibbsSampler)
        cfg = GibbsSamplerConfig(
            max_new_tokens=4,
            block_size=4,
            steps=4,
            edit_freq=1,
            edit_step=1,
            edit_strategy="not_a_real_strategy",
        )
        with pytest.raises(AssertionError):
            gibbs.sample(prompts, config=cfg)
