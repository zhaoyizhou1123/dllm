"""
DAgger-style pool for training diffusion language models.

Simulates the inference sampling procedure during training.  Each micro-step
reveals ``ceil(L_eff / K)`` tokens via one of three strategies:

  - ``low_confidence``: reveal highest-confidence masked positions within the
    current block window (default, mirrors inference-time block sampling)
  - ``uniform``: randomly choose masked positions within the current block
  - ``markovian``: reveal ALL tokens, then re-mask with block structure
    (earlier blocks kept, frontier block partially revealed, later blocks
    masked).  Uses interval-based ratio sampling for stochasticity.

Policy interpolation mixes expert (ground-truth) and policy (model-predicted)
tokens via a beta schedule.

Provides:
  - DAggerPool: pool class with block-aware unmasking and DAgger interpolation
"""

import math
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from dllm.core.trainers.pool import unmask_from_scores, build_intervals


class DAggerPool:
    """
    DAgger pool for diffusion LM training.

    Maintains a batch of partially-masked sequences.  Each micro-step reveals
    ``tokens_per_step = ceil(L_eff / K)`` tokens, advancing through blocks as
    they are completed (for block-based strategies).  When a sequence is fully
    unmasked it is replaced with a fresh sample from the DataLoader.

    Strategies:
      - ``low_confidence``: block-based, reveal by highest model confidence
      - ``uniform``:        block-based, reveal random masked positions
      - ``markovian``:      block-level; reveal all, then re-mask with block
                           structure + interval-sampled ratio

    Policy interpolation (DAgger):
      - Each revealed token uses the ground-truth value with probability beta
        and the model's greedy prediction with probability 1-beta.
      - beta decays linearly from 1 to 0 over ``beta_warmup_steps`` global
        optimizer steps, then stays at 0.

    Batch dict format expected from the DataLoader::

        {"labels": LongTensor[L], "prompt_mask": BoolTensor[L]}
    """

    STRATEGIES = ("low_confidence", "uniform", "markovian")

    def __init__(
        self,
        train_loader: DataLoader,
        batch_size: int,
        mask_id: int,
        block_size: int,
        K: int,
        device: torch.device,
        L: int,
        strategy: str = "low_confidence",
    ):
        assert strategy in self.STRATEGIES, (
            f"invalid strategy: {strategy!r}, must be one of {self.STRATEGIES}"
        )
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.mask_id = mask_id
        self.block_size = block_size
        self.K = K
        self.device = device
        self.L = L
        self.strategy = strategy

        # Interval tensors for markovian ratio sampling
        intervals = build_intervals(K)
        self._interval_lo = torch.tensor(
            [a for a, _ in intervals], device=device, dtype=torch.float
        )
        self._interval_hi = torch.tensor(
            [b for _, b in intervals], device=device, dtype=torch.float
        )

        self._reset_iter()
        self._initialize_pool()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def compute_beta(global_step: int, beta_warmup_steps: int) -> float:
        """Linear beta schedule: 1 → 0 over *beta_warmup_steps*, then 0."""
        if beta_warmup_steps <= 0:
            return 0.0
        return max(0.0, 1.0 - global_step / beta_warmup_steps)

    def reset_loader_iter(self):
        """Reset the internal DataLoader iterator (call at epoch start)."""
        self._reset_iter()

    @torch.no_grad()
    def update_from_logits(
        self,
        logits: torch.Tensor,
        global_step: int,
        beta_warmup_steps: int,
    ):
        """
        Reveal tokens using the DAgger policy.

        Args:
            logits:            [B, L, V] raw model output
            global_step:       current optimizer step (for beta schedule)
            beta_warmup_steps: total steps for linear beta decay 1→0
        """
        B, L, V = logits.shape
        device = self.device

        pred_tokens = logits.argmax(dim=-1)  # [B, L]
        max_log_conf = (
            logits.max(dim=-1)[0] - torch.logsumexp(logits, dim=-1)
        )  # [B, L]

        beta = self.compute_beta(global_step, beta_warmup_steps)

        # --- DAgger source tokens --------------------------------------------
        if beta >= 1.0:
            reveal_source = self.x0
        elif beta <= 0.0:
            reveal_source = pred_tokens
        else:
            coin = torch.rand(B, L, device=device) < beta
            reveal_source = torch.where(coin, self.x0, pred_tokens)

        if self.strategy == "markovian":
            self._update_markovian(reveal_source)
        else:
            self._update_block_based(max_log_conf, reveal_source)

    # ------------------------------------------------------------------
    # Strategy: block-based (low_confidence / uniform)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_block_based(
        self,
        max_log_conf: torch.Tensor,
        reveal_source: torch.Tensor,
    ):
        """Reveal ceil(L_eff/K) tokens from current block, advance blocks."""
        B = self.xt.shape[0]
        L = self.xt.shape[1]
        device = self.device

        L_eff = self.state["L_eff"]
        tokens_per_step = (L_eff + self.K - 1) // self.K
        tokens_remaining = tokens_per_step.clone()

        response_pos = self.state["response_pos"]
        prompt_mask = self.state["prompt_mask"]

        for _ in range(max(1, math.ceil(self.L / self.block_size))):
            if (tokens_remaining <= 0).all():
                break

            current_block = self.state["current_block"]
            block_lo = current_block * self.block_size
            block_hi = block_lo + self.block_size
            in_current_block = (
                (response_pos >= block_lo.unsqueeze(1))
                & (response_pos < block_hi.unsqueeze(1))
            )
            eligible = (
                (self.xt == self.mask_id) & ~prompt_mask & in_current_block
            )
            num_eligible = eligible.sum(dim=1)
            to_reveal = torch.minimum(tokens_remaining, num_eligible)

            if to_reveal.max().item() > 0:
                NEG_INF = torch.finfo(max_log_conf.dtype).min
                if self.strategy == "uniform":
                    score = torch.where(
                        eligible,
                        torch.rand(B, L, device=device, dtype=max_log_conf.dtype),
                        torch.full_like(max_log_conf, NEG_INF),
                    )
                else:  # low_confidence
                    score = torch.where(
                        eligible,
                        max_log_conf,
                        torch.full_like(max_log_conf, NEG_INF),
                    )
                self.xt = unmask_from_scores(
                    score, to_reveal, reveal_source, self.xt
                )

            tokens_remaining = tokens_remaining - to_reveal

            still_eligible = (
                (self.xt == self.mask_id) & ~prompt_mask & in_current_block
            )
            block_done = still_eligible.sum(dim=1) == 0
            self.state["current_block"] = torch.where(
                block_done, current_block + 1, current_block
            )

        # --- replace completed sequences -------------------------------------
        done = self.state["current_block"] >= self.state["num_blocks"]
        self._replace_done(done)

    # ------------------------------------------------------------------
    # Strategy: markovian
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_markovian(self, reveal_source: torch.Tensor):
        """Reveal all tokens, then re-mask with block structure."""
        B, L = self.xt.shape
        device = self.device
        prompt_mask = self.state["prompt_mask"]
        L_eff = self.state["L_eff"]
        response_pos = self.state["response_pos"]
        steps_taken = self.state["steps_taken"]

        is_response = ~prompt_mask

        # 1. Fill all response positions with reveal_source (memoryless)
        fully_revealed = torch.where(is_response, reveal_source, self.xt)

        # 2. Interval-based keep ratio
        next_step = steps_taken + 1
        stage_idx = next_step.clamp(0, self.K - 1)
        lo = self._interval_lo.index_select(0, stage_idx)
        hi = self._interval_hi.index_select(0, stage_idx)
        keep_ratio = lo + torch.rand_like(lo) * (hi - lo)  # [B]
        num_to_keep = (keep_ratio * L_eff.float()).round().long().clamp_min(0)
        num_to_keep = torch.minimum(num_to_keep, L_eff)

        # 3. Block-structured re-masking
        full_blocks = num_to_keep // self.block_size
        tokens_in_frontier = num_to_keep % self.block_size

        new_xt = torch.full_like(self.xt, self.mask_id)
        new_xt = torch.where(prompt_mask, self.x0, new_xt)

        # Reveal all positions in completed blocks
        block_boundary = full_blocks * self.block_size
        in_completed = (response_pos >= 0) & (
            response_pos < block_boundary.unsqueeze(1)
        )
        new_xt = torch.where(in_completed, fully_revealed, new_xt)

        # Reveal random positions in the frontier block
        frontier_lo = block_boundary
        frontier_hi = frontier_lo + self.block_size
        in_frontier = (
            (response_pos >= frontier_lo.unsqueeze(1))
            & (response_pos < frontier_hi.unsqueeze(1))
        )

        k_max = int(tokens_in_frontier.max().item())
        if k_max > 0:
            rand_scores = torch.rand(B, L, device=device)
            rand_scores = torch.where(
                in_frontier,
                rand_scores,
                torch.full_like(rand_scores, torch.finfo(rand_scores.dtype).min),
            )
            new_xt = unmask_from_scores(
                rand_scores, tokens_in_frontier, fully_revealed, new_xt
            )

        self.xt = new_xt
        self.state["steps_taken"] = next_step

        # --- replace completed sequences -------------------------------------
        done = next_step >= self.K
        self._replace_done(done)

    # ------------------------------------------------------------------
    # Shared replacement logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _replace_done(self, done: torch.Tensor):
        """Replace completed sequences with fresh fully-masked samples."""
        n_new = int(done.sum().item())
        if n_new == 0:
            return

        idx = done.nonzero(as_tuple=False).squeeze(1)
        new_x0, new_pm = self._get_new_seq(n_new)
        new_L_eff = (~new_pm).sum(dim=1).long()
        new_resp_pos = self._compute_response_pos(new_pm)
        new_num_blocks = (new_L_eff + self.block_size - 1) // self.block_size

        # start fully masked
        new_xt = torch.full_like(new_x0, self.mask_id)
        new_xt = torch.where(new_pm, new_x0, new_xt)

        self.x0[idx] = new_x0
        self.xt[idx] = new_xt
        self.state["prompt_mask"][idx] = new_pm
        self.state["L_eff"][idx] = new_L_eff
        self.state["response_pos"][idx] = new_resp_pos
        self.state["num_blocks"][idx] = new_num_blocks
        self.state["current_block"][idx] = 0
        self.state["steps_taken"][idx] = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_iter(self):
        self._iter = iter(self.train_loader)

    @staticmethod
    def _compute_response_pos(prompt_mask: torch.Tensor) -> torch.Tensor:
        """0-indexed position within the response region, -1 for prompt."""
        resp = (~prompt_mask).long()
        return (resp.cumsum(dim=1) - 1) * resp + prompt_mask.long() * (-1)

    @torch.no_grad()
    def _get_new_seq(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw *n* sequences from the DataLoader (handles exhaustion)."""
        out, masks = [], []
        while len(out) < n:
            try:
                batch = next(self._iter)
            except StopIteration:
                print(
                    "Warning: train loader exhausted — reshuffling and "
                    "restarting."
                )
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

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_at_random_timestep(
        self,
        x0: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Block-aware initialisation with uniform random timesteps.

        Returns:
            xt:            [B, L]  partially masked input
            current_block: [B]     block index each sequence is currently in
        """
        B, L = x0.shape
        device = self.device

        L_eff = (~prompt_mask).sum(dim=1).long()
        response_pos = self._compute_response_pos(prompt_mask)

        t = torch.rand(B, device=device)
        num_unmasked = torch.minimum(
            (t * L_eff.float()).long(), (L_eff - 1).clamp_min(0)
        ).clamp_min(0)

        current_block = num_unmasked // self.block_size
        tokens_in_block = num_unmasked % self.block_size

        xt = torch.full_like(x0, self.mask_id)
        xt = torch.where(prompt_mask, x0, xt)

        block_boundary = current_block * self.block_size
        fully_unmasked = (response_pos >= 0) & (
            response_pos < block_boundary.unsqueeze(1)
        )
        xt = torch.where(fully_unmasked, x0, xt)

        in_current_block = (
            (response_pos >= block_boundary.unsqueeze(1))
            & (response_pos < (block_boundary + self.block_size).unsqueeze(1))
        )
        rand_scores = torch.rand(B, L, device=device)
        rand_scores = torch.where(
            in_current_block,
            rand_scores,
            torch.full_like(rand_scores, torch.finfo(rand_scores.dtype).min),
        )
        k_max = int(tokens_in_block.max().item())
        if k_max > 0:
            xt = unmask_from_scores(rand_scores, tokens_in_block, x0, xt)

        return xt, current_block

    @torch.no_grad()
    def _init_markovian(
        self,
        x0: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Markovian initialisation with block structure and interval sampling.

        Returns:
            xt:          [B, L]  partially masked input
            steps_taken: [B]     how many steps already completed
        """
        B, L = x0.shape
        device = self.device

        L_eff = (~prompt_mask).sum(dim=1).long()
        response_pos = self._compute_response_pos(prompt_mask)

        # Sample random step for each sequence
        steps_taken = (torch.rand(B, device=device) * self.K).long().clamp(0, self.K - 1)

        # Interval-based keep ratio
        lo = self._interval_lo.index_select(0, steps_taken)
        hi = self._interval_hi.index_select(0, steps_taken)
        keep_ratio = lo + torch.rand_like(lo) * (hi - lo)
        num_to_keep = (keep_ratio * L_eff.float()).round().long().clamp_min(0)
        num_to_keep = torch.minimum(num_to_keep, L_eff)

        # Block-structured masking
        full_blocks = num_to_keep // self.block_size
        tokens_in_frontier = num_to_keep % self.block_size

        xt = torch.full_like(x0, self.mask_id)
        xt = torch.where(prompt_mask, x0, xt)

        # Reveal completed blocks
        block_boundary = full_blocks * self.block_size
        in_completed = (response_pos >= 0) & (
            response_pos < block_boundary.unsqueeze(1)
        )
        xt = torch.where(in_completed, x0, xt)

        # Reveal random positions in frontier block
        in_frontier = (
            (response_pos >= block_boundary.unsqueeze(1))
            & (response_pos < (block_boundary + self.block_size).unsqueeze(1))
        )
        k_max = int(tokens_in_frontier.max().item())
        if k_max > 0:
            rand_scores = torch.rand(B, L, device=device)
            rand_scores = torch.where(
                in_frontier,
                rand_scores,
                torch.full_like(rand_scores, torch.finfo(rand_scores.dtype).min),
            )
            xt = unmask_from_scores(rand_scores, tokens_in_frontier, x0, xt)

        return xt, steps_taken

    @torch.no_grad()
    def _initialize_pool(self):
        """Fill the pool with uniformly scattered timesteps."""
        B = self.batch_size
        x0, pm = self._get_new_seq(B)
        L_eff = (~pm).sum(dim=1).long()
        response_pos = self._compute_response_pos(pm)
        num_blocks = (L_eff + self.block_size - 1) // self.block_size

        if self.strategy == "markovian":
            xt, steps_taken = self._init_markovian(x0, pm)
            current_block = torch.zeros(B, device=self.device, dtype=torch.long)
        else:
            xt, current_block = self._init_at_random_timestep(x0, pm)
            steps_taken = torch.zeros(B, device=self.device, dtype=torch.long)

        self.x0 = x0
        self.xt = xt
        self.state = dict(
            prompt_mask=pm,
            L_eff=L_eff,
            response_pos=response_pos,
            num_blocks=num_blocks,
            current_block=current_block,
            steps_taken=steps_taken,
        )
