"""
model/dsr.py

Deep Symbolic Regression training loop.

Orchestrates the full DSR pipeline:
  1. Encode dataset (X, y) → embedding z via DatasetEncoder
  2. Sample B expression token sequences via RNNSampler
  3. Decode token sequences → expression trees
  4. Compute rewards via reward.batch_reward
  5. Update ParetoFrontier
  6. REINFORCE gradient update
  7. Repeat; stop early if a perfect expression is found
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grammar"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from cfg import CFG, sequence_to_tree
from encoder import DatasetEncoder
from reinforce import REINFORCEConfig, REINFORCETrainer
from reward import ParetoEntry, ParetoFrontier, batch_reward, is_perfect
from rnn import RNNSampler
from tree import Node


@dataclass
class DSRConfig:
    batch_size: int = 256
    max_len: int = 30
    lambda_complexity: float = 0.001
    perfect_tol: float = 1e-4  # NMSE threshold for early stopping
    embed_dim: int = 256
    encoder_hidden: int = 128
    rnn_hidden: int = 256
    token_embed_dim: int = 64


class DSR:
    """
    Full DSR training loop.

    Parameters
    ----------
    cfg : CFG
        Grammar (defines vocab, masking rules).
    n_vars : int
        Number of input variables.
    config : DSRConfig
    """

    def __init__(self, cfg: CFG, n_vars: int, config: DSRConfig | None = None):
        self.cfg = cfg
        self.dsr_config = config or DSRConfig()
        c = self.dsr_config

        self.encoder = DatasetEncoder(
            n_vars=n_vars,
            hidden_dim=c.encoder_hidden,
            embed_dim=c.embed_dim,
        )
        self.sampler = RNNSampler(
            cfg=cfg,
            embed_dim=c.embed_dim,
            token_embed_dim=c.token_embed_dim,
            hidden_dim=c.rnn_hidden,
        )

        # Combine encoder + sampler parameters under one optimizer
        all_params = list(self.encoder.parameters()) + list(self.sampler.parameters())

        class _CombinedPolicy(torch.nn.Module):
            def __init__(self, encoder, sampler):
                super().__init__()
                self.encoder = encoder
                self.sampler = sampler

        combined = _CombinedPolicy(self.encoder, self.sampler)
        self.trainer = REINFORCETrainer(combined, REINFORCEConfig())
        # Override the optimizer to cover both modules
        self.trainer.optimizer = torch.optim.Adam(all_params, lr=self.trainer.cfg.lr)

        self.frontier = ParetoFrontier()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_steps: int = 1000,
        log_every: int = 100,
    ) -> ParetoEntry | None:
        """
        Run the DSR training loop.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_vars)
        y : np.ndarray, shape (N,)
        n_steps : int
            Number of gradient steps.
        log_every : int
            Print progress every this many steps.

        Returns
        -------
        Best ParetoEntry found, or None if nothing improved.
        """
        c = self.dsr_config
        B = c.batch_size

        var_values = {f"x{i+1}": X[:, i] for i in range(X.shape[1])}

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float()

        # Encoder expects (B, N, n_vars) — broadcast the single dataset
        X_batch = X_t.unsqueeze(0).expand(B, -1, -1)  # (B, N, n_vars)
        y_batch = y_t.unsqueeze(0).expand(B, -1)  # (B, N)

        for step in range(1, n_steps + 1):
            # 1. Encode
            z = self.encoder(X_batch, y_batch)  # (B, embed_dim)

            # 2. Sample
            token_ids, log_probs = self.sampler.sample(z, max_len=c.max_len)

            # 3. Decode token sequences → trees
            nodes = self._decode_batch(token_ids)

            # 4. Compute rewards (numpy)
            rewards_np = batch_reward(nodes, var_values, y, c.lambda_complexity)
            rewards = torch.from_numpy(rewards_np)

            # 5. Update Pareto frontier; check for perfect solution
            found_perfect = False
            for node in nodes:
                if node is None:
                    continue
                self.frontier.update(node, var_values, y, c.lambda_complexity)
                if is_perfect(node, var_values, y, tol=c.perfect_tol):
                    found_perfect = True

            # 6. REINFORCE update
            stats = self.trainer.update(log_probs, rewards)

            if log_every and step % log_every == 0:
                best = self.frontier.best()
                best_str = str(best.expression) if best else "none"
                print(
                    f"step {step:5d} | loss={stats['loss']:+.4f} "
                    f"reward_mean={rewards_np.mean():.4f} "
                    f"baseline={stats['baseline']:.4f} "
                    f"best={best_str}"
                )

            if found_perfect:
                print(f"Perfect expression found at step {step}.")
                break

        return self.frontier.best()

    def _decode_batch(self, token_ids: torch.Tensor) -> list[Node | None]:
        """
        Convert a batch of token-id sequences to expression trees.

        Walks each sequence using ParseState to find the completion point,
        then calls sequence_to_tree. Returns None for sequences that fail.
        """
        nodes = []
        for i in range(token_ids.shape[0]):
            ids = token_ids[i].tolist()
            state = self.cfg.initial_state()
            seq = []
            try:
                for tid in ids:
                    token = self.cfg.idx_to_token(tid)
                    seq.append(token)
                    state = state.advance(token)
                    if state.is_complete:
                        break
                node = sequence_to_tree(seq, self.cfg)
            except Exception:
                node = None
            nodes.append(node)
        return nodes


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)

    # Target: x1 + x2
    N = 50
    X = rng.uniform(-2, 2, (N, 2))
    y = X[:, 0] + X[:, 1]

    cfg = CFG(n_vars=2, max_depth=4)
    dsr = DSR(cfg, n_vars=2, config=DSRConfig(batch_size=64, max_len=20))

    best = dsr.fit(X, y, n_steps=500, log_every=100)

    print("\n=== Pareto frontier ===")
    dsr.frontier.print()
    if best:
        print(f"\nBest: {best.expression}  reward={best.reward:.4f}")
