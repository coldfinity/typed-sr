"""
train/reinforce.py

REINFORCE policy gradient update for typed-sr.

The policy is the RNNSampler. For a batch of sampled expressions:

  loss = -mean_over_batch( advantage * sum_of_log_probs ) - beta * entropy

Where:
  advantage   = reward - baseline   (baseline = running mean of rewards)
  entropy     = -sum( p * log p )   averaged over active tokens in the batch
  beta        = entropy bonus weight (encourages exploration)

The baseline reduces variance without introducing bias (standard REINFORCE
with a learned-free moving average baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class REINFORCEConfig:
    lr: float = 1e-3
    entropy_beta: float = 0.005   # weight on entropy bonus
    baseline_momentum: float = 0.99  # EMA decay for reward baseline


class REINFORCETrainer:
    """
    Wraps an RNNSampler (and optionally a DatasetEncoder) with a REINFORCE
    update rule.

    Parameters
    ----------
    policy : nn.Module
        The model whose parameters are updated (typically RNNSampler, or
        a combined encoder+sampler module).
    config : REINFORCEConfig
    """

    def __init__(self, policy: nn.Module, config: REINFORCEConfig | None = None):
        self.policy = policy
        self.cfg = config or REINFORCEConfig()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.cfg.lr)
        self._baseline: float = 0.0   # running mean reward

    def update(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Perform one REINFORCE gradient step.

        Parameters
        ----------
        log_probs : torch.Tensor, shape (B, T)
            Per-token log probabilities from RNNSampler.sample(). Padding
            steps should already be zeroed (RNNSampler guarantees this).
        rewards : torch.Tensor, shape (B,)
            Scalar reward per sequence, in [0, 1].
        entropy : torch.Tensor or None, shape (B, T) or scalar
            Per-token entropy values. If None, the entropy bonus is skipped.

        Returns
        -------
        dict with keys "loss", "policy_loss", "entropy_loss", "baseline"
        """
        # Sum log probs across tokens -> (B,)
        seq_log_probs = log_probs.sum(dim=1)

        # Update baseline with EMA
        mean_reward = rewards.mean().item()
        self._baseline = (
            self.cfg.baseline_momentum * self._baseline
            + (1 - self.cfg.baseline_momentum) * mean_reward
        )

        advantage = rewards - self._baseline  # (B,)

        policy_loss = -(advantage.detach() * seq_log_probs).mean()

        if entropy is not None:
            entropy_loss = -self.cfg.entropy_beta * entropy.mean()
        else:
            entropy_loss = torch.tensor(0.0, device=log_probs.device)

        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "baseline": self._baseline,
        }


def compute_entropy(log_probs_full: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy from a full log-probability distribution.

    Parameters
    ----------
    log_probs_full : torch.Tensor, shape (B, T, vocab_size)
        Log probabilities over the full vocabulary at each step.

    Returns
    -------
    torch.Tensor, shape (B, T)
    """
    probs = log_probs_full.exp()
    return -(probs * log_probs_full).sum(dim=-1)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grammar"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

    import torch
    from cfg import CFG
    from rnn import RNNSampler

    torch.manual_seed(0)

    cfg = CFG(n_vars=2, max_depth=4)
    sampler = RNNSampler(cfg, embed_dim=256)
    trainer = REINFORCETrainer(sampler)

    print("Running 10 update steps with random rewards...\n")
    for step in range(10):
        z = torch.randn(16, 256)
        token_ids, log_probs = sampler.sample(z, max_len=30)

        # Fake rewards: in real training these come from reward.compute_reward
        rewards = torch.rand(16)

        stats = trainer.update(log_probs, rewards)
        print(
            f"  step {step+1:2d} | loss={stats['loss']:+.4f} "
            f"policy={stats['policy_loss']:+.4f} "
            f"baseline={stats['baseline']:.4f}"
        )

    print("\nSmoke test passed.")
