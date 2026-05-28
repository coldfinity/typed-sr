"""
model/rnn.py

LSTM sampler for typed-sr.

At each decoding step the LSTM produces logits over the token vocabulary.
The CFG validity mask is applied to those logits (as a -inf additive mask)
before sampling, making it impossible to produce a grammatically invalid
expression.

Interface
---------
  sampler = RNNSampler(cfg, embed_dim=256)
  token_ids, log_probs = sampler.sample(z, max_len=30)

  z          : (B, embed_dim)   — from DatasetEncoder
  token_ids  : (B, T)           — sampled token indices (int)
  log_probs  : (B, T)           — log prob of each sampled token (float)
                                   zero-padded after sequence completes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grammar"))
from cfg import CFG


class RNNSampler(nn.Module):
    """
    LSTM that samples token sequences conditioned on a dataset embedding.

    Parameters
    ----------
    cfg : CFG
        Grammar used to compute validity masks and decode token indices.
    embed_dim : int
        Dimension of the context embedding z (output of DatasetEncoder).
    token_embed_dim : int
        Dimension of the token embedding fed as LSTM input.
    hidden_dim : int
        LSTM hidden state dimension.
    """

    def __init__(
        self,
        cfg: CFG,
        embed_dim: int = 256,
        token_embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.cfg = cfg

        # Initialise LSTM state from encoder embedding
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim)

        # Learned start-of-sequence input
        self.start_embed = nn.Parameter(torch.zeros(token_embed_dim))

        self.token_embed = nn.Embedding(cfg.vocab_size, token_embed_dim)
        self.lstm = nn.LSTM(token_embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, cfg.vocab_size)

    def sample(
        self, z: torch.Tensor, max_len: int = 30
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of token sequences from the grammar-constrained policy.

        Parameters
        ----------
        z : torch.Tensor, shape (B, embed_dim)
        max_len : int
            Hard upper bound on sequence length.

        Returns
        -------
        token_ids : torch.Tensor, shape (B, T)   — int64
        log_probs : torch.Tensor, shape (B, T)   — float32, 0.0 for padding steps
        """
        B = z.shape[0]
        device = z.device

        h = self.init_h(z).unsqueeze(0)  # (1, B, hidden_dim)
        c = self.init_c(z).unsqueeze(0)  # (1, B, hidden_dim)

        # (B, 1, token_embed_dim)
        inp = self.start_embed.unsqueeze(0).expand(B, -1).unsqueeze(1)

        states = [self.cfg.initial_state() for _ in range(B)]
        finished = [False] * B  # plain list — never part of the autograd graph

        all_token_ids: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []

        for _ in range(max_len):
            if all(finished):
                break

            out, (h, c) = self.lstm(inp, (h, c))  # out: (B, 1, hidden_dim)
            logits = self.output(out.squeeze(1))  # (B, vocab_size)

            # Build validity mask for entire batch
            masks = []
            for i, state in enumerate(states):
                if finished[i]:
                    # Sequence done — any mask works; use uniform so sampling
                    # doesn't hit -inf and produce NaN gradients
                    masks.append(torch.ones(self.cfg.vocab_size, device=device))
                else:
                    m = torch.from_numpy(self.cfg.validity_mask(state)).to(device)
                    masks.append(m)
            batch_mask = torch.stack(masks)  # (B, vocab_size)

            # Apply mask: -inf where token is invalid
            masked_logits = logits + (batch_mask - 1.0) * 1e9

            dist = Categorical(logits=masked_logits)
            token_ids_step = dist.sample()  # (B,)
            log_probs_step = dist.log_prob(token_ids_step)  # (B,)

            # Zero out log probs for already-finished sequences
            finished_tensor = torch.tensor(finished, dtype=torch.bool, device=device)
            log_probs_step = log_probs_step.masked_fill(finished_tensor, 0.0)

            all_token_ids.append(token_ids_step)
            all_log_probs.append(log_probs_step)

            # Advance grammar states for active sequences
            for i, tid in enumerate(token_ids_step.tolist()):
                if finished[i]:
                    continue
                token = self.cfg.idx_to_token(tid)
                states[i] = states[i].advance(token)
                if states[i].is_complete:
                    finished[i] = True

            # Next input: embedding of sampled token
            inp = self.token_embed(token_ids_step).unsqueeze(
                1
            )  # (B, 1, token_embed_dim)

        token_ids = torch.stack(all_token_ids, dim=1)  # (B, T)
        log_probs = torch.stack(all_log_probs, dim=1)  # (B, T)
        return token_ids, log_probs


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grammar"))
    from cfg import CFG, sequence_to_tree

    torch.manual_seed(0)

    cfg = CFG(n_vars=2, max_depth=4)
    sampler = RNNSampler(cfg, embed_dim=256, token_embed_dim=64, hidden_dim=256)

    B = 8
    z = torch.randn(B, 256)

    token_ids, log_probs = sampler.sample(z, max_len=30)
    print(f"token_ids : {tuple(token_ids.shape)}")
    print(f"log_probs : {tuple(log_probs.shape)}")

    # Every sampled sequence should be a complete, valid expression
    print("\nSampled sequences:")
    for i in range(B):
        tokens = [cfg.idx_to_token(t) for t in token_ids[i].tolist()]
        # Strip trailing tokens that belong to padding steps (log_prob == 0
        # after completion) — find first completion point
        state = cfg.initial_state()
        seq = []
        for tok in tokens:
            seq.append(tok)
            state = state.advance(tok)
            if state.is_complete:
                break
        try:
            tree = sequence_to_tree(seq, cfg)
            total_lp = log_probs[i][: len(seq)].sum().item()
            print(f"  [{i}] {tree}  (len={len(seq)}, logp={total_lp:.2f})")
        except Exception as e:
            print(f"  [{i}] INVALID: {e}  tokens={seq}")

    param_count = sum(p.numel() for p in sampler.parameters())
    print(f"\nParameters: {param_count:,}")
