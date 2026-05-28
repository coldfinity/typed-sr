"""
model/encoder.py

Dataset context encoder for typed-sr.

Maps a batch of (X, y) datasets to fixed-size embeddings that condition
the RNN sampler. Each dataset is a set of N input-output points; the
encoder is permutation-invariant via mean pooling.

Architecture
------------
  per-point MLP : (n_vars + 1) → hidden_dim → hidden_dim
  mean pool     : aggregate over N points
  projection    : hidden_dim → embed_dim
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DatasetEncoder(nn.Module):
    """
    Encodes a batch of datasets into fixed-size embeddings.

    Parameters
    ----------
    n_vars : int
        Number of input variables per data point.
    hidden_dim : int
        Width of the per-point MLP hidden layer.
    embed_dim : int
        Dimension of the output embedding.
    """

    def __init__(self, n_vars: int, hidden_dim: int = 128, embed_dim: int = 256):
        super().__init__()
        input_dim = n_vars + 1  # x1...xN concatenated with y

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.project = nn.Linear(hidden_dim, embed_dim)

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor, shape (B, N, n_vars)
            Input variable values for B datasets, each with N points.
        y : torch.Tensor, shape (B, N)
            Target values.

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
        """
        # (B, N, n_vars+1)
        points = torch.cat([X, y.unsqueeze(-1)], dim=-1)
        # (B, N, hidden_dim)
        hidden = self.mlp(points)
        # (B, hidden_dim) — permutation-invariant aggregation
        pooled = hidden.mean(dim=1)
        # (B, embed_dim)
        return self.project(pooled)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B, N, n_vars = 4, 50, 3
    encoder = DatasetEncoder(n_vars=n_vars, hidden_dim=128, embed_dim=256)

    X = torch.randn(B, N, n_vars)
    y = torch.randn(B, N)

    z = encoder(X, y)
    print(f"Input  X: {tuple(X.shape)}")
    print(f"Input  y: {tuple(y.shape)}")
    print(f"Output z: {tuple(z.shape)}")
    assert z.shape == (B, 256), f"Expected (4, 256), got {z.shape}"

    # Permutation invariance: shuffling points should not change the embedding
    perm = torch.randperm(N)
    z2 = encoder(X[:, perm, :], y[:, perm])
    assert torch.allclose(z, z2, atol=1e-5), "Permutation invariance failed"
    print("Permutation invariance: OK")

    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {param_count:,}")
