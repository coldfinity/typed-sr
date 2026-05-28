"""
eval/feynman.py

Subset of the AI Feynman benchmark for symbolic regression.

Each problem is defined by:
  - A ground-truth function (numpy)
  - Variable names, ranges, and count
  - A human-readable formula string

Only equations expressible by the grammar token set are included:
  ops: add sub mul div sin cos exp log sqrt neg
  constants: -1, 0.5, 1, 2, pi, e

Reference: Udrescu & Tegmark 2020 — AI Feynman (arXiv:1905.11819)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Dim = (L, T, M) integer exponents — matches typed_cfg.Dim
Dim = tuple[int, int, int]
D: Dim = (0, 0, 0)   # dimensionless
L: Dim = (1, 0, 0)   # length
T: Dim = (0, 1, 0)   # time
M: Dim = (0, 0, 1)   # mass
iT: Dim = (0, -1, 0)  # 1/T  (angular frequency)
iL: Dim = (-1, 0, 0)  # 1/L
LiT2: Dim = (1, -2, 0)   # L/T²  (acceleration)
MLiT2: Dim = (1, -2, 1)  # M·L/T²  (force)
ML2iT2: Dim = (2, -2, 1)  # M·L²/T²  (energy)
ML2iT: Dim = (2, -1, 1)  # M·L²/T  (action / ħ)
MiT2: Dim = (0, -2, 1)   # M/T²  (spring constant)


@dataclass
class FeynmanProblem:
    """
    One symbolic regression problem from the Feynman benchmark.

    Attributes
    ----------
    name : str
        Short identifier (e.g. "I.6.20").
    formula : str
        Human-readable formula.
    n_vars : int
        Number of input variables.
    var_ranges : list[tuple[float, float]]
        Sampling range (low, high) for each variable.
    fn : Callable
        Ground-truth function. Takes n_vars positional numpy arrays,
        returns a numpy array of the same length.
    var_dims : list[Dim]
        SI dimension of each variable (for TypedCFG).
    target_dim : Dim
        SI dimension of the output (for TypedCFG).
    """

    name: str
    formula: str
    n_vars: int
    var_ranges: list[tuple[float, float]]
    fn: Callable[..., np.ndarray]
    var_dims: list[Dim]
    target_dim: Dim

    def generate(
        self, n_samples: int = 200, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample data points for this problem.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_vars)
        y : np.ndarray, shape (n_samples,)
        """
        rng = np.random.default_rng(seed)
        cols = [rng.uniform(low, high, n_samples) for low, high in self.var_ranges]
        X = np.stack(cols, axis=1).astype(np.float32)
        y = self.fn(*cols).astype(np.float32)
        return X, y


# ---------------------------------------------------------------------------
# Problem table
# ---------------------------------------------------------------------------
# Variables are named x1, x2, ... to match the grammar token set.
# Ranges are chosen to keep outputs finite and avoid degenerate regions.

PROBLEMS: list[FeynmanProblem] = [
    FeynmanProblem(
        name="I.6.20a",
        formula="exp(-x1^2 / 2) / sqrt(2*pi)",
        n_vars=1,
        var_ranges=[(-3.0, 3.0)],
        fn=lambda x1: np.exp(-(x1**2) / 2) / np.sqrt(2 * np.pi),
        var_dims=[D],
        target_dim=D,
    ),
    FeynmanProblem(
        name="I.12.1",
        formula="x1 * x2",           # F = m * a
        n_vars=2,
        var_ranges=[(1.0, 5.0), (1.0, 5.0)],
        fn=lambda x1, x2: x1 * x2,
        var_dims=[M, LiT2],
        target_dim=MLiT2,
    ),
    FeynmanProblem(
        name="I.12.4",
        formula="x1 * x2 / x3^2",    # k = F * r / r^2  (spring/Coulomb-like)
        n_vars=3,
        var_ranges=[(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        fn=lambda x1, x2, x3: x1 * x2 / x3**2,
        var_dims=[MLiT2, L, L],
        target_dim=MiT2,
    ),
    FeynmanProblem(
        name="I.15.10",
        formula="x1 / sqrt(1 - x2^2)",   # relativistic mass; x2 = v/c
        n_vars=2,
        var_ranges=[(1.0, 5.0), (0.0, 0.9)],
        fn=lambda x1, x2: x1 / np.sqrt(1 - x2**2),
        var_dims=[M, D],
        target_dim=M,
    ),
    FeynmanProblem(
        name="I.34.8",
        formula="x1 / (x2^2 - x3^2)",    # x = a / (ω₀² - ω²)
        n_vars=3,
        var_ranges=[(1.0, 3.0), (3.0, 6.0), (1.0, 2.5)],
        fn=lambda x1, x2, x3: x1 / (x2**2 - x3**2),
        var_dims=[LiT2, iT, iT],
        target_dim=L,
    ),
    FeynmanProblem(
        name="I.34.27",
        formula="x1 * x2",           # E = ħ * ω
        n_vars=2,
        var_ranges=[(0.1, 10.0), (0.1, 10.0)],
        fn=lambda x1, x2: x1 * x2,
        var_dims=[ML2iT, iT],
        target_dim=ML2iT2,
    ),
    FeynmanProblem(
        name="I.50.26",
        formula="x1 * cos(x2 * x3)",  # x = A * cos(ω * t)
        n_vars=3,
        var_ranges=[(1.0, 3.0), (1.0, 3.0), (0.0, 2 * np.pi)],
        fn=lambda x1, x2, x3: x1 * np.cos(x2 * x3),
        var_dims=[L, iT, T],
        target_dim=L,
    ),
    FeynmanProblem(
        name="II.11.17",
        formula="x1 * exp(-x2 * x3)",  # n = n₀ * exp(-x/λ); x2 = 1/λ, x3 = x
        n_vars=3,
        var_ranges=[(1.0, 5.0), (0.1, 1.0), (0.1, 5.0)],
        fn=lambda x1, x2, x3: x1 * np.exp(-x2 * x3),
        var_dims=[D, iL, L],
        target_dim=D,
    ),
    FeynmanProblem(
        name="II.11.27",
        formula="x1 * x2 / (x3 - x2)",   # polarizability-like; x2, x3 dimensionless
        n_vars=3,
        var_ranges=[(1.0, 5.0), (1.0, 3.0), (4.0, 8.0)],
        fn=lambda x1, x2, x3: x1 * x2 / (x3 - x2),
        var_dims=[M, D, D],
        target_dim=M,
    ),
    FeynmanProblem(
        name="III.4.33",
        formula="x1 * x2 / (exp(x1 * x2 / x3) - 1)",  # Planck: ħω/(exp(ħω/kT)-1)
        n_vars=3,
        var_ranges=[(0.1, 1.0), (0.1, 1.0), (0.5, 5.0)],
        fn=lambda x1, x2, x3: (x1 * x2) / (np.exp(x1 * x2 / x3) - 1),
        var_dims=[ML2iT, iT, ML2iT2],
        target_dim=ML2iT2,
    ),
    FeynmanProblem(
        name="bonus.spring",
        formula="sqrt(x1 / x2)",   # ω = sqrt(k / m)
        n_vars=2,
        var_ranges=[(1.0, 10.0), (1.0, 10.0)],
        fn=lambda x1, x2: np.sqrt(x1 / x2),
        var_dims=[MiT2, M],
        target_dim=iT,
    ),
    FeynmanProblem(
        name="bonus.logsum",
        formula="log(x1 + x2)",   # log of dimensionless ratio
        n_vars=2,
        var_ranges=[(0.5, 5.0), (0.5, 5.0)],
        fn=lambda x1, x2: np.log(x1 + x2),
        var_dims=[D, D],
        target_dim=D,
    ),
]

_PROBLEM_INDEX: dict[str, FeynmanProblem] = {p.name: p for p in PROBLEMS}


def get(name: str) -> FeynmanProblem:
    """Return a problem by name. Raises KeyError if not found."""
    return _PROBLEM_INDEX[name]


def list_problems() -> list[str]:
    """Return the names of all available problems."""
    return [p.name for p in PROBLEMS]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Available problems ({len(PROBLEMS)}):\n")
    for p in PROBLEMS:
        X, y = p.generate(n_samples=200, seed=0)
        finite = np.all(np.isfinite(y))
        print(
            f"  {p.name:20s}  vars={p.n_vars}  "
            f"y in [{y.min():.3f}, {y.max():.3f}]  finite={finite}  "
            f"formula: {p.formula}"
        )
