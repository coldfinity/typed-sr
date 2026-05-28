"""
eval/metrics.py

Evaluation metrics for symbolic regression experiments.

Two primary metrics (standard in SR literature):
  - Recovery rate : fraction of problems where NMSE < threshold
  - Complexity    : size (node count) of the best recovered expression

These are computed over a suite of FeynmanProblem runs, not individual
expressions. Use ExperimentResult to store per-problem outcomes, then
summarise with suite_summary().
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grammar"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from reward import nmse
from tree import Node


@dataclass
class ExperimentResult:
    """
    Outcome of running DSR on one FeynmanProblem.

    Attributes
    ----------
    problem_name : str
    best_expr : str
        Human-readable best expression found.
    best_node : Node or None
        Expression tree of the best candidate.
    nmse_val : float
        NMSE of best_node on the test set (inf if none found).
    complexity : int
        Node count of best_node (0 if none found).
    n_steps : int
        Training steps taken.
    """

    problem_name: str
    best_expr: str
    best_node: Node | None
    nmse_val: float
    complexity: int
    n_steps: int


def evaluate_result(
    node: Node | None,
    var_values: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> tuple[float, int]:
    """
    Compute NMSE and complexity for a candidate expression.

    Returns (nmse_val, complexity). Returns (inf, 0) if node is None or
    evaluation fails.
    """
    if node is None:
        return float("inf"), 0
    try:
        y_pred = node.evaluate(var_values)
        if not np.all(np.isfinite(y_pred)):
            return float("inf"), 0
        return nmse(y_pred, y_true), node.size
    except Exception:
        return float("inf"), 0


def recovery_rate(results: list[ExperimentResult], tol: float = 1e-4) -> float:
    """
    Fraction of problems where NMSE < tol.

    Parameters
    ----------
    results : list[ExperimentResult]
    tol : float
        NMSE threshold below which an expression counts as "recovered".

    Returns
    -------
    float in [0, 1]
    """
    if not results:
        return 0.0
    recovered = sum(1 for r in results if r.nmse_val < tol)
    return recovered / len(results)


def mean_complexity(results: list[ExperimentResult], tol: float = 1e-4) -> float:
    """
    Mean complexity (node count) of recovered expressions.

    Only counts problems that were actually recovered (NMSE < tol).
    Returns nan if nothing was recovered.
    """
    recovered = [r.complexity for r in results if r.nmse_val < tol]
    return float(np.mean(recovered)) if recovered else float("nan")


def suite_summary(
    results: list[ExperimentResult], tol: float = 1e-4
) -> dict[str, float]:
    """
    Aggregate metrics over a suite of experiment results.

    Returns a dict with:
      recovery_rate     — fraction recovered
      mean_complexity   — avg node count of recovered expressions
      mean_nmse         — avg NMSE over all problems (inf excluded as 2.0)
      n_problems        — total problems
      n_recovered       — count recovered
    """
    clipped_nmse = [min(r.nmse_val, 2.0) for r in results]
    return {
        "recovery_rate": recovery_rate(results, tol),
        "mean_complexity": mean_complexity(results, tol),
        "mean_nmse": float(np.mean(clipped_nmse)) if results else float("nan"),
        "n_problems": len(results),
        "n_recovered": sum(1 for r in results if r.nmse_val < tol),
    }


def print_summary(results: list[ExperimentResult], tol: float = 1e-4) -> None:
    """Print a formatted per-problem table plus aggregate summary."""
    summary = suite_summary(results, tol)

    print(f"{'Problem':<22} {'NMSE':>10} {'Size':>6}  Expression")
    print("-" * 70)
    for r in results:
        nmse_str = f"{r.nmse_val:.4f}" if np.isfinite(r.nmse_val) else "   inf"
        recovered = "✓" if r.nmse_val < tol else " "
        print(
            f"{recovered} {r.problem_name:<20} {nmse_str:>10} {r.complexity:>6}  {r.best_expr}"
        )

    print("-" * 70)
    print(
        f"Recovery rate : {summary['recovery_rate']:.1%}  "
        f"({summary['n_recovered']}/{summary['n_problems']})"
    )
    print(f"Mean NMSE     : {summary['mean_nmse']:.4f}")
    print(f"Mean complexity (recovered): {summary['mean_complexity']:.1f}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grammar"))
    from tree import make_const, make_op, make_var

    # Fake results: two recovered, one not
    r1 = ExperimentResult(
        problem_name="I.6.20a",
        best_expr="exp(neg(mul(x1, x1)))",
        best_node=make_op(
            "exp", make_op("neg", make_op("mul", make_var("x1"), make_var("x1")))
        ),
        nmse_val=0.00002,
        complexity=5,
        n_steps=300,
    )
    r2 = ExperimentResult(
        problem_name="I.12.1",
        best_expr="(x1 * x2)",
        best_node=make_op("mul", make_var("x1"), make_var("x2")),
        nmse_val=0.0,
        complexity=3,
        n_steps=150,
    )
    r3 = ExperimentResult(
        problem_name="I.15.10",
        best_expr="x1",
        best_node=make_var("x1"),
        nmse_val=0.85,
        complexity=1,
        n_steps=500,
    )

    print_summary([r1, r2, r3])
