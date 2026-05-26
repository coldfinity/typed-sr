"""
grammar/reward.py

Reward computation for typed-sr.

The reward signal guides REINFORCE training. It combines:
  1. Normalised MSE (NMSE) - how well the expression fits the data
  2. Complexity penalty  - penalises deep / large trees (Occam's razor)

Final reward is in [0, 1], where 1 is a perfect fit with zero complexity.

Pareto tracking
---------------
During a search run we maintain the Pareto frontier of expressions
trading off accuracy vs complexity. `ParetoFrontier` handles this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from tree import Node

# ---------------------------------------------------------------------------
# Core reward
# ---------------------------------------------------------------------------


def nmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Normalised Mean Squared Error.

        NMSE = MSE(y_pred, y_true) / Var(y_true)

    Returns a non-negative float. Values > 1 mean the expression is
    worse than predicting the mean. Clamped to [0, 2] for stability.
    """
    var = np.var(y_true)
    if var < 1e-8:
        # Target is essentially constant - reward 1 if we match it
        return 0.0 if np.mean((y_pred - y_true) ** 2) < 1e-8 else 1.0
    return float(np.clip(np.mean((y_pred - y_true) ** 2) / var, 0.0, 2.0))


def complexity(node: Node) -> int:
    """
    Complexity of an expression tree.

    Uses tree *size* (total node count) rather than depth, because size
    better captures how many operations the expression performs.
    """
    return node.size


def compute_reward(
    node: Node,
    var_values: dict[str, np.ndarray],
    y_true: np.ndarray,
    lambda_complexity: float = 0.001,
) -> float:
    """
    Compute the scalar reward for an expression tree.

    Parameters
    ----------
    node : Node
        The expression tree to evaluate.
    var_values : dict
        Input variable arrays.
    y_true : np.ndarray
        Target values.
    lambda_complexity : float
        Weight on the complexity penalty. Larger values push the search
        toward simpler expressions.

    Returns
    -------
    float
        Reward in [0, 1]. Higher is better.

    Notes
    -----
    Reward formula:

        R = max(0, 1 - NMSE) - lambda * size

    The `max(0, ...)` clamps negative accuracy to 0 so the complexity
    penalty doesn't push reward below zero.
    """
    try:
        y_pred = node.evaluate(var_values)
    except Exception:
        # Malformed or numerically unstable expression
        return 0.0

    if not np.all(np.isfinite(y_pred)):
        return 0.0

    accuracy = max(0.0, 1.0 - nmse(y_pred, y_true))
    penalty = lambda_complexity * complexity(node)
    reward = max(0.0, accuracy - penalty)
    return float(reward)


def is_perfect(
    node: Node,
    var_values: dict[str, np.ndarray],
    y_true: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    """
    Return True if the expression recovers y_true within tolerance.

    Used to decide whether to stop the search early.
    """
    try:
        y_pred = node.evaluate(var_values)
    except Exception:
        return False
    if not np.all(np.isfinite(y_pred)):
        return False
    return bool(nmse(y_pred, y_true) < tol)


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


@dataclass
class ParetoEntry:
    """One point on the Pareto frontier."""

    expression: str  # human-readable string
    node: Node  # expression tree
    accuracy: float  # 1 - NMSE, in [0, 1]
    size: int  # number of nodes
    reward: float  # combined reward


@dataclass
class ParetoFrontier:
    """
    Maintains the Pareto-optimal set of expressions w.r.t.
    accuracy (maximise) and size (minimise).

    An expression A dominates B iff:
        A.accuracy >= B.accuracy  AND  A.size <= B.size
    with at least one strict inequality.
    """

    entries: list[ParetoEntry] = field(default_factory=list)

    def update(
        self,
        node: Node,
        var_values: dict[str, np.ndarray],
        y_true: np.ndarray,
        lambda_complexity: float = 0.001,
    ) -> bool:
        """
        Attempt to add `node` to the frontier.

        Returns True if the expression was added (i.e. it is
        non-dominated by any existing entry).
        """
        try:
            y_pred = node.evaluate(var_values)
        except Exception:
            return False
        if not np.all(np.isfinite(y_pred)):
            return False

        acc = float(max(0.0, 1.0 - nmse(y_pred, y_true)))
        size = complexity(node)
        rew = compute_reward(node, var_values, y_true, lambda_complexity)

        # Check if any existing entry dominates this one
        for e in self.entries:
            if e.accuracy >= acc and e.size <= size:
                return False  # dominated

        # Remove entries that this new one dominates
        self.entries = [
            e for e in self.entries if not (acc >= e.accuracy and size <= e.size)
        ]

        self.entries.append(
            ParetoEntry(
                expression=str(node),
                node=node,
                accuracy=acc,
                size=size,
                reward=rew,
            )
        )
        # Keep sorted by size for readability
        self.entries.sort(key=lambda e: e.size)
        return True

    def best(self) -> Optional[ParetoEntry]:
        """Return the entry with the highest reward."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.reward)

    def print(self) -> None:
        print(f"{'Size':>6}  {'Accuracy':>10}  {'Reward':>8}  Expression")
        print("-" * 70)
        for e in self.entries:
            print(f"{e.size:>6}  {e.accuracy:>10.4f}  {e.reward:>8.4f}  {e.expression}")


# ---------------------------------------------------------------------------
# Batch reward (for evaluating a population of expressions)
# ---------------------------------------------------------------------------


def batch_reward(
    nodes: list[Node],
    var_values: dict[str, np.ndarray],
    y_true: np.ndarray,
    lambda_complexity: float = 0.001,
) -> np.ndarray:
    """
    Compute rewards for a list of expression trees.

    Returns
    -------
    np.ndarray of shape (len(nodes),)
    """
    return np.array(
        [compute_reward(n, var_values, y_true, lambda_complexity) for n in nodes],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from tree import make_const, make_op, make_var

    rng = np.random.default_rng(0)
    x1 = rng.uniform(-3, 3, 200)
    x2 = rng.uniform(-3, 3, 200)
    var_values = {"x1": x1, "x2": x2}

    # True expression: sin(x1^2) + x2/2
    true_expr = make_op(
        "add",
        make_op("sin", make_op("mul", make_var("x1"), make_var("x1"))),
        make_op("div", make_var("x2"), make_const(2.0)),
    )
    y_true = true_expr.evaluate(var_values)

    print("=== Reward tests ===\n")

    # Perfect prediction
    r = compute_reward(true_expr, var_values, y_true)
    print(f"True expression  : reward={r:.4f}  (expected ~1.0)")

    # Slightly wrong
    wrong = make_op(
        "add",
        make_op("sin", make_op("mul", make_var("x1"), make_var("x1"))),
        make_op("div", make_var("x2"), make_const(3.0)),
    )
    r2 = compute_reward(wrong, var_values, y_true)
    print(f"Wrong constant   : reward={r2:.4f}  (expected <1.0)")

    # Terrible expression
    terrible = make_var("x1")
    r3 = compute_reward(terrible, var_values, y_true)
    print(f"Just x1          : reward={r3:.4f}  (expected ~0)")

    print(f"\nis_perfect (true expr): {is_perfect(true_expr, var_values, y_true)}")
    print(f"is_perfect (wrong)    : {is_perfect(wrong, var_values, y_true)}")

    print("\n=== Pareto frontier ===\n")
    frontier = ParetoFrontier()

    candidates = [
        make_var("x1"),
        make_op("mul", make_var("x1"), make_var("x2")),
        wrong,
        true_expr,
    ]
    for node in candidates:
        added = frontier.update(node, var_values, y_true)
        print(f"  {'Added  ' if added else 'Ignored'}: {node}")

    print()
    frontier.print()

    print(f"\nBest: {frontier.best().expression}")
