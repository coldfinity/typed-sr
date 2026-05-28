"""
experiments/typed.py

Typed DSR experiment (core research contribution).

Identical to baseline.py but uses TypedCFG, which enforces dimensional
consistency during expression search. The typed validity mask zeroes out
tokens that would produce a dimensional type error at the current parse
position.

Usage
-----
  python experiments/typed.py [--problems I.6.20a I.12.1 ...]
                               [--n-samples 200]
                               [--n-steps 1000]
                               [--batch-size 256]
                               [--seed 0]

NOTE: Requires grammar/typed_cfg.py to expose a TypedCFG class with the
same interface as CFG. Run baseline.py first to validate the untyped
pipeline before implementing TypedCFG.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grammar"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from dsr import DSR, DSRConfig
from feynman import PROBLEMS
from feynman import get as get_problem
from metrics import ExperimentResult, evaluate_result, print_summary

try:
    from typed_cfg import TypedCFG

    _TYPED_CFG_AVAILABLE = True
except ImportError:
    _TYPED_CFG_AVAILABLE = False


def run(
    problem_names: list[str] | None = None,
    n_samples: int = 200,
    n_steps: int = 1000,
    batch_size: int = 256,
    seed: int = 0,
) -> list[ExperimentResult]:
    if not _TYPED_CFG_AVAILABLE:
        raise RuntimeError(
            "TypedCFG is not implemented yet. "
            "Implement grammar/typed_cfg.py before running this experiment."
        )

    problems = [get_problem(n) for n in problem_names] if problem_names else PROBLEMS

    results = []
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem.name}  ({problem.formula})")
        print(f"{'='*60}")

        np.random.seed(seed)
        X, y = problem.generate(n_samples=n_samples, seed=seed)
        var_values = {f"x{i+1}": X[:, i] for i in range(X.shape[1])}

        cfg = TypedCFG(
            n_vars=problem.n_vars,
            var_dims=problem.var_dims,
            target_dim=problem.target_dim,
            max_depth=5,
        )
        dsr_config = DSRConfig(batch_size=batch_size, max_len=30)
        dsr = DSR(cfg, n_vars=problem.n_vars, config=dsr_config)

        best = dsr.fit(X, y, n_steps=n_steps, log_every=200)

        best_node = best.node if best else None
        best_expr = best.expression if best else "none"
        nmse_val, complexity = evaluate_result(best_node, var_values, y)

        result = ExperimentResult(
            problem_name=problem.name,
            best_expr=best_expr,
            best_node=best_node,
            nmse_val=nmse_val,
            complexity=complexity,
            n_steps=n_steps,
        )
        results.append(result)

    print(f"\n{'='*60}")
    print("TYPED DSR RESULTS")
    print(f"{'='*60}\n")
    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problems",
        nargs="*",
        default=None,
        help="Problem names to run (default: all)",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(
        problem_names=args.problems,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
