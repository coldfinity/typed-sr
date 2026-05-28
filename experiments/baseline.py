"""
experiments/baseline.py

Untyped DSR baseline experiment.

Runs the standard (untyped) CFG on each Feynman benchmark problem and
reports recovery rate and complexity. This is the control condition for
the typed CFG comparison.

Usage
-----
  python experiments/baseline.py [--problems I.6.20a I.12.1 ...]
                                  [--n-samples 200]
                                  [--n-steps 1000]
                                  [--batch-size 256]
                                  [--seed 0]
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

from cfg import CFG
from dsr import DSR, DSRConfig
from feynman import PROBLEMS
from feynman import get as get_problem
from metrics import ExperimentResult, evaluate_result, print_summary


def run(
    problem_names: list[str] | None = None,
    n_samples: int = 200,
    n_steps: int = 1000,
    batch_size: int = 256,
    seed: int = 0,
) -> list[ExperimentResult]:
    problems = [get_problem(n) for n in problem_names] if problem_names else PROBLEMS

    results = []
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem.name}  ({problem.formula})")
        print(f"{'='*60}")

        np.random.seed(seed)
        X, y = problem.generate(n_samples=n_samples, seed=seed)
        var_values = {f"x{i+1}": X[:, i] for i in range(X.shape[1])}

        cfg = CFG(n_vars=problem.n_vars, max_depth=5)
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
    print("BASELINE RESULTS")
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
