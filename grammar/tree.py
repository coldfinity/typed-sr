"""
grammar/tree.py

Expression tree representation and evaluation for typed-sr.

Nodes are either:
  - Operator nodes (binary or unary) with children
  - Leaf nodes (variables or constants)

The tree can be evaluated on numpy arrays, converted to a
human-readable string, and inspected for depth/complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Token / operator definitions
# ---------------------------------------------------------------------------

BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: np.where(np.abs(b) > 1e-8, a / b, 1.0),  # protected div
}

UNARY_OPS = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": lambda x: np.exp(np.clip(x, -20, 20)),  # protected exp
    "log": lambda x: np.log(np.abs(x) + 1e-8),  # protected log
    "sqrt": lambda x: np.sqrt(np.abs(x)),  # protected sqrt
    "neg": lambda x: -x,
}

ALL_OPS = {**BINARY_OPS, **UNARY_OPS}

OP_ARITY = {op: 2 for op in BINARY_OPS} | {op: 1 for op in UNARY_OPS}

OP_STR = {
    "add": lambda a, b: f"({a} + {b})",
    "sub": lambda a, b: f"({a} - {b})",
    "mul": lambda a, b: f"({a} * {b})",
    "div": lambda a, b: f"({a} / {b})",
    "sin": lambda a: f"sin({a})",
    "cos": lambda a: f"cos({a})",
    "exp": lambda a: f"exp({a})",
    "log": lambda a: f"log({a})",
    "sqrt": lambda a: f"sqrt({a})",
    "neg": lambda a: f"(-{a})",
}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """
    A single node in an expression tree.

    Attributes
    ----------
    token : str
        Operator name (e.g. "add", "sin"), variable name (e.g. "x1"),
        or the string representation of a constant (e.g. "3.14").
    children : list[Node]
        Child nodes. Empty for leaves.
    """

    token: str
    children: list[Node] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_variable(self) -> bool:
        return self.is_leaf and self.token.startswith("x")

    @property
    def is_constant(self) -> bool:
        return self.is_leaf and not self.token.startswith("x")

    @property
    def arity(self) -> int:
        return OP_ARITY.get(self.token, 0)

    # ------------------------------------------------------------------
    # Depth / complexity
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Maximum depth of the subtree rooted here (leaf = 0)."""
        if self.is_leaf:
            return 0
        return 1 + max(c.depth for c in self.children)

    @property
    def size(self) -> int:
        """Total number of nodes in the subtree."""
        return 1 + sum(c.size for c in self.children)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, var_values: dict[str, np.ndarray]) -> np.ndarray:
        """
        Evaluate the expression on a dict of variable arrays.

        Parameters
        ----------
        var_values : dict
            Maps variable names (e.g. "x1") to numpy arrays of the
            same length.

        Returns
        -------
        np.ndarray
            Result of evaluating the expression element-wise.
        """
        if self.is_variable:
            if self.token not in var_values:
                raise KeyError(f"Variable '{self.token}' not found in var_values.")
            return np.asarray(var_values[self.token], dtype=float)

        if self.is_constant:
            # Broadcast scalar constant to match the length of any variable
            val = float(self.token)
            # Infer length from first variable in scope
            n = next(iter(var_values.values())).shape[0] if var_values else 1
            return np.full(n, val)

        # Operator node
        child_vals = [c.evaluate(var_values) for c in self.children]
        fn = ALL_OPS[self.token]
        return fn(*child_vals)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        if self.is_leaf:
            return self.token
        fmt = OP_STR[self.token]
        child_strs = [str(c) for c in self.children]
        return fmt(*child_strs)

    def __repr__(self) -> str:
        return f"Node({self.token!r}, children={self.children!r})"

    # ------------------------------------------------------------------
    # Copying
    # ------------------------------------------------------------------

    def clone(self) -> Node:
        """Return a deep copy of this subtree."""
        return Node(self.token, [c.clone() for c in self.children])


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def make_var(name: str) -> Node:
    """Create a variable leaf node, e.g. make_var('x1')."""
    return Node(name)


def make_const(value: float) -> Node:
    """Create a constant leaf node, e.g. make_const(3.14)."""
    return Node(str(round(float(value), 6)))


def make_op(op: str, *children: Node) -> Node:
    """
    Create an operator node.

    Example
    -------
    >>> make_op("add", make_var("x1"), make_const(1.0))
    """
    expected = OP_ARITY[op]
    if len(children) != expected:
        raise ValueError(
            f"Operator '{op}' expects {expected} children, got {len(children)}."
        )
    return Node(op, list(children))


# ---------------------------------------------------------------------------
# Token-sequence <-> tree conversion
# ---------------------------------------------------------------------------


def from_token_sequence(tokens: list[str]) -> Node:
    """
    Build an expression tree from a prefix (pre-order) token sequence.

    This is the format produced by the RNN sampler: tokens are written
    in pre-order traversal order, so the operator comes before its
    children.

    Parameters
    ----------
    tokens : list[str]
        Pre-order token list, e.g. ["add", "x1", "mul", "x2", "x2"].

    Returns
    -------
    Node
        Root of the constructed tree.

    Raises
    ------
    ValueError
        If the token sequence is malformed (too short or leftover tokens).
    """
    tokens = list(tokens)  # make a copy so we can pop safely

    def _parse(toks: list[str]) -> Node:
        if not toks:
            raise ValueError("Token sequence ended unexpectedly.")
        tok = toks.pop(0)
        arity = OP_ARITY.get(tok, 0)
        children = [_parse(toks) for _ in range(arity)]
        return Node(tok, children)

    root = _parse(tokens)
    if tokens:
        raise ValueError(f"Leftover tokens after parsing: {tokens}")
    return root


def to_token_sequence(node: Node) -> list[str]:
    """
    Serialise a tree to its prefix token sequence (inverse of
    from_token_sequence).
    """
    result = [node.token]
    for child in node.children:
        result.extend(to_token_sequence(child))
    return result


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Manually build: sin(x1^2) + x2/3
    # Note: no power operator yet — represent x1^2 as x1*x1
    expr = make_op(
        "add",
        make_op("sin", make_op("mul", make_var("x1"), make_var("x1"))),
        make_op("div", make_var("x2"), make_const(3.0)),
    )

    print("Expression:", expr)
    print("Depth:     ", expr.depth)
    print("Size:      ", expr.size)

    rng = np.random.default_rng(0)
    data = {"x1": rng.uniform(-2, 2, 100), "x2": rng.uniform(-2, 2, 100)}
    y = expr.evaluate(data)
    print("Output sample:", y[:5])

    # Round-trip through token sequence
    tokens = to_token_sequence(expr)
    print("Token sequence:", tokens)
    reconstructed = from_token_sequence(tokens)
    print("Reconstructed: ", reconstructed)

    y2 = reconstructed.evaluate(data)
    assert np.allclose(y, y2), "Round-trip evaluation mismatch!"
    print("Round-trip OK.")
