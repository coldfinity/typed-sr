"""
grammar/typed_cfg.py

Typed context-free grammar for symbolic regression.

Extends cfg.py with SI dimension types so the validity mask rules out
tokens that would produce a dimensional type error. The search never
proposes expressions like sin(velocity + mass).

Dimension representation
------------------------
  Dim = tuple[int, int, int]   — (L, T, M) exponent triple
  (1, 0, 0) = length
  (0, 1, 0) = time
  (0, 0, 1) = mass
  (1, -1, 0) = velocity (L / T)
  (0, 0, 0) = dimensionless

Type rules per operator
-----------------------
  add(a, b) → D   : type(a) = type(b) = D
  sub(a, b) → D   : type(a) = type(b) = D
  mul(a, b) → D   : type(a) + type(b) = D  (component-wise)
  div(a, b) → D   : type(a) - type(b) = D  (component-wise)
  neg(a)    → D   : type(a) = D
  sin/cos/exp/log : type(a) = dimensionless, result = dimensionless
  sqrt(a)   → D   : type(a) = 2*D  (all exponents even → integer result)
  variable xk     : valid iff var_dims[k] == expected type
  constant        : valid iff expected type == dimensionless
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cfg import CFG, ParseState

# ---------------------------------------------------------------------------
# Dimension algebra
# ---------------------------------------------------------------------------

Dim = tuple[int, int, int]

DIMENSIONLESS: Dim = (0, 0, 0)


def dim_add(a: Dim, b: Dim) -> Dim:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def dim_sub(a: Dim, b: Dim) -> Dim:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dim_scale(d: Dim, k: int) -> Dim:
    return (d[0] * k, d[1] * k, d[2] * k)


def dim_halve(d: Dim) -> Optional[Dim]:
    """Return d / 2 if all exponents are even, else None."""
    if all(x % 2 == 0 for x in d):
        return (d[0] // 2, d[1] // 2, d[2] // 2)
    return None


# ---------------------------------------------------------------------------
# Type vocabulary
# ---------------------------------------------------------------------------


def compute_type_vocab(seed_dims: list[Dim], max_exp: int = 3) -> frozenset[Dim]:
    """
    Generate the finite set of dimensions reachable from seed_dims via
    multiplication and division, bounded by |exponent| <= max_exp.

    Also includes halved dimensions (needed as sqrt inputs).
    """
    vocab: set[Dim] = {DIMENSIONLESS} | set(seed_dims)

    changed = True
    while changed:
        changed = False
        additions: set[Dim] = set()
        for d1 in vocab:
            for d2 in vocab:
                for candidate in (dim_add(d1, d2), dim_sub(d1, d2)):
                    if (
                        candidate not in vocab
                        and all(abs(x) <= max_exp for x in candidate)
                    ):
                        additions.add(candidate)
        if additions:
            vocab |= additions
            changed = True

    # Include 2*d for every d, so sqrt(2*d) → d is available
    doubles = {dim_scale(d, 2) for d in vocab if all(abs(x) * 2 <= max_exp for x in d)}
    vocab |= doubles

    return frozenset(vocab)


# ---------------------------------------------------------------------------
# Typed CFG
# ---------------------------------------------------------------------------


class TypedCFG(CFG):
    """
    CFG extended with dimensional type checking.

    Parameters
    ----------
    n_vars : int
    var_dims : list[Dim]
        Dimension of each variable x1 ... xN.
    target_dim : Dim
        Expected output dimension of the expression.
    max_exp : int
        Bounds the generated type vocabulary (|exponent| <= max_exp).
    All other CFG parameters are forwarded unchanged.
    """

    def __init__(
        self,
        n_vars: int,
        var_dims: list[Dim],
        target_dim: Dim,
        max_exp: int = 3,
        **cfg_kwargs,
    ):
        if len(var_dims) != n_vars:
            raise ValueError(f"var_dims length {len(var_dims)} != n_vars {n_vars}")
        super().__init__(n_vars=n_vars, **cfg_kwargs)
        self.var_dims: list[Dim] = var_dims
        self.target_dim: Dim = target_dim
        self.type_vocab: frozenset[Dim] = compute_type_vocab(
            var_dims + [target_dim], max_exp=max_exp
        )

    # ------------------------------------------------------------------
    # Precomputed type split tables (cached per expected type)
    # ------------------------------------------------------------------

    def _mul_splits(self, d_target: Dim) -> list[tuple[Dim, Dim]]:
        """All (left, right) in type_vocab² where left + right = d_target."""
        return [
            (d_left, dim_sub(d_target, d_left))
            for d_left in self.type_vocab
            if dim_sub(d_target, d_left) in self.type_vocab
        ]

    def _div_splits(self, d_target: Dim) -> list[tuple[Dim, Dim]]:
        """All (left, right) in type_vocab² where left - right = d_target."""
        return [
            (dim_add(d_target, d_right), d_right)
            for d_right in self.type_vocab
            if dim_add(d_target, d_right) in self.type_vocab
        ]

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def initial_state(self) -> "TypedParseState":
        return TypedParseState(
            slots_open=1,
            depth_stack=[0],
            type_stack=[self.target_dim],
            cfg=self,
        )

    def validity_mask(self, state: "TypedParseState") -> np.ndarray:  # type: ignore[override]
        """
        Binary mask of shape (vocab_size,) combining depth and type constraints.

        A token is valid iff:
          1. slots_open > 0
          2. depth constraint (same as untyped CFG)
          3. type constraint (new): the token is compatible with the expected
             dimension at the top of the type stack.
        """
        mask = np.zeros(self.vocab_size, dtype=np.float32)

        if state.slots_open == 0:
            return mask

        expected = state.type_stack[-1]
        current_depth = state.depth_stack[-1] if state.depth_stack else 0
        at_max_depth = current_depth >= self.max_depth

        for i, token in enumerate(self._vocab):
            # Depth constraint: at max depth only leaves are allowed
            if at_max_depth and not self.is_leaf(token):
                continue

            # Type constraint
            if token in ("add", "sub"):
                valid = True  # any expected type works; children get same type
            elif token == "mul":
                valid = bool(self._mul_splits(expected))
            elif token == "div":
                valid = bool(self._div_splits(expected))
            elif token == "neg":
                valid = True  # type is preserved
            elif token in ("sin", "cos", "exp", "log"):
                valid = expected == DIMENSIONLESS
            elif token == "sqrt":
                double = dim_scale(expected, 2)
                valid = double in self.type_vocab
            elif token.startswith("x"):
                # Variable: index is 1-based in token name
                k = int(token[1:]) - 1
                valid = self.var_dims[k] == expected
            else:
                # Constant: always dimensionless
                valid = expected == DIMENSIONLESS

            if valid:
                mask[i] = 1.0

        return mask


# ---------------------------------------------------------------------------
# Typed parse state
# ---------------------------------------------------------------------------


@dataclass
class TypedParseState(ParseState):
    """
    ParseState extended with a type stack.

    type_stack[-1] is the expected dimension for the next token/subtree.
    The stack grows and shrinks in lock-step with depth_stack.
    """

    type_stack: list[Dim] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.slots_open == 0

    def advance(self, token: str) -> "TypedParseState":  # type: ignore[override]
        if self.slots_open == 0:
            raise ValueError("Cannot advance a complete parse state.")

        cfg: TypedCFG = self.cfg  # type: ignore[assignment]

        current_depth = self.depth_stack[-1]
        expected = self.type_stack[-1]

        new_depth_stack = self.depth_stack[:-1]
        new_type_stack = self.type_stack[:-1]

        if cfg.is_binary(token):
            child_depth = current_depth + 1
            new_slots = self.slots_open + 1  # -1 + 2

            if token in ("add", "sub"):
                left_type = right_type = expected
            elif token == "mul":
                splits = cfg._mul_splits(expected)
                left_type, right_type = splits[0]
            else:  # div
                splits = cfg._div_splits(expected)
                left_type, right_type = splits[0]

            # Push right first so left is on top (processed first in pre-order)
            new_depth_stack = new_depth_stack + [child_depth, child_depth]
            new_type_stack = new_type_stack + [right_type, left_type]

        elif cfg.is_unary(token):
            child_depth = current_depth + 1
            new_slots = self.slots_open  # -1 + 1

            if token in ("sin", "cos", "exp", "log"):
                child_type = DIMENSIONLESS
            elif token == "sqrt":
                child_type = dim_scale(expected, 2)
            else:  # neg
                child_type = expected

            new_depth_stack = new_depth_stack + [child_depth]
            new_type_stack = new_type_stack + [child_type]

        else:  # leaf
            new_slots = self.slots_open - 1

        return TypedParseState(
            slots_open=new_slots,
            depth_stack=new_depth_stack,
            type_stack=new_type_stack,
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Problem: recover  F = x1 * x2  where x1: force (M·L/T²), x2: time (T)
    # Target: M·L/T  (momentum)
    L = (1, 0, 0)
    T = (0, 1, 0)
    M = (0, 0, 1)
    MLT2 = dim_sub(dim_add(M, L), dim_scale(T, 2))  # (1, -2, 1) force
    ML_T = dim_sub(dim_add(M, L), T)                # (1, -1, 1) momentum

    cfg = TypedCFG(
        n_vars=2,
        var_dims=[MLT2, T],   # x1: force, x2: time
        target_dim=ML_T,       # target: momentum
        max_depth=4,
    )

    print(f"Vocab size    : {cfg.vocab_size}")
    print(f"Type vocab    : {len(cfg.type_vocab)} distinct dimensions")
    print(f"Target dim    : {cfg.target_dim}")
    print()

    # Valid expression: mul x1 x2  (force * time = momentum)
    valid_seq = ["mul", "x1", "x2"]

    # Invalid expression: add x1 x2  (force + time — type error at x2)
    # add requires both children to have target type ML_T,
    # but x1 is force (MLT2) not momentum (ML_T).
    # So x1 itself will be invalid.

    state = cfg.initial_state()
    print("Valid sequence (mul x1 x2):")
    for tok in valid_seq:
        mask = cfg.validity_mask(state)
        idx = cfg.token_to_idx(tok)
        print(f"  emit '{tok:6s}'  valid={bool(mask[idx])}  expected_type={state.type_stack[-1]}")
        state = state.advance(tok)
    print(f"  complete={state.is_complete}\n")

    # Confirm x2 (time) is blocked when the expected type is force
    state2 = cfg.initial_state()
    state2 = state2.advance("add")  # now expects two momentum children
    state2 = state2.advance("x1")   # x1 (force) should be INVALID here
    # Actually add makes both children expect ML_T (momentum)
    # x1 is force MLT2, not momentum ML_T → should be invalid
    # Let's check

    state_after_add = cfg.initial_state()
    state_after_add = state_after_add.advance("add")
    mask_after_add = cfg.validity_mask(state_after_add)
    x1_idx = cfg.token_to_idx("x1")
    x2_idx = cfg.token_to_idx("x2")
    print("After 'add' (both children must be momentum ML_T):")
    print(f"  x1 (force)  valid={bool(mask_after_add[x1_idx])}  (expect False)")
    print(f"  x2 (time)   valid={bool(mask_after_add[x2_idx])}  (expect False)")
    print()

    # mul is valid from initial state (force * time splits exist)
    state0 = cfg.initial_state()
    mask0 = cfg.validity_mask(state0)
    mul_idx = cfg.token_to_idx("mul")
    add_idx = cfg.token_to_idx("add")
    print("From initial state (expecting momentum):")
    print(f"  mul valid={bool(mask0[mul_idx])}  (expect True — force * time = momentum)")
    print(f"  add valid={bool(mask0[add_idx])}  (expect True — momentum + momentum)")
