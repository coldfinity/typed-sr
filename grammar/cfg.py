"""
grammar/cfg.py

Context-free grammar for symbolic regression.

The grammar defines:
  - The token vocabulary (operators + variables + constants)
  - Production rules (what tokens can appear where)
  - A stateful parser that tracks how many child slots remain open,
    used to produce a validity mask at each RNN decoding step.

Token types
-----------
  BINARY_OP  : "add", "sub", "mul", "div"
  UNARY_OP   : "sin", "cos", "exp", "log", "sqrt", "neg"
  VARIABLE   : "x1", "x2", ..., "xN"
  CONSTANT   : "c_neg1", "c_0.5", "c_1", "c_2", "c_pi", "c_e"

Grammar (prefix / pre-order)
-----------------------------
  E -> BINARY_OP E E
  E -> UNARY_OP  E
  E -> VARIABLE
  E -> CONSTANT

The RNN samples tokens one at a time in pre-order. After each token
the grammar tracks how many expression slots are still open (i.e. how
many child expressions are yet to be produced). A token is *valid* at
step t iff:

  - slots_open > 0  (we still need an expression)
  - AND if slots_open == 1 and we are at max_depth, token must be a leaf
    (prevents infinite trees)

This gives us a binary validity mask over the vocabulary at every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from tree import (
    BINARY_OPS,
    OP_ARITY,
    UNARY_OPS,
    Node,
    from_token_sequence,
    make_const,
    make_var,
)

# ---------------------------------------------------------------------------
# Constant pool
# ---------------------------------------------------------------------------

CONSTANTS: dict[str, float] = {
    "c_neg1": -1.0,
    "c_0.5": 0.5,
    "c_1": 1.0,
    "c_2": 2.0,
    "c_pi": 3.141592653589793,
    "c_e": 2.718281828459045,
}


# ---------------------------------------------------------------------------
# CFG class
# ---------------------------------------------------------------------------


@dataclass
class CFG:
    """
    Defines the token vocabulary and grammar rules for symbolic regression.

    Parameters
    ----------
    n_vars : int
        Number of input variables (x1 ... xN).
    binary_ops : list[str]
        Binary operators to include. Subset of BINARY_OPS keys.
    unary_ops : list[str]
        Unary operators to include. Subset of UNARY_OPS keys.
    constants : list[str]
        Constant tokens to include. Subset of CONSTANTS keys.
    max_depth : int
        Maximum allowed tree depth. When the current partial tree is at
        this depth, only leaf tokens are valid.
    """

    n_vars: int
    binary_ops: list[str] = field(default_factory=lambda: list(BINARY_OPS))
    unary_ops: list[str] = field(default_factory=lambda: list(UNARY_OPS))
    constants: list[str] = field(default_factory=lambda: list(CONSTANTS))
    max_depth: int = 5

    def __post_init__(self):
        # Build ordered vocabulary
        variables = [f"x{i+1}" for i in range(self.n_vars)]

        self._vocab: list[str] = (
            self.binary_ops + self.unary_ops + variables + self.constants
        )
        self._token_to_idx: dict[str, int] = {
            tok: i for i, tok in enumerate(self._vocab)
        }

        # Precompute index sets for masking
        self._binary_idx = [self._token_to_idx[t] for t in self.binary_ops]
        self._unary_idx = [self._token_to_idx[t] for t in self.unary_ops]
        self._variable_idx = [self._token_to_idx[t] for t in variables]
        self._constant_idx = [self._token_to_idx[t] for t in self.constants]
        self._leaf_idx = self._variable_idx + self._constant_idx
        self._op_idx = self._binary_idx + self._unary_idx

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> list[str]:
        return list(self._vocab)

    def token_to_idx(self, token: str) -> int:
        return self._token_to_idx[token]

    def idx_to_token(self, idx: int) -> str:
        return self._vocab[idx]

    # ------------------------------------------------------------------
    # Validity mask
    # ------------------------------------------------------------------

    def initial_state(self) -> "ParseState":
        """Return the parser state at the beginning of a sequence."""
        return ParseState(slots_open=1, depth_stack=[0], cfg=self)

    def validity_mask(self, state: "ParseState") -> np.ndarray:
        """
        Binary mask of shape (vocab_size,) where 1 means the token is
        valid to emit given the current parse state.

        Rules
        -----
        - If slots_open == 0: no token is valid (sequence is complete).
        - If current depth >= max_depth: only leaf tokens are valid.
        - Otherwise: all tokens are valid.
        """
        mask = np.zeros(self.vocab_size, dtype=np.float32)

        if state.slots_open == 0:
            return mask  # sequence complete, nothing valid

        current_depth = state.depth_stack[-1] if state.depth_stack else 0

        if current_depth >= self.max_depth:
            # Force a leaf
            for idx in self._leaf_idx:
                mask[idx] = 1.0
        else:
            # All tokens valid
            mask[:] = 1.0

        return mask

    # ------------------------------------------------------------------
    # Token evaluation helpers
    # ------------------------------------------------------------------

    def is_binary(self, token: str) -> bool:
        return token in self.binary_ops

    def is_unary(self, token: str) -> bool:
        return token in self.unary_ops

    def is_leaf(self, token: str) -> bool:
        return not (self.is_binary(token) or self.is_unary(token))

    def token_value(self, token: str) -> Optional[float]:
        """Return the numeric value of a constant token, or None."""
        return CONSTANTS.get(token, None)


# ---------------------------------------------------------------------------
# Parse state
# ---------------------------------------------------------------------------


@dataclass
class ParseState:
    """
    Tracks the state of the grammar during sequence generation.

    Attributes
    ----------
    slots_open : int
        Number of expression slots that still need to be filled.
        Starts at 1 (one full expression expected). Increases by
        (arity - 1) when an operator is emitted (it consumes one slot
        but opens arity new ones). Decreases by 1 when a leaf is emitted.
    depth_stack : list[int]
        Stack of depths for each open slot. Used to enforce max_depth.
    cfg : CFG
        Reference to the grammar, used for arity lookup.
    """

    slots_open: int
    depth_stack: list[int]
    cfg: CFG

    @property
    def is_complete(self) -> bool:
        return self.slots_open == 0

    def advance(self, token: str) -> "ParseState":
        """
        Return the new parse state after emitting `token`.

        Does not mutate self - returns a new ParseState.
        """
        if self.slots_open == 0:
            raise ValueError("Cannot advance a complete parse state.")

        current_depth = self.depth_stack[-1]
        new_depth_stack = self.depth_stack[:-1]  # pop current slot

        if self.cfg.is_binary(token):
            # Opens 2 child slots at depth + 1
            child_depth = current_depth + 1
            new_depth_stack = new_depth_stack + [child_depth, child_depth]
            new_slots = self.slots_open - 1 + 2
        elif self.cfg.is_unary(token):
            # Opens 1 child slot at depth + 1
            child_depth = current_depth + 1
            new_depth_stack = new_depth_stack + [child_depth]
            new_slots = self.slots_open - 1 + 1
        else:
            # Leaf: closes the slot
            new_slots = self.slots_open - 1

        return ParseState(
            slots_open=new_slots,
            depth_stack=new_depth_stack,
            cfg=self.cfg,
        )


# ---------------------------------------------------------------------------
# Sequence -> tree, with constant substitution
# ---------------------------------------------------------------------------


def sequence_to_tree(tokens: list[str], cfg: CFG) -> Node:
    """
    Build a Node tree from a token sequence, substituting constant
    tokens (e.g. "c_pi") with their numeric string values so that
    tree.py's evaluator can handle them.
    """
    resolved = []
    for tok in tokens:
        val = cfg.token_value(tok)
        resolved.append(str(round(val, 6)) if val is not None else tok)
    return from_token_sequence(resolved)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = CFG(n_vars=2, max_depth=4)

    print(f"Vocab size : {cfg.vocab_size}")
    print(f"Vocab      : {cfg.vocab}")
    print()

    # Simulate generating: add sin mul x1 x1 div x2 c_2
    # = sin(x1*x1) + x2/2
    token_seq = ["add", "sin", "mul", "x1", "x1", "div", "x2", "c_2"]

    state = cfg.initial_state()
    print("Step-by-step generation:")
    for tok in token_seq:
        mask = cfg.validity_mask(state)
        idx = cfg.token_to_idx(tok)
        valid = bool(mask[idx])
        print(f"  emit '{tok:8s}'  valid={valid}  slots_open={state.slots_open}")
        state = state.advance(tok)

    print(f"\nFinal state: complete={state.is_complete}, slots_open={state.slots_open}")

    # Build tree and evaluate
    tree = sequence_to_tree(token_seq, cfg)
    print(f"\nExpression : {tree}")
    print(f"Depth      : {tree.depth}")

    import numpy as np

    rng = np.random.default_rng(42)
    data = {"x1": rng.uniform(-2, 2, 5), "x2": rng.uniform(-2, 2, 5)}
    print(f"Evaluation : {tree.evaluate(data)}")

    # Check that depth enforcement works
    print("\nDepth enforcement test (max_depth=1, only leaves valid):")
    cfg_shallow = CFG(n_vars=2, max_depth=1)
    state2 = cfg_shallow.initial_state()
    mask2 = cfg_shallow.validity_mask(state2)
    for i, tok in enumerate(cfg_shallow.vocab):
        if mask2[i]:
            print(f"  valid: {tok}")
