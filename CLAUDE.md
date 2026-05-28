# CLAUDE.md

This file provides context for AI assistants working on typed-sr.

## Project Summary

typed-sr is a research implementation of Deep Symbolic Regression (DSR)
extended with a typed context-free grammar (CFG) that enforces dimensional
consistency. The core research question is whether constraining the expression
search space using physical unit types improves equation recovery on physics
benchmarks (AI Feynman dataset).

## Architecture Overview

```
Data (X, y)
    ³
    
Context Encoder       small MLP: (X, y)  embedding z
    ³
    
RNN Sampler           LSTM conditioned on z, samples token sequences
    ³   grammar mask (valid tokens only at each step)
    
Expression Evaluator  builds tree, evaluates on X, computes reward
    ³
    
REINFORCE Update      policy gradient + entropy bonus
```

## Project Structure

```
typed-sr/
ÃÄÄ grammar/
³   ÃÄÄ cfg.py          # production rules, token vocab, validity mask
³   ÃÄÄ typed_cfg.py    # dimension type system + typed masks
³   ÀÄÄ tree.py         # expression tree, evaluator
ÃÄÄ model/
³   ÃÄÄ encoder.py      # dataset  embedding
³   ÃÄÄ rnn.py          # LSTM sampler
³   ÀÄÄ dsr.py          # full DSR loop
ÃÄÄ train/
³   ÃÄÄ reinforce.py    # policy gradient, entropy bonus
³   ÀÄÄ reward.py       # NMSE, complexity penalty, Pareto frontier
ÃÄÄ eval/
³   ÃÄÄ feynman.py      # Feynman benchmark loader
³   ÀÄÄ metrics.py      # recovery rate, complexity
ÀÄÄ experiments/
    ÃÄÄ baseline.py     # untyped DSR
    ÀÄÄ typed.py        # typed DSR
```

## Key Design Decisions

**Token representation** - expressions are serialised as prefix (pre-order)
token sequences. The RNN samples one token at a time left-to-right. This
maps cleanly onto the CFG grammar rules.

**Grammar masking** - at each decoding step, `cfg.validity_mask(state)`
returns a binary mask over the vocabulary. This is applied to RNN logits
before softmax, making it impossible to sample grammatically invalid
expressions. `ParseState` tracks open slots and depth to enforce `max_depth`.

**Protected operations** - division, log, sqrt, and exp are numerically
guarded in `tree.py` to prevent NaN/Inf during reward computation. Never
remove these guards - the RNN samples many unstable expressions early in
training.

**Reward** - `R = max(0, 1 - NMSE) - lambda * size`, clamped to [0, 1].
`lambda_complexity` defaults to 0.001. NMSE is normalised by `Var(y_true)`
so reward is scale-invariant across datasets.

**Pareto frontier** - `ParetoFrontier` in `reward.py` tracks the
non-dominated set of expressions w.r.t. accuracy (maximise) and size
(minimise). Always update this during search, not just the best reward.

**Typed CFG** - `typed_cfg.py` extends `cfg.py` with SI dimension types
(L, T, M and products/quotients thereof). Each production rule has a type
signature. The validity mask zeroes out tokens that would cause a type error
at the current parse position. This is the core research contribution -
implement and validate the untyped baseline fully before touching this.

## Implementation Status

- [x] `grammar/tree.py` - expression tree, evaluator, token sequence I/O
- [x] `grammar/cfg.py` - untyped CFG, ParseState, validity mask
- [x] `train/reward.py` - NMSE, complexity penalty, Pareto frontier
- [ ] `model/encoder.py` - dataset context encoder (MLP)
- [ ] `model/rnn.py` - LSTM sampler with grammar masking
- [ ] `model/dsr.py` - full DSR training loop
- [ ] `train/reinforce.py` - REINFORCE with baseline + entropy bonus
- [ ] `grammar/typed_cfg.py` - dimensional type system
- [ ] `eval/feynman.py` - AI Feynman benchmark loader
- [ ] `eval/metrics.py` - recovery rate, expression complexity metrics
- [ ] `experiments/baseline.py` - untyped DSR experiment
- [ ] `experiments/typed.py` - typed DSR experiment

## Conventions

- Python 3.11+, PyTorch for model components, NumPy for expression eval
- All files have a `if __name__ == "__main__":` smoke test at the bottom
- Functions are pure where possible - `ParseState.advance()` returns a new
  state rather than mutating; keep this pattern in new code
- Reward functions always handle exceptions and return 0.0 on failure -
  never let a bad expression crash a training run
- Variable names are always `x1, x2, ..., xN` (1-indexed)
- Constants are named `c_*` (e.g. `c_pi`, `c_2`) in token sequences

## Common Gotchas

- `from_token_sequence` consumes a list in-place - always pass a copy if
  you need the original
- `validity_mask` returns all-zeros when `slots_open == 0` - the caller
  must check `state.is_complete` before sampling the next token
- REINFORCE is unstable early in training - if rewards are all zero for
  many batches, first check that the grammar mask is applied correctly
  (logits, not probabilities) and that `max_depth` isn't too small
- Do not implement `typed_cfg.py` until the untyped baseline recovers
  simple expressions like `x1**2 + x2` reliably

## References

- Petersen et al. 2021 - Deep Symbolic Regression (core method)
- Udrescu & Tegmark 2020 - AI Feynman (benchmark + dimensional analysis)
- Landajuela et al. 2022 - Unified Training of SR (extensions)
- Lample & Charton 2019 - Deep Learning for Symbolic Math (transformer alt)
