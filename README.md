# typed-sr

Symbolic regression via grammar-guided deep reinforcement learning,
with a typed CFG that enforces dimensional consistency.

## Motivation

Standard symbolic regression systems treat the expression grammar as
unconstrained — any syntactically valid tree is a candidate regardless
of whether it makes physical sense. This project adds a **typed CFG**
where each production rule carries dimensional type signatures (length,
time, mass, etc.), so the search never wastes capacity on expressions
like `sin(velocity + mass)`.

## Research Question

Does enforcing dimensional consistency via a typed CFG improve symbolic
regression on physics datasets, compared to an untyped baseline?

## Architecture

```
Data (X, y)
    │
    ▼
Context Encoder       small MLP: (X, y) → embedding z
    │
    ▼
RNN Sampler           LSTM conditioned on z, samples token sequences
    │  ↑ grammar mask (valid tokens only at each step)
    ▼
Expression Evaluator  builds tree, evaluates on X, computes reward
    │
    ▼
REINFORCE Update      policy gradient + entropy bonus
```

## Project Structure

```
typed-sr/
├── grammar/
│   ├── cfg.py          # production rules, token vocab, validity mask
│   ├── typed_cfg.py    # dimension type system + typed masks
│   └── tree.py         # expression tree, evaluator
├── model/
│   ├── encoder.py      # dataset → embedding
│   ├── rnn.py          # LSTM sampler
│   └── dsr.py          # full DSR loop
├── train/
│   ├── reinforce.py    # policy gradient, entropy bonus
│   └── reward.py       # NMSE, complexity penalty, Pareto frontier
├── eval/
│   ├── feynman.py      # Feynman benchmark loader
│   └── metrics.py      # recovery rate, complexity
└── experiments/
    ├── baseline.py     # untyped DSR
    └── typed.py        # typed DSR
```

## Roadmap

- [x] Expression tree + evaluator (`tree.py`)
- [x] Untyped CFG + validity mask (`cfg.py`)
- [x] Reward function + Pareto frontier (`reward.py`)
- [ ] LSTM sampler (`rnn.py`)
- [ ] REINFORCE training loop (`reinforce.py`)
- [ ] Typed CFG extension (`typed_cfg.py`)
- [ ] Feynman benchmark (`feynman.py`)
- [ ] Experiments + results

## References

- Petersen et al. 2021 — [Deep Symbolic Regression](https://arxiv.org/abs/1912.04871)
- Udrescu & Tegmark 2020 — [AI Feynman](https://arxiv.org/abs/1905.11819)
- Landajuela et al. 2022 — [Unified Training of SR](https://arxiv.org/abs/2205.13548)
- Lample & Charton 2019 — [Deep Learning for Symbolic Math](https://arxiv.org/abs/1912.01412)
