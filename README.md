# BayesMAS

Empirical Bayes collaboration for multi-agent systems. Agents exchange a compact
set of **sufficient statistics** — consensus \(\hat\mu\), local variance
\(\sigma^2\), and disagreement \(\tau^2\) — instead of growing text or KV-cache
context. Posterior variance of the group mean is the stopping signal.

This repository is a CPU-only research package: the update rules, a
receiver-side Byzantine filter, four deliberation protocols, and Apple-style
puzzle simulators with noisy planners. Live LLM / LatentMAS comparisons are
documented as follow-on work.

## Install

```bash
uv sync --all-groups
```

Python 3.12+ is required (`uv` creates `.venv`).

## Test and lint

```bash
make test
make lint
make typecheck
```

## Experiments

Opinion-dynamics Monte Carlo (Fixed, DyLAN, BayesMAS, BayesMAS-F):

```bash
uv run bayesmas simulate --trials 100 --seed 0
```

Apple-style puzzle complexity sweep (Tower of Hanoi, checker jumping, river
crossing, blocks world) with noisy solvers — no GPU or API key:

```bash
uv run bayesmas puzzles --trials 20 --seed 0
```

## Protocols

| Protocol | Communication | Stop rule |
|---|---|---|
| Fixed | scalar reports | hard cap |
| DyLAN | scalar reports | > 2/3 of agents agree |
| BayesMAS | \((\hat\mu,\sigma^2,\tau^2)\) | posterior-stability threshold |
| BayesMAS-F | same, after SAC filter | same, on the retained set |

## Documentation

- [docs/math.md](docs/math.md) — update rules, stopping, SAC filter.
- [docs/experiments.md](docs/experiments.md) — Monte Carlo regimes and
  Apple-style puzzle sweep; LLM / LatentMAS follow-on.

## License

MIT. See [LICENSE](LICENSE).
