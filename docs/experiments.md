# Experiments

Both experiments are CPU-only, seeded, and wired to the `bayesmas` CLI.

## Opinion-dynamics Monte Carlo

Four stopping / filtering protocols share the same one-shot reports and a
15-step cap:

| Protocol | Update | Stop |
|---|---|---|
| `fixed` | claimed-precision partial pooling | always 15 steps |
| `dylan` | DeGroot pull to the unweighted mean | > 2/3 cluster |
| `bayesmas` | claimed-precision partial pooling | posterior stability |
| `bayesmas-f` | same, after per-receiver SAC | posterior stability on the retained set |

Regimes (`ComplexitySetting`):

| Setting | Byzantine | Honest noise |
|---|---|---|
| `low` | 0 | small \(\sigma\), small bias |
| `medium` | 0 | moderate |
| `high` | 1, target \(75\), dishonest \(\sigma^2 = 10^{-6}\) | `bias_sd = 12` |

```bash
uv run bayesmas simulate --trials 100 --seed 0
```

Qualitative claims locked by `tests/test_simulation.py`:

- High complexity: BayesMAS-F mean error < unfiltered BayesMAS; both use
  fewer steps than Fixed (which is always 15).
- Low complexity: BayesMAS and DyLAN stop well before the cap.

These are the same claims as the Notion v2 table. Exact cell means from
that unpublished Studio run are **not** asserted.

## Apple-style puzzle sweep

Four controllable environments from *The Illusion of Thinking*
(Shojaee et al., 2025):

| Puzzle | Complexity \(N\) | Optimal length |
|---|---|---|
| Tower of Hanoi | disks | \(2^N - 1\) |
| Checker jumping | pairs | \((N+1)^2 - 1\) |
| River crossing | actor/agent pairs | BFS (boat capacity 2) |
| Blocks world | blocks | BFS (3 stacks, reverse-order goal) |

Agents are **noisy solvers**, not LLMs. Honest agents score each legal move
as \(-\)remaining-length \(+ \mathcal{N}(0,\sigma^2)\). A Byzantine agent
(when present) boosts the worst move and claims \(\sigma^2 \to 0\).
BayesMAS-F drops that agent via SAC on remaining-length reports.

```bash
uv run bayesmas puzzles --trials 20 --seed 0
```

`tests/test_planner.py` locks:

- zero-noise oracles solve Hanoi-\(N=3\) in 7 moves;
- with a Byzantine agent and noise, BayesMAS-F solves more often than
  unfiltered BayesMAS.

This tests BayesMAS **mechanics** against Apple **complexity settings**.
It does not evaluate frontier LRMs. A later loop can swap the noisy
solver for an OpenRouter / Hugging Face adapter without changing the
puzzle simulators.

## Follow-on (out of scope here)

1. LLM adapter: map a model completion to \((y_j, \sigma_j^2)\) and a
   preferred move; keep the same protocols.
2. TextMAS / LatentMAS comparison using `dpletta/latentmas-a890104f`.
3. Bit-exact replay of `stopping_simulation-v2.py` if that script is
   recovered from Studio.
