# BayesMAS update rules

The package implements the hierarchical-normal model in the project notes
([Notion: BayesMAS](https://app.notion.com/p/3b376a8b898e809a852ae689e4a6271d)).
This page is the executable specification: identities here are locked by
`tests/test_pooling.py`, `tests/test_stopping.py`, and `tests/test_filter.py`.

## Generative model

For \(J\) agents,

\[
\theta_j \mid \mu,\tau \sim \mathcal{N}(\mu, \tau^2),
\qquad
y_j \mid \theta_j \sim \mathcal{N}(\theta_j, \sigma_j^2).
\]

\(\mu\) is the group consensus and \(\tau^2\) is disagreement. Agents never
append raw messages. They reshape the sufficient statistics
\((\hat\mu, \bar\sigma^2, \tau^2)\).

## Consensus

\[
\hat\mu = \frac{\sum_j w_j y_j}{\sum_j w_j},
\qquad
w_j = \frac{1}{\sigma_j^2 + \tau^2}.
\]

`consensus_mean` implements this. The **claimed-precision** mean used inside
the unfiltered protocols is the same formula with \(\tau^2 = 0\), i.e.
inverse-variance weights on the *self-reported* \(\sigma_j^2\). That is the
dishonest-confidence attack surface: a Byzantine agent that reports
\(\sigma^2 \to 0\) owns \(\hat\mu\).

## Partial pooling

\[
\hat\theta_j
= \frac{(1/\sigma_j^2)\, y_j + (1/\tau^2)\, \mu}
       {1/\sigma_j^2 + 1/\tau^2}.
\]

- \(\tau^2 \to 0\): the agent collapses onto the consensus (unified group).
- \(\tau^2 \to \infty\): the agent stays on its hidden anchor \(y_j\).

`partial_pool` implements this, with \(\tau^2 = 0\) returning \(\mu\) exactly.

## Residual disagreement

The method-of-moments estimator (unbiased sample variance of the \(y_j\))

\[
\hat\tau^2 = \max\bigl(0,\; s^2(y) - \overline{\sigma^2}\bigr)
\]

is stored on `SufficientStats` as a diagnostic. The **pooling** \(\tau^2\)
inside a deliberation round is an annealing group-trust term
(`pool_tau2` decayed by `pool_tau_decay` each step). As trust grows, honest
agents are pulled harder toward \(\hat\mu\). Unfiltered, that is the
herd-effect collapse toward a Byzantine target; filtered, \(\hat\mu\) is
computed on the retained set and the same annealing just tightens an honest
consensus.

## Stopping

The posterior variance of the consensus is

\[
\mathrm{Var}(\mu \mid y) \approx \Bigl(\sum_j \tfrac{1}{\sigma_j^2 + \tau^2}\Bigr)^{-1}.
\]

`should_stop` fires after `min_steps` when either

1. \(\mathrm{Var}(\mu) \le\) `abs_threshold` (easy tasks: do not overthink), or
2. the last change in \(\mathrm{Var}(\mu)\) is \(\le\) `delta_threshold`
   (hard tasks: diminishing returns).

DyLAN instead stops when more than two-thirds of reports lie in a
tolerance ball (fractional consensus). Fixed never stops before `max_steps`.

## SAC filter (BayesMAS-F)

Receiver \(j\) scores neighbor \(i\) with a Gaussian kernel on the
*predictions*, not on claimed confidence:

\[
s_{i \to j} = \exp\Bigl( -\frac{(r_j - r_i)^2}{2\sigma_{\mathrm{eval}}^2} \Bigr).
\]

The bottom-\(F\) neighbors are dropped (W-MSR). The receiver always keeps
itself. `sac_retain_mask` implements this.

## What this is not

The original Studio script `stopping_simulation-v2.py` is not in git. The
Monte Carlo table in the Notion page is therefore a **qualitative** target
(filter isolates the adversary; BayesMAS stops earlier than Fixed; high
complexity needs more steps), not a bit-exact reproduction. LLM-backed
TextMAS / LatentMAS comparisons are follow-on work; see
[experiments.md](experiments.md).
