# Where green nested simulation stops paying for itself

Green nested simulation reuses Monte Carlo paths across scenarios by
reweighting them with a likelihood ratio, turning a per-scenario path budget
into the whole budget. It needs the conditional density `f`.

This reproduces the method's headline result, then measures what happens when
`f` has to be estimated instead — which is the situation in every domain that
does not come with a generative model attached.

```
  d   standard   GNS known   GNS kernel   pooled mean
  ────────────────────────────────────────────────────────
  1     0.0141     0.0002     0.0010     0.1678
  2     0.0135     0.0001     0.0030     0.1688
  4     0.0145     0.0001     0.0090     0.1785
  8     0.0141     0.0001     0.0135     0.1802
```

Integrated mean squared error against a closed form, lower is better. Budget
4,000 paths, 40 trials, seeded.

**With `f` known, GNS is ~100× better than standard nested simulation.** That is
the paper's result, reproduced.

**Estimating `f` costs a factor of five in one dimension and the entire
advantage by eight**, where the kernel version and plain nested simulation are
the same number. It still beats ignoring the conditioning altogether — so some
information survives — but nothing is left of the reason to use the method.

## Why this is the interesting number

The obvious question about applying GNS outside its home domain is "can you do
it without `f`?" That has a one-line answer: yes, you replace the likelihood
ratio with kernel similarity, and you are now doing kernel regression with a
bandwidth problem.

The useful question is **where the crossover sits** — the dimension at which
the cost of estimating the density exceeds what the reuse gains. Here it falls
between four and eight for a budget of 4,000, and it should move with budget,
bandwidth rule, and how much of the conditioning is actually informative.

Nobody has to find that crossover in the original setting, because Lee–Carter
hands you a closed-form normal.

The mechanism is ordinary curse of dimensionality, but it bites somewhere
specific: the mixture denominator `g = (1/M) Σ f(·|κᵢ)` is the thing that keeps
the likelihood ratio from exploding between distant scenarios, and the mixture
denominator is exactly what cannot be estimated reliably in moderate dimension.
The variance control and the dimensionality problem are not two limitations.
They are one limitation seen from two sides.

## The setup

Deliberately the paper's own K-call case, so the reproduction is checkable:

- A risk factor following a random walk with drift, `κ' = κ + μ + σZ`. The
  conditional density is a closed-form normal.
- A call-style payoff at maturity, `H(x) = max(x − K, 0)`.
- The target is `V(κ) = E[H | κ]` across many outer scenarios.
- `V` has an exact solution — Bachelier's formula — so error is **measured, not
  estimated**. The analytic value is itself checked against brute-force Monte
  Carlo in the tests, because an IMSE computed against a wrong closed form
  would be worthless.

One addition: the conditioning state is embedded in `d` dimensions of which
only the first drives the payoff. That is the situation you are in whenever you
have features and do not know which of them matter — you cannot drop the
irrelevant ones, because you cannot tell which they are.

Four estimators:

| | what it does |
| --- | --- |
| `standard` | classic nested simulation — each scenario keeps only its own paths |
| `GNS known` | reweight by `f(x\|κ)/g(x)`, self-normalised, mixture denominator |
| `GNS kernel` | same, with the ratio replaced by kernel similarity over the state |
| `pooled mean` | ignore the conditioning entirely — the floor |

`standard` and `GNS known` never touch the conditioning features, so their
columns are flat in `d`. That is the control: it shows the movement in the
`GNS kernel` column is the dimension doing it, not the simulation drifting.

## Running it

```bash
npm install
npm start     # prints the table
npm test      # 9 tests, including a Monte Carlo check of the closed form
```

TypeScript, no dependencies beyond `tsx`. No data files, no network.

## Reference

Feng, B. M., Li, Z., & Zhou, J. (2022). *Green nested simulation via likelihood
ratio: Applications to longevity risk management.* Insurance: Mathematics and
Economics, 106, 285–301.

The method: pool the inner paths and reweight rather than resimulate; use a
mixture sampling density rather than a pairwise ratio, so weights stay finite
between distant scenarios; self-normalise, which is biased in finite samples
but consistent and much lower variance; and under a Markov risk factor
telescope the product of density ratios down to the single step after the
horizon. Unbiased, `O(Γ⁻¹)` CLT, 200×–4,000× lower IMSE than the Taylor-style
approximations it replaces — whose bias does not shrink with budget.

## Where this came from

It fell out of asking whether the method could estimate conditional expected
returns in a stock signal desk
([lp-stock-signals](https://github.com/Matr1xOw/lp-stock-signals)). It cannot,
and the arithmetic above is why: eight conditioning features and a few thousand
samples put you at the right-hand end of that table. Written up there in
`docs/green-simulation.md`.
