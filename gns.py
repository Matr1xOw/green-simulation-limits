"""Green nested simulation, and the same method with an estimated density.

Feng, B. M., Li, Z., & Zhou, J. (2022). Green nested simulation via likelihood
ratio: Applications to longevity risk management. Insurance: Mathematics and
Economics, 106, 285-301.

The problem is estimating V(k) = E[H | k] across many outer scenarios. Naive
nested simulation spends N inner paths per scenario and discards them. GNS
pools every path and reweights by a likelihood ratio f(x|k)/g(x), so each path
informs every scenario.

That reweighting needs f. The paper has it because its risk factor is a random
walk with drift. This module also implements the version you are pushed into
when there is no generative model -- kernel similarity over the conditioning
state -- so the two can be measured against each other.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

MU = 0.02
SIGMA = 1.0
STRIKE = 0.5


def payoff(x):
    """Call-style payoff, as in the paper's K-call case study."""
    return np.maximum(x - STRIKE, 0.0)


def truth(kappa):
    """The exact answer.

    X ~ N(kappa + MU, SIGMA^2), so E[max(X - K, 0)] is Bachelier's formula.
    Having this in closed form is what makes the comparison a measurement
    rather than one more estimate.
    """
    m = kappa + MU
    z = (m - STRIKE) / SIGMA
    return (m - STRIKE) * norm.cdf(z) + SIGMA * norm.pdf(z)


@dataclass
class Experiment:
    """One simulated study.

    kappa   the conditioning value that drives the payoff, per outer scenario
    noise   extra conditioning features carrying no information
    draws   pooled inner draws
    payoffs their payoffs
    source  which scenario produced each draw
    """

    kappa: np.ndarray
    noise: np.ndarray
    draws: np.ndarray
    payoffs: np.ndarray
    source: np.ndarray

    @property
    def budget(self):
        return self.draws.size


def simulate(seed, dimensions, outer, inner):
    rng = np.random.default_rng(seed)

    kappa = rng.standard_normal(outer)
    noise = rng.standard_normal((outer, dimensions - 1))

    source = np.repeat(np.arange(outer), inner)
    draws = kappa[source] + MU + SIGMA * rng.standard_normal(outer * inner)

    return Experiment(kappa, noise, draws, payoff(draws), source)


def imse(estimates, kappa):
    """Integrated mean squared error against the closed form."""
    return float(np.mean((estimates - truth(kappa)) ** 2))


def standard(e):
    """Classic nested simulation: each scenario keeps only its own paths."""
    outer = e.kappa.size
    totals = np.bincount(e.source, weights=e.payoffs, minlength=outer)
    counts = np.bincount(e.source, minlength=outer)
    return totals / counts


def _self_normalised(weights, payoffs):
    """Weighted average with weights rescaled to sum to one.

    The paper's recommended variant: biased in finite samples, consistent, and
    materially lower variance than leaving the weights raw.
    """
    totals = weights.sum(axis=1)
    return (weights @ payoffs) / np.where(totals > 0, totals, np.nan)


def gns_known(e):
    """GNS with the density known, as in the paper.

    The denominator is the equal mixture over all scenarios rather than a
    pairwise ratio against one reference. That choice is what keeps the weights
    finite when two scenarios sit far apart.
    """
    # f[i, j] = density of draw j under scenario i.
    f = norm.pdf((e.draws[None, :] - e.kappa[:, None] - MU) / SIGMA) / SIGMA
    mixture = f.mean(axis=0)
    return _self_normalised(f / mixture, e.payoffs)


def gns_kernel(e, dimensions):
    """GNS with the density estimated.

    Without a generative model the likelihood ratio has to be replaced by
    something learned from the sample, and kernel similarity over the
    conditioning state is the natural choice. The bandwidth follows the usual
    Silverman-style rate, whose exponent is where dimension does its damage.
    """
    state = np.column_stack([e.kappa, e.noise])[:, :dimensions]
    bandwidth = e.budget ** (-1.0 / (dimensions + 4))

    # Distance from every scenario to the scenario behind every pooled draw.
    gaps = state[:, None, :] - state[e.source][None, :, :]
    weights = np.exp(-0.5 * np.sum(gaps**2, axis=2) / bandwidth**2)
    return _self_normalised(weights, e.payoffs)


def effective_sample_size(e, dimensions):
    """How many draws the kernel weighting is really averaging, per scenario.

    ESS = 1 / sum(w^2) on self-normalised weights, the standard importance
    sampling diagnostic. It is the number that explains the headline result:
    reuse only helps while the weights spread across many draws, and in high
    dimension almost all the weight lands on a handful of near neighbours.
    """
    state = np.column_stack([e.kappa, e.noise])[:, :dimensions]
    bandwidth = e.budget ** (-1.0 / (dimensions + 4))
    gaps = state[:, None, :] - state[e.source][None, :, :]
    weights = np.exp(-0.5 * np.sum(gaps**2, axis=2) / bandwidth**2)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return float(np.median(1.0 / np.sum(weights**2, axis=1)))


def pooled(e):
    """The floor: ignore the conditioning entirely."""
    return np.full(e.kappa.size, e.payoffs.mean())


ESTIMATORS = {
    "standard": lambda e, d: standard(e),
    "GNS known": lambda e, d: gns_known(e),
    "GNS kernel": lambda e, d: gns_kernel(e, d),
    "pooled mean": lambda e, d: pooled(e),
}


def study(dimensions, outer=200, inner=20, trials=40, seed=1000):
    """Mean IMSE per estimator at one conditioning dimension."""
    totals = {name: [] for name in ESTIMATORS}
    ess = []
    for trial in range(trials):
        e = simulate(seed + trial * 7, dimensions, outer, inner)
        for name, estimate in ESTIMATORS.items():
            totals[name].append(imse(estimate(e, dimensions), e.kappa))
        ess.append(effective_sample_size(e, dimensions))

    result = {name: float(np.mean(values)) for name, values in totals.items()}
    result["kernel ESS"] = float(np.mean(ess))
    return result
