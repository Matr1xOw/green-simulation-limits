"""Green nested simulation, and the same method with an estimated density.

Feng, B. M., Li, Z., & Zhou, J. (2022). Green nested simulation via likelihood
ratio: Applications to longevity risk management. IME 106, 285-301.

Estimating V(k) = E[H | k] over many outer scenarios. Naive nested simulation
spends N inner paths per scenario and discards them; GNS pools every path and
reweights by f(x|k)/g(x) so each path informs every scenario. That needs f,
which the paper has because its risk factor is a random walk with drift.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

MU = 0.02
SIGMA = 1.0
STRIKE = 0.5


def payoff(x):
    return np.maximum(x - STRIKE, 0.0)


def truth(kappa):
    """Exact V(kappa). Bachelier, not Black-Scholes: the underlying is normal."""
    m = kappa + MU
    z = (m - STRIKE) / SIGMA
    return (m - STRIKE) * norm.cdf(z) + SIGMA * norm.pdf(z)


@dataclass
class Experiment:
    kappa: np.ndarray  # conditioning value per scenario; drives the payoff
    noise: np.ndarray  # extra conditioning features carrying no information
    draws: np.ndarray  # pooled inner draws
    payoffs: np.ndarray
    source: np.ndarray  # which scenario produced each draw

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
    return float(np.mean((estimates - truth(kappa)) ** 2))


def _densities(e):
    """f[i, j]: density of draw j under scenario i."""
    return norm.pdf((e.draws[None, :] - e.kappa[:, None] - MU) / SIGMA) / SIGMA


def _self_normalised(weights, payoffs):
    """Weights rescaled to sum to one. Biased, consistent, far lower variance."""
    totals = weights.sum(axis=1)
    return (weights @ payoffs) / np.where(totals > 0, totals, np.nan)


def standard(e):
    """Each scenario keeps only its own paths."""
    outer = e.kappa.size
    totals = np.bincount(e.source, weights=e.payoffs, minlength=outer)
    return totals / np.bincount(e.source, minlength=outer)


def gns_known(e):
    """The paper's estimator."""
    f = _densities(e)
    return _self_normalised(f / f.mean(axis=0), e.payoffs)  # mixture denominator


def gns_pairwise(e):
    """Same, but dividing by each draw's own source instead of the mixture.

    The paper's 'individual likelihood ratio'. Nothing bounds the ratio when
    target and source sit far apart, which is the failure the mixture fixes.
    """
    f = _densities(e)
    own = f[e.source, np.arange(e.budget)]
    return _self_normalised(f / own, e.payoffs)


def _kernel_weights(e, dimensions, scale=1.0):
    state = np.column_stack([e.kappa, e.noise])[:, :dimensions]
    # d+4: the curse, algebraically. The exponent barely moves the result;
    # `scale` does, which is the tuning difficulty Hong et al. are known for.
    bandwidth = scale * e.budget ** (-1.0 / (dimensions + 4))
    gaps = state[:, None, :] - state[e.source][None, :, :]
    return np.exp(-0.5 * np.sum(gaps**2, axis=2) / bandwidth**2)


def gns_kernel(e, dimensions, scale=1.0):
    """GNS with f estimated: similarity over the conditioning state."""
    return _self_normalised(_kernel_weights(e, dimensions, scale), e.payoffs)


def bandwidth_sensitivity(scales, dimensions=1, budgets=(1000, 2000, 4000, 8000, 16000),
                          outer=200, trials=12, seed=2000):
    """IMSE and convergence rate against the bandwidth multiplier.

    Untuned, this is the dominant free parameter: error at a fixed budget moves
    by an order of magnitude across plausible scales, and the scale minimising
    error at one budget is not the one giving the best rate.
    """
    out = {}
    for scale in scales:
        vals = []
        for budget in budgets:
            errs = []
            for trial in range(trials):
                e = simulate(seed + trial * 7, dimensions, outer, budget // outer)
                errs.append(imse(gns_kernel(e, dimensions, scale), e.kappa))
            vals.append(float(np.mean(errs)))
        out[scale] = {"slope": slope(list(budgets), vals), "imse": dict(zip(budgets, vals))}
    return out


def effective_sample_size(e, dimensions):
    """1/sum(w^2): how many draws the weighting really averages per scenario."""
    w = _kernel_weights(e, dimensions)
    w = w / w.sum(axis=1, keepdims=True)
    return float(np.median(1.0 / np.sum(w**2, axis=1)))


def pooled(e):
    """The floor: ignore the conditioning."""
    return np.full(e.kappa.size, e.payoffs.mean())


ESTIMATORS = {
    "standard": lambda e, d: standard(e),
    "GNS known": lambda e, d: gns_known(e),
    "GNS pairwise": lambda e, d: gns_pairwise(e),
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

    result = {name: float(np.mean(v)) for name, v in totals.items()}
    result["kernel ESS"] = float(np.mean(ess))
    return result


def convergence(budgets, dimensions=1, outer=200, trials=20, seed=2000):
    """IMSE against total budget. The paper's O(1/budget) claim."""
    out = {}
    for budget in budgets:
        inner = budget // outer
        totals = {name: [] for name in ESTIMATORS}
        for trial in range(trials):
            e = simulate(seed + trial * 7, dimensions, outer, inner)
            for name, estimate in ESTIMATORS.items():
                totals[name].append(imse(estimate(e, dimensions), e.kappa))
        out[budget] = {n: float(np.mean(v)) for n, v in totals.items()}
    return out


def slope(budgets, values):
    """Log-log slope. -1 is the rate the theory predicts."""
    return float(np.polyfit(np.log(budgets), np.log(values), 1)[0])


def error_by_tail(dimensions=1, outer=200, inner=20, trials=40, seed=3000, bins=4):
    """Squared error grouped by |kappa|, central bin first.

    The paper's own caveat: scenarios far from the centre of the mixture get
    few well-weighted paths. Averaging over all scenarios hides it, and the
    tails are where VaR lives.
    """
    totals = {name: [[] for _ in range(bins)] for name in ESTIMATORS}
    for trial in range(trials):
        e = simulate(seed + trial * 7, dimensions, outer, inner)
        # Rank scenarios by distance from the mixture centre, then split evenly.
        group = np.argsort(np.argsort(np.abs(e.kappa))) * bins // e.kappa.size
        for name, estimate in ESTIMATORS.items():
            squared = (estimate(e, dimensions) - truth(e.kappa)) ** 2
            for b in range(bins):
                totals[name][b].append(float(np.mean(squared[group == b])))
    return {n: [float(np.mean(b)) for b in bs] for n, bs in totals.items()}
