import numpy as np
import pytest

from gns import (
    MU,
    SIGMA,
    Experiment,
    gns_kernel,
    gns_known,
    imse,
    payoff,
    pooled,
    simulate,
    standard,
    truth,
)


def test_closed_form_matches_brute_force():
    """If the analytic truth were wrong, every IMSE below would be meaningless."""
    rng = np.random.default_rng(42)
    kappa = 0.3
    draws = kappa + MU + SIGMA * rng.standard_normal(2_000_000)
    assert payoff(draws).mean() == pytest.approx(truth(kappa), abs=0.005)


def test_truth_is_monotone_in_kappa():
    values = truth(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert np.all(np.diff(values) > 0)


@pytest.fixture
def experiment():
    return simulate(seed=1234, dimensions=1, outer=200, inner=20)


def test_every_estimator_returns_one_value_per_scenario(experiment):
    for estimates in (
        standard(experiment),
        gns_known(experiment),
        gns_kernel(experiment, 1),
        pooled(experiment),
    ):
        assert estimates.shape == experiment.kappa.shape
        assert np.all(np.isfinite(estimates))


def test_reuse_beats_discarding_paths(experiment):
    """The paper's headline, reproduced."""
    reuse = imse(gns_known(experiment), experiment.kappa)
    naive = imse(standard(experiment), experiment.kappa)
    assert reuse < naive / 10


def test_conditioning_beats_ignoring_it(experiment):
    assert imse(gns_known(experiment), experiment.kappa) < imse(
        pooled(experiment), experiment.kappa
    )


def test_kernel_loses_its_advantage_with_dimension():
    """The result this repo exists to report."""
    low = simulate(seed=99, dimensions=1, outer=200, inner=20)
    high = simulate(seed=99, dimensions=8, outer=200, inner=20)
    assert imse(gns_kernel(high, 8), high.kappa) > 5 * imse(
        gns_kernel(low, 1), low.kappa
    )


def test_weights_are_self_normalised():
    """A constant payoff must come back unchanged whatever the weights are."""
    e = simulate(seed=7, dimensions=1, outer=50, inner=10)
    flat = Experiment(e.kappa, e.noise, e.draws, np.full(e.budget, 3.0), e.source)
    assert gns_known(flat) == pytest.approx(3.0)
    assert gns_kernel(flat, 1) == pytest.approx(3.0)


def test_simulation_is_deterministic():
    a = simulate(seed=7, dimensions=2, outer=50, inner=5)
    b = simulate(seed=7, dimensions=2, outer=50, inner=5)
    np.testing.assert_array_equal(a.draws, b.draws)
    np.testing.assert_array_equal(a.kappa, b.kappa)
