"""Synthetic analytical fixtures, not GAMMA observations or performance results."""

import numpy as np
import pytest

from hierarch_inf.queueing.experiments import (
    MeanResponseConstraint, response_tolerance, run_paired_inference,
)


def test_absolute_and_relative_tolerances():
    np.testing.assert_allclose(response_tolerance([2, 4], kind="absolute", value=0.5), [0.5, 0.5])
    np.testing.assert_allclose(response_tolerance([2, 4], kind="relative", value=0.1), [0.2, 0.4])


@pytest.mark.parametrize("reference,kind,value", [
    ([], "absolute", 1), ([0], "relative", 0.1), ([1], "unknown", 1),
    ([1], "absolute", -1), ([1], "absolute", np.nan), ([1, 2], "absolute", [1]),
    ([np.inf], "absolute", 1), ([-1], "absolute", 1),
])
def test_invalid_tolerances(reference, kind, value):
    with pytest.raises(ValueError):
        response_tolerance(reference, kind=kind, value=value)


def test_full_interval_enclosure_must_fit_and_inputs_are_copied():
    ref = np.array([2.0, 3.0])
    c = MeanResponseConstraint([100, 200], ref, [0.5, 0.5])
    ref[:] = 0
    assert c.accepts_bounds([1.5, 2.5], [2.5, 3.5])
    assert not c.accepts_bounds([1.4, 2.5], [2.5, 3.5])
    assert not c.accepts_bounds([-np.inf, 2.5], [np.inf, 3.5])
    with pytest.raises(ValueError):
        c.reference[0] = 0
    with pytest.raises(ValueError):
        c.accepts_bounds([3, 3], [2, 4])
    with pytest.raises(ValueError):
        c.accepts_bounds([np.nan, 3], [3, 4])
    with pytest.raises(ValueError):
        c.accepts_curve([2])


@pytest.mark.parametrize("workload", [[2, 1], [1, 1], [-1, 2], [np.nan, 2], [1]])
def test_invalid_workload_grid(workload):
    with pytest.raises(ValueError):
        MeanResponseConstraint(workload, [2, 3], [0.5, 0.5])


def test_paired_inference_filters_before_likelihood_and_changes_target():
    c = MeanResponseConstraint([100, 200], [1.0, 2.0], [0.25, 0.25])
    def bounds(p):
        curve = np.array([p["theta"], p["theta"] + 1])
        return curve, curve
    report = run_paired_inference(
        likelihood=lambda p, _: -0.5 * (p["theta"] - 1)**2,
        data=None, prior={"theta": ["uniform", 0, 3]}, param_names=["theta"],
        initial_walkers=np.linspace(0.9, 1.1, 8)[:, None], curve_bounds=bounds,
        constraint=c, nsteps=100, discard=10, random_seed=7, setup_seconds=0.1,
        compute_diagnostics=False,
    )
    full, restricted = report["full"], report["restricted"]
    samples = restricted["result"].flat_samples[:, 0]
    assert np.all((samples >= 0.75) & (samples <= 1.25))
    assert np.any((full["result"].flat_samples[:, 0] < 0.75)
                  | (full["result"].flat_samples[:, 0] > 1.25))
    assert restricted["likelihood_calls"] < full["likelihood_calls"]
    assert restricted["certificate_calls"] > restricted["certificate_passes"]
    assert full["certificate_calls"] == 0
    assert restricted["total_seconds"] >= restricted["wall_seconds"] + 0.1
    np.testing.assert_array_equal(full["result"].initial_walkers, restricted["result"].initial_walkers)


def test_initial_walkers_require_certificate():
    with pytest.raises(ValueError, match="every initial walker"):
        run_paired_inference(
            likelihood=lambda p, _: 0.0, data=None, prior={"x": ["uniform", 0, 5]},
            param_names=["x"], initial_walkers=np.ones((8, 1)),
            curve_bounds=lambda p: ([4], [4]),
            constraint=MeanResponseConstraint([1], [1], [0.1]),
            nsteps=10, discard=0, random_seed=0, setup_seconds=0,
        )


def test_overflow_never_certifies_infinite_response():
    with pytest.raises(ValueError, match="overflows"):
        MeanResponseConstraint([1], [1e308], [1e308])
    with pytest.raises(ValueError, match="overflows"):
        response_tolerance([1e308], kind="relative", value=10)
    c = MeanResponseConstraint([1], [1], [1])
    assert not c.accepts_curve([np.inf])
