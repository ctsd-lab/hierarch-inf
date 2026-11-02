"""Analytical and seeded round-trip checks for the PS forward simulator."""

import numpy as np
import pytest

from hierarch_inf.queueing import ps_departures, ps_service_times


@pytest.mark.parametrize("arrivals,services,expected", [
    ([], [], []),
    ([3.0], [2.0], [5.0]),
    ([0.0, 1.0], [2.0, 2.0], [3.0, 4.0]),
    ([0.0, 0.0], [1.0, 2.0], [2.0, 3.0]),
    ([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]),
    ([0.0, 2.0, 3.0], [2.0, 1.0, 1.0], [2.0, 3.0, 4.0]),
    ([0.0, 5.0], [1.0, 2.0], [1.0, 7.0]),
    ([0.0, 0.0, 1.0], [2.0, 0.0, 0.0], [2.0, 0.0, 1.0]),
    ([1.0, 1.0, 10.0], [0.0, 0.0, 0.0], [1.0, 1.0, 10.0]),
    ([1.0, 0.0], [2.0, 2.0], [4.0, 3.0]),
])
def test_analytic_schedules(arrivals, services, expected):
    departures = ps_departures(arrivals, services)
    np.testing.assert_allclose(departures, expected)
    np.testing.assert_allclose(ps_service_times(arrivals, departures), services, atol=1e-14)


@pytest.mark.parametrize("seed", range(10))
def test_forward_inverse_round_trip_and_permutation(seed):
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(0.3, 100))
    services = rng.exponential(1.0, 100)
    arrivals_before, services_before = arrivals.copy(), services.copy()
    departures = ps_departures(arrivals, services)
    np.testing.assert_allclose(ps_service_times(arrivals, departures), services, atol=1e-12)
    assert np.all(departures >= arrivals)
    permutation = rng.permutation(arrivals.size)
    np.testing.assert_allclose(
        ps_departures(arrivals[permutation], services[permutation]), departures[permutation],
        atol=1e-12,
    )
    np.testing.assert_array_equal(arrivals, arrivals_before)
    np.testing.assert_array_equal(services, services_before)


def test_ps_work_conservation_for_single_busy_period():
    arrivals = np.arange(100) / 10.0
    services = np.ones(100)
    departures = ps_departures(arrivals, services)
    assert max(departures) == pytest.approx(sum(services), abs=1e-11)


def test_time_translation_and_scaling():
    arrivals = np.array([0.1, 0.3, 0.9, 2.0])
    services = np.array([0.5, 0.2, 2.0, 0.7])
    expected = ps_departures(arrivals, services)
    np.testing.assert_allclose(ps_departures(arrivals + 10, services), expected + 10)
    np.testing.assert_allclose(ps_departures(arrivals * 100, services * 100), expected * 100)


@pytest.mark.parametrize("arrivals,services", [
    ([np.nan], [1]), ([0], [np.inf]), ([-1], [1]), ([0], [-1]),
    ([0, 1], [2]), ([[0]], [[1]]),
])
def test_invalid_inputs(arrivals, services):
    with pytest.raises(ValueError):
        ps_departures(arrivals, services)


def test_time_overflow():
    with pytest.raises(ValueError, match="overflow"):
        ps_departures([1e308], [1e308])


def test_unrepresentable_completion_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        ps_departures([1e20], [1.0])
