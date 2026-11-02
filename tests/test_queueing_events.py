"""Analytical unit fixtures only; these are not the Cloudstone data."""

import numpy as np
import pytest

from hierarch_inf.queueing import (
    fcfs_departures,
    fcfs_service_times,
    ps_log_jacobian,
    ps_service_times,
)


def test_fcfs_forward_inverse_waiting_and_idle_intervals():
    arrivals = np.array([0.0, 1.0, 5.0])
    services = np.array([2.0, 1.0, 0.5])
    departures = fcfs_departures(arrivals, services)
    np.testing.assert_allclose(departures, [2.0, 3.0, 5.5])
    np.testing.assert_allclose(fcfs_service_times(arrivals, departures), services)
    np.testing.assert_allclose(departures - arrivals - services, [0.0, 1.0, 0.0])


def test_fcfs_ties_zero_service_and_initial_backlog():
    arrivals = [0.0, 0.0, 2.0]
    services = [1.0, 0.0, 2.0]
    departures = fcfs_departures(arrivals, services, initial_departure=1.0)
    np.testing.assert_allclose(departures, [2.0, 2.0, 4.0])
    np.testing.assert_allclose(
        fcfs_service_times(arrivals, departures, initial_departure=1.0), services,
    )


def test_fcfs_rejects_infeasible_departures_not_clipping_service():
    with pytest.raises(ValueError, match="infeasible"):
        fcfs_service_times([0.0, 1.0], [3.0, 2.0])


def test_ps_service_integral_and_reciprocal_occupancy_jacobian():
    # Occupancy is 1 on (0,1), 2 on (1,3), then 1 on (3,4).
    arrivals, departures = [0.0, 1.0], [3.0, 4.0]
    np.testing.assert_allclose(ps_service_times(arrivals, departures), [2.0, 2.0])
    assert ps_log_jacobian(arrivals, departures) == pytest.approx(-np.log(2.0))


def test_ps_is_order_independent_and_conserves_busy_time():
    arrivals = np.array([5.0, 0.0, 1.0, 6.0])
    departures = np.array([8.0, 3.0, 4.0, 7.0])
    services = ps_service_times(arrivals, departures)
    np.testing.assert_allclose(services, [2.5, 2.0, 2.0, 0.5])
    assert services.sum() == pytest.approx(7.0)  # Busy on (0,4) and (5,8).
    perm = [3, 1, 0, 2]
    np.testing.assert_allclose(
        ps_service_times(arrivals[perm], departures[perm]), services[perm],
    )
    assert ps_log_jacobian(arrivals[perm], departures[perm]) == pytest.approx(-2 * np.log(2))


def test_ps_tied_endpoints_and_zero_duration_have_explicit_integral():
    np.testing.assert_allclose(ps_service_times([0, 0, 2, 3], [2, 2, 3, 3]), [1, 1, 1, 0])
    np.testing.assert_allclose(ps_service_times([1, 1], [1, 1]), [0, 0])


@pytest.mark.parametrize("arrivals,departures", [([0, 0], [2, 3]), ([0, 1], [2, 2]), ([0, 2], [2, 3]), ([1], [1])])
def test_ps_jacobian_rejects_nondifferentiable_ties(arrivals, departures):
    with pytest.raises(ValueError, match="distinct event times"):
        ps_log_jacobian(arrivals, departures)


def test_ps_jacobian_matches_finite_difference_determinant():
    arrivals = np.array([0.0, 0.4, 0.8])
    departures = np.array([2.0, 1.4, 3.0])
    step = 1e-6
    jacobian = np.column_stack([
        (ps_service_times(arrivals, departures + step * basis)
         - ps_service_times(arrivals, departures - step * basis)) / (2 * step)
        for basis in np.eye(3)
    ])
    _, logdet = np.linalg.slogdet(jacobian)
    assert ps_log_jacobian(arrivals, departures) == pytest.approx(logdet, abs=1e-8)


@pytest.mark.parametrize("transform", [fcfs_departures, fcfs_service_times, ps_service_times])
def test_empty_events(transform):
    assert transform([], []).shape == (0,)
    assert ps_log_jacobian([], []) == 0.0


@pytest.mark.parametrize("transform", [fcfs_departures, fcfs_service_times, ps_service_times, ps_log_jacobian])
@pytest.mark.parametrize("first,second", [([np.nan], [1]), ([0], [np.inf]), ([-1], [1]), ([0, 1], [2]), ([[0]], [[1]])])
def test_invalid_and_missing_inputs_are_rejected(transform, first, second):
    with pytest.raises(ValueError):
        transform(first, second)


@pytest.mark.parametrize("transform", [fcfs_departures, fcfs_service_times])
def test_fcfs_rejects_unsorted_arrivals_and_invalid_initial_time(transform):
    with pytest.raises(ValueError, match="nondecreasing"):
        transform([1, 0], [2, 3])
    for initial in [-1, np.inf, np.nan, [0]]:
        with pytest.raises(ValueError, match="initial_departure"):
            transform([0], [2], initial_departure=initial)


@pytest.mark.parametrize("transform", [ps_service_times, ps_log_jacobian])
def test_ps_rejects_departure_before_arrival(transform):
    with pytest.raises(ValueError, match="at or after"):
        transform([2], [1])


def test_fcfs_does_not_return_infinite_departures_on_overflow():
    with pytest.raises(ValueError, match="overflow"):
        fcfs_departures([1e308], [1e308])
