"""Complete-event transforms, without a service law or inference assumptions.

References are equations (1), (7), and (8) of Sutton and Jordan (2011),
https://arxiv.org/abs/1001.3355v3. All times use one consistent unit and a
nonnegative time origin. A queue's *entire* event history must be supplied;
applying these transforms to only logged jobs is not missing-data inference.
"""

from __future__ import annotations

import numpy as np


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and nonnegative; missing events are unsupported.")
    return array


def _paired(arrival_times, other_times, other_name):
    arrivals = _vector(arrival_times, "arrival_times")
    other = _vector(other_times, other_name)
    if arrivals.shape != other.shape:
        raise ValueError(f"arrival_times and {other_name} must have the same shape.")
    return arrivals, other


def _fcfs_inputs(arrival_times, other_times, other_name, initial_departure):
    arrivals, other = _paired(arrival_times, other_times, other_name)
    if np.any(np.diff(arrivals) < 0):
        raise ValueError("FCFS arrivals must be in nondecreasing order.")
    initial = np.asarray(initial_departure, dtype=float)
    if initial.ndim != 0 or not np.isfinite(initial) or initial < 0:
        raise ValueError("initial_departure must be a finite nonnegative scalar.")
    return arrivals, other, float(initial)


def fcfs_departures(arrival_times, service_times, *, initial_departure=0.0):
    """Single-server FCFS forward map, ``d[e] = s[e] + max(a[e], d[e-1])``.

    Inputs are already in arrival order; tied arrivals use input order.
    ``initial_departure`` summarizes any preceding job; zero denotes an
    initially empty queue. Zero-length services and empty arrays are allowed.
    This is equation (1), not the multi-server or processor-sharing model.
    """
    arrivals, services, previous = _fcfs_inputs(
        arrival_times, service_times, "service_times", initial_departure,
    )
    departures = np.empty_like(arrivals)
    for index, (arrival, service) in enumerate(zip(arrivals, services)):
        previous = max(float(arrival), previous) + float(service)
        if not np.isfinite(previous):
            raise ValueError("Departure time overflow; rescale the input time unit.")
        departures[index] = previous
    return departures


def fcfs_service_times(arrival_times, departure_times, *, initial_departure=0.0):
    """Invert equation (1) for a complete single-server FCFS event sequence.

    An impossible departure (before its service commencement) is rejected,
    never clipped. Arrival ties use input order, as in ``fcfs_departures``.
    """
    arrivals, departures, initial = _fcfs_inputs(
        arrival_times, departure_times, "departure_times", initial_departure,
    )
    previous = np.concatenate(([initial], departures[:-1])) if arrivals.size else arrivals
    services = departures - np.maximum(arrivals, previous)
    if np.any(services < 0):
        raise ValueError("Departures are infeasible for a single-server FCFS queue.")
    return services


def _ps_events(arrival_times, departure_times):
    arrivals, departures = _paired(arrival_times, departure_times, "departure_times")
    if np.any(departures < arrivals):
        raise ValueError("Each departure must be at or after its arrival.")
    return arrivals, departures


def ps_service_times(arrival_times, departure_times):
    """Invert complete processor-sharing events using ``s[e] = integral 1/N(t) dt``.

    Equation (7) shares one unit of processing capacity equally among active
    jobs. Arrays retain job correspondence but need not be sorted. All jobs
    that overlap the supplied events must be present, including jobs that
    were not logged in a partially observed experiment.

    Ties are valid for this integral: occupancy on each open inter-event
    interval is used. Zero-duration events have zero service. Empty arrays
    return an empty array. No event is sampled, inferred, or silently removed.
    """
    arrivals, departures = _ps_events(arrival_times, departure_times)
    if not arrivals.size:
        return np.empty(0, dtype=float)

    knots = np.unique(np.concatenate((arrivals, departures)))
    arrival_indices = np.searchsorted(knots, arrivals)
    departure_indices = np.searchsorted(knots, departures)
    changes = np.zeros(knots.size, dtype=np.int64)
    np.add.at(changes, arrival_indices, 1)
    np.add.at(changes, departure_indices, -1)
    occupancy = np.cumsum(changes)[:-1]
    increments = np.divide(
        np.diff(knots), occupancy,
        out=np.zeros(knots.size - 1, dtype=float), where=occupancy > 0,
    )
    virtual_time = np.concatenate(([0.0], np.cumsum(increments)))
    return virtual_time[departure_indices] - virtual_time[arrival_indices]


def ps_log_jacobian(arrival_times, departure_times):
    """Log absolute Jacobian for complete, distinct-time PS events (equation 8).

    Returns ``-sum(log(N(d[e]-)))``; it is *not* the full likelihood. Service
    density and arrival/routing factors must be supplied separately. The
    reciprocal occupancy follows equation (8); the printed equation (11)
    instead omits the reciprocal.

    At tied event times the map can be nondifferentiable. Rather than invent
    a tie-breaking density, this function raises ``ValueError`` for any tie,
    including zero-duration events. The service-time integral above still
    supports those cases. An empty event list has log Jacobian zero.
    """
    arrivals, departures = _ps_events(arrival_times, departure_times)
    events = np.concatenate((arrivals, departures))
    if np.unique(events).size != events.size:
        raise ValueError("PS log Jacobian requires distinct event times (no zero-duration events).")
    number_before_departure = (
        np.searchsorted(np.sort(arrivals), departures, side="left")
        - np.searchsorted(np.sort(departures), departures, side="left")
    )
    return float(-np.log(number_before_departure).sum())
