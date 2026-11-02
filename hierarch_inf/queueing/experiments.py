"""Configurable mean-response certificates and paired inference experiments.

The caller supplies the detailed likelihood and valid bounds on its mean
response curve. This module does not select a queue discipline or service law.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import importlib

import numpy as np

from ..inference import py_inference


def response_tolerance(reference, *, kind, value):
    """Build explicit absolute or fractional tolerances in the reference unit."""
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 1 or reference.size == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("reference must be a nonempty finite vector")
    if np.any(reference < 0):
        raise ValueError("mean response reference must be nonnegative")
    value = np.asarray(value, dtype=float)
    if value.ndim > 1 or (value.ndim == 1 and value.shape != reference.shape):
        raise ValueError("tolerance must be scalar or match the reference vector")
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    if kind == "absolute":
        return np.broadcast_to(value, reference.shape).copy()
    if kind == "relative":
        if np.any(reference <= 0):
            raise ValueError("relative tolerance requires positive reference means")
        with np.errstate(over="ignore"):
            tolerance = np.broadcast_to(value, reference.shape) * reference
        if not np.all(np.isfinite(tolerance)):
            raise ValueError("relative tolerance overflows")
        return tolerance
    raise ValueError("kind must be 'absolute' or 'relative'")


@dataclass(frozen=True)
class MeanResponseConstraint:
    """A workload grid, reference mean curve, and absolute error allowances."""

    workloads: np.ndarray
    reference: np.ndarray
    tolerance: np.ndarray

    def __post_init__(self):
        workloads = np.asarray(self.workloads, dtype=float)
        reference = np.asarray(self.reference, dtype=float)
        tolerance = response_tolerance(reference, kind="absolute", value=self.tolerance)
        with np.errstate(over="ignore"):
            finite_band = np.all(np.isfinite(reference + tolerance))
        if not finite_band:
            raise ValueError("reference tolerance band overflows")
        if workloads.shape != reference.shape or not np.all(np.isfinite(workloads)):
            raise ValueError("workloads must be finite and match the reference vector")
        if np.any(workloads < 0) or np.any(np.diff(workloads) <= 0):
            raise ValueError("workloads must be nonnegative and strictly increasing")
        for name, values in (("workloads", workloads), ("reference", reference), ("tolerance", tolerance)):
            values = values.copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def accepts_bounds(self, lower, upper):
        """Certify the full enclosure at the configured workload grid points.

        Soundness is conditional on the caller's lower/upper bounds actually
        enclosing the detailed mean curve. Infinite endpoints are allowed and
        cannot certify a finite tolerance band. NaNs and inverted bounds fail.
        """
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if lower.shape != self.reference.shape or upper.shape != self.reference.shape:
            raise ValueError("curve bounds must match the reference vector")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("curve bounds must be ordered and cannot contain NaN")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            return False
        return bool(np.all(lower >= self.reference - self.tolerance)
                    and np.all(upper <= self.reference + self.tolerance))

    def accepts_curve(self, curve):
        return self.accepts_bounds(curve, curve)


def run_paired_inference(*, likelihood, data, prior, param_names, initial_walkers,
                         curve_bounds, constraint, nsteps, discard, random_seed,
                         setup_seconds, thin=1, compute_diagnostics=True):
    """Run full and behavior-restricted inference with matched configurations.

    ``curve_bounds(params)`` returns lower/upper detailed mean bounds at
    ``constraint.workloads``. It must be fixed for the experiment and valid on
    the stated model domain. Every initial walker must be certified. The
    baseline and restricted runs start from that same ensemble and use the
    same random seed; their subsequent paths can differ.

    The caller reports all external construction/calibration/learning cost
    through ``setup_seconds``. Returned wall times include initialization and
    diagnostics; likelihood and certificate calls are counted separately.
    An ESS estimate, when available, pertains to each run's own target.
    """
    if not isinstance(constraint, MeanResponseConstraint):
        raise TypeError("constraint must be a MeanResponseConstraint")
    if not np.isfinite(setup_seconds) or setup_seconds < 0:
        raise ValueError("setup_seconds must be finite and nonnegative")
    walkers = np.asarray(initial_walkers, dtype=float)
    names = tuple(param_names)
    if walkers.ndim != 2 or walkers.shape[1] != len(names) or not np.all(np.isfinite(walkers)):
        raise ValueError("initial_walkers must be finite with one column per parameter")
    if not names or len(set(names)) != len(names):
        raise ValueError("parameter names must be unique and nonempty")

    validation_start = time.perf_counter()
    for row in walkers:
        bounds = curve_bounds(dict(zip(names, row)))
        if not constraint.accepts_bounds(*bounds):
            raise ValueError("every initial walker must satisfy the behavioral certificate")
    validation_seconds = time.perf_counter() - validation_start
    common = dict(data=data, params_to_estimate=names, prior=prior,
                  initial_walkers=walkers.copy(), nwalkers=len(walkers),
                  nsteps=nsteps, discard=discard, thin=thin,
                  random_seed=random_seed, progress=False,
                  compute_diagnostics=compute_diagnostics)
    # Exclude the shared lazy dependency import from both timed runs.
    importlib.import_module("emcee")
    results = {}
    for label in ("full", "restricted"):
        counts = {"likelihood_calls": 0, "certificate_calls": 0,
                  "certificate_passes": 0, "certificate_seconds": 0.0}

        def counted_likelihood(params, observations):
            counts["likelihood_calls"] += 1
            return likelihood(params, observations)

        def certificate(params):
            started = time.perf_counter()
            counts["certificate_calls"] += 1
            accepted = constraint.accepts_bounds(*curve_bounds(params))
            counts["certificate_seconds"] += time.perf_counter() - started
            counts["certificate_passes"] += int(accepted)
            return 0.0 if accepted else np.inf

        started = time.perf_counter()
        result = py_inference(likelihood=counted_likelihood,
                              custom_joint_prior=certificate if label == "restricted" else None,
                              **common)
        wall = time.perf_counter() - started
        total = wall + (setup_seconds + validation_seconds if label == "restricted" else 0.0)
        results[label] = {"result": result, "wall_seconds": wall,
                          "total_seconds": total, **counts}
        results[label]["ess_is_exploratory"] = True
        results[label]["convergence_verified"] = False
        results[label]["min_ess_per_total_second"] = (
            float(np.min(result.ess) / total) if result.ess is not None and total > 0 else None
        )
    results["configuration"] = {
        "nsteps": int(nsteps), "discard": int(discard), "thin": int(thin),
        "random_seed": int(random_seed), "external_setup_seconds": float(setup_seconds),
        "initial_certificate_seconds": validation_seconds,
        "initial_certificate_calls": len(walkers),
        "workloads": constraint.workloads.tolist(), "reference": constraint.reference.tolist(),
        "tolerance": constraint.tolerance.tolist(),
        "guarantee_scope": "supplied curve bounds at configured workload grid points",
    }
    return results
