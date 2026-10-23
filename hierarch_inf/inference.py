"""Direct and behaviorally guided Bayesian inference with emcee.

Model-specific simulations and likelihoods are supplied by the caller.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np


ParamsDict = Dict[str, float]
LogLikelihoodFn = Callable[[ParamsDict, Any], float]


def _prior_logpdf_one(value: float, specification: Any) -> float:
    """
    Evaluate one scalar prior entry.

    Supported compact specifications
    --------------------------------
    ["uniform", lower, upper]
    ["log-uniform", lower, upper]
    ["normal", mean, std]
    ["log-normal", log_mean, log_std]
    callable(value) -> log density

    The log-uniform prior includes the -log(value) density term.
    """
    if callable(specification):
        out = float(specification(value))
        return out

    if not isinstance(specification, (tuple, list)) or len(specification) < 1:
        raise ValueError("Prior specification must be a callable or list/tuple.")

    kind = str(specification[0]).strip().lower().replace("_", "-")

    if kind == "uniform":
        _, lower, upper = specification
        if value < lower or value > upper:
            return -np.inf
        return -math.log(float(upper) - float(lower))

    if kind in {"log-uniform", "loguniform"}:
        _, lower, upper = specification
        lower = float(lower)
        upper = float(upper)
        if value <= 0 or value < lower or value > upper:
            return -np.inf
        return -math.log(value) - math.log(math.log(upper / lower))

    if kind in {"normal", "gaussian"}:
        _, mean, std = specification
        std = float(std)
        if std <= 0:
            raise ValueError("Normal prior std must be positive.")
        z = (value - float(mean)) / std
        return -0.5 * z * z - math.log(std * math.sqrt(2.0 * math.pi))

    if kind in {"log-normal", "lognormal"}:
        _, log_mean, log_std = specification
        log_std = float(log_std)
        if value <= 0:
            return -np.inf
        if log_std <= 0:
            raise ValueError("Log-normal prior log_std must be positive.")
        z = (math.log(value) - float(log_mean)) / log_std
        return (
            -0.5 * z * z
            - math.log(value)
            - math.log(log_std * math.sqrt(2.0 * math.pi))
        )

    raise ValueError(f"Unsupported prior type: {specification[0]!r}")


def _joint_log_prior(
    theta: np.ndarray,
    param_names: Sequence[str],
    prior: Mapping[str, Any],
    custom_joint_prior: Optional[Callable[[Mapping[str, float]], float]],
) -> float:
    """Sum the independent parameter log priors for a parameter vector."""
    params = dict(zip(param_names, np.asarray(theta, dtype=float)))

    logp = 0.0
    for name in param_names:
        if name not in prior:
            raise KeyError(f"Missing prior specification for parameter {name!r}.")
        lp = _prior_logpdf_one(float(params[name]), prior[name])
        if not np.isfinite(lp):
            return -np.inf
        logp += lp

    if custom_joint_prior is not None:
        # Match Bioscrape: finite values are additive log-prior contributions;
        # non-finite values reject a parameter proposal.
        joint_lp = float(custom_joint_prior(params))
        if not np.isfinite(joint_lp):
            return -np.inf
        logp += joint_lp

    return float(logp)


@dataclass
class InferenceResult:
    """MCMC samples, log probabilities, timing, and optional diagnostics."""
    sampler: Any
    chain: np.ndarray
    flat_samples: np.ndarray
    log_prob: np.ndarray
    flat_log_prob: np.ndarray
    acceptance_fraction: np.ndarray
    autocorr_time: Optional[np.ndarray]
    ess: Optional[np.ndarray]
    runtime_seconds: float
    param_names: Tuple[str, ...]
    discard: int
    thin: int
    initial_walkers: np.ndarray


def _default_initial_walkers_from_prior(
    prior: Mapping[str, Any],
    param_names: Sequence[str],
    nwalkers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw an initial ensemble from supported priors.
    """
    p0 = np.zeros((nwalkers, len(param_names)), dtype=float)

    for j, name in enumerate(param_names):
        spec = prior[name]

        if callable(spec):
            raise ValueError(
                "Callable priors require explicit initial_walkers."
            )

        kind = str(spec[0]).strip().lower().replace("_", "-")

        if kind == "uniform":
            _, lo, hi = spec
            p0[:, j] = rng.uniform(float(lo), float(hi), size=nwalkers)
        elif kind in {"log-uniform", "loguniform"}:
            _, lo, hi = spec
            p0[:, j] = np.exp(
                rng.uniform(np.log(float(lo)), np.log(float(hi)), size=nwalkers)
            )
        elif kind in {"normal", "gaussian"}:
            _, mean, std = spec
            p0[:, j] = rng.normal(float(mean), float(std), size=nwalkers)
        elif kind in {"log-normal", "lognormal"}:
            _, log_mean, log_std = spec
            p0[:, j] = np.exp(
                rng.normal(float(log_mean), float(log_std), size=nwalkers)
            )
        else:
            raise ValueError(
                f"Unsupported prior type {spec[0]!r} for automatic initialization."
            )

    return p0


def py_inference(
    *,
    likelihood: LogLikelihoodFn,
    data: Any,
    params_to_estimate: Sequence[str],
    prior: Mapping[str, Any],
    custom_joint_prior: Optional[
        Callable[[Mapping[str, float]], float]
    ] = None,
    nwalkers: int = 32,
    nsteps: int = 8000,
    discard: int = 1000,
    thin: int = 1,
    initial_walkers: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
    progress: bool = False,
    moves: Any = None,
    compute_diagnostics: bool = True,
    emcee_kwargs: Optional[Mapping[str, Any]] = None,
) -> InferenceResult:
    """
    emcee-native direct Bayesian inference wrapper.

    Parameters
    ----------
    likelihood
        Callable with signature

            likelihood(params_dict, data) -> log_likelihood

        The callable owns all model-specific simulation and observation-model
        logic. It must return a log likelihood, with -np.inf for impossible
        parameter values.

    data
        Passed unchanged to ``likelihood``.

    params_to_estimate
        Ordered parameter names.

    prior
        Dictionary keyed by parameter name. Supported compact prior entries:

            ["uniform", lower, upper]
            ["log-uniform", lower, upper]
            ["normal", mean, std]
            ["log-normal", log_mean, log_std]

        A scalar callable ``prior[name](value) -> log_density`` is also allowed.

    custom_joint_prior
        Optional callable with the same convention used in Bioscrape:

            custom_joint_prior(params_dict) == 0.0    -> allowed
            custom_joint_prior(params_dict) == inf    -> reject
            finite value                              -> additive log-prior contribution

        Therefore a learned ``GPValidityModel`` plugs in directly as

            custom_joint_prior=gp_result.custom_joint_prior

        or

            custom_joint_prior=gp_result.validity_model.custom_joint_prior

    initial_walkers
        Shape (nwalkers, ndim). If omitted, walkers are drawn independently
        from the supplied priors until all have finite joint prior. Supplying
        explicit walkers is preferable when baseline and GP-filtered runs must
        begin from the same ensemble.

    Returns
    -------
    InferenceResult
        Contains the native emcee sampler plus raw/flattened chains, log
        probabilities, acceptance fractions, runtime, and autocorrelation/ESS
        diagnostics when available.

    Notes
    -----
    This function intentionally does not know how to simulate the toy model or
    any biological model.
    """
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "py_inference requires emcee. Install it with `pip install emcee`."
        ) from exc

    names = tuple(params_to_estimate)
    ndim = len(names)

    if ndim == 0:
        raise ValueError("params_to_estimate cannot be empty.")
    if nwalkers < 2 * ndim:
        raise ValueError(
            "For the default emcee ensemble moves, use at least 2 * ndim walkers."
        )
    if nsteps <= 0:
        raise ValueError("nsteps must be positive.")
    if discard < 0 or discard >= nsteps:
        raise ValueError("discard must satisfy 0 <= discard < nsteps.")
    if thin <= 0:
        raise ValueError("thin must be positive.")

    rng = np.random.default_rng(random_seed)

    def log_probability(theta: np.ndarray) -> float:
        """Combine prior and likelihood, applying behavioral screening first."""
        lp = _joint_log_prior(
            theta,
            names,
            prior,
            custom_joint_prior,
        )
        if not np.isfinite(lp):
            return -np.inf

        params = dict(zip(names, np.asarray(theta, dtype=float)))
        ll = float(likelihood(params, data))
        if not np.isfinite(ll):
            return -np.inf

        return lp + ll

    if initial_walkers is None:
        # Draw from the ordinary priors and resample any walkers rejected by the
        # custom joint prior. This keeps GP filtering inside the prior layer.
        p0 = np.empty((nwalkers, ndim), dtype=float)
        n_filled = 0
        attempts = 0
        max_attempts = max(10000, 1000 * nwalkers)

        while n_filled < nwalkers:
            batch_n = max(2 * (nwalkers - n_filled), 8)
            candidates = _default_initial_walkers_from_prior(
                prior,
                names,
                batch_n,
                rng,
            )
            for candidate in candidates:
                attempts += 1
                if np.isfinite(
                    _joint_log_prior(
                        candidate,
                        names,
                        prior,
                        custom_joint_prior,
                    )
                ):
                    p0[n_filled] = candidate
                    n_filled += 1
                    if n_filled == nwalkers:
                        break

                if attempts >= max_attempts:
                    raise RuntimeError(
                        "Could not generate enough finite-prior initial walkers. "
                        "The prior/custom_joint_prior retained region may be very small; "
                        "pass explicit initial_walkers."
                    )
    else:
        p0 = np.asarray(initial_walkers, dtype=float)
        if p0.shape != (nwalkers, ndim):
            raise ValueError(
                f"initial_walkers must have shape {(nwalkers, ndim)}, got {p0.shape}."
            )

        bad = [
            i for i, theta in enumerate(p0)
            if not np.isfinite(
                _joint_log_prior(
                    theta,
                    names,
                    prior,
                    custom_joint_prior,
                )
            )
        ]
        if bad:
            raise ValueError(
                f"Initial walkers at indices {bad[:10]} have non-finite joint prior."
            )

    sampler_kwargs = dict(emcee_kwargs or {})
    if moves is not None:
        sampler_kwargs["moves"] = moves
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        **sampler_kwargs,
    )
    if random_seed is not None:
        # Seed proposal draws as well as initialization, without changing the
        # caller's global NumPy random state.
        sampler.random_state = np.random.RandomState(random_seed).get_state()

    start = time.perf_counter()
    sampler.run_mcmc(
        p0,
        int(nsteps),
        progress=progress,
    )
    runtime = time.perf_counter() - start

    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    flat_samples = sampler.get_chain(
        discard=int(discard),
        thin=int(thin),
        flat=True,
    )
    flat_log_prob = sampler.get_log_prob(
        discard=int(discard),
        thin=int(thin),
        flat=True,
    )
    acceptance_fraction = np.asarray(
        sampler.acceptance_fraction,
        dtype=float,
    )

    tau = None
    ess = None

    if compute_diagnostics:
        try:
            tau = np.asarray(
                sampler.get_autocorr_time(
                    discard=int(discard),
                    thin=int(thin),
                    tol=0,
                ),
                dtype=float,
            )
            # emcee returns tau in original iteration units, so express the
            # retained sample count in those same units before computing ESS.
            n_postburn = flat_samples.shape[0] * int(thin)
            ess = n_postburn / tau
        except Exception:
            # Short chains can legitimately fail the autocorrelation-time
            # estimate. Preserve the completed inference and expose None.
            tau = None
            ess = None

    return InferenceResult(
        sampler=sampler,
        chain=chain,
        flat_samples=flat_samples,
        log_prob=log_prob,
        flat_log_prob=flat_log_prob,
        acceptance_fraction=acceptance_fraction,
        autocorr_time=tau,
        ess=ess,
        runtime_seconds=float(runtime),
        param_names=names,
        discard=int(discard),
        thin=int(thin),
        initial_walkers=p0,
    )
