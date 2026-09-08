"""
implementation_methods.py

Reusable inference and GP-MPM methods for hierarchical behavioral inference.

Design goals
------------
1. Keep model simulation outside this module. Reduced- and detailed-model
   simulation are supplied as callables.
2. Encapsulate the GP coordinate map, GP fitting, GP-MPM acquisition, and
   sequential level-set learning.
3. Provide a high-level wrapper that can start from a reduced-model posterior
   chain and construct a GP approximation of the detailed-model retained set.
4. Provide one emcee-native ``py_inference`` entry point. GP-guided inference is
   handled through ``custom_joint_prior`` in the same style used in the
   Bioscrape workflow.

The GP learns the sign of

    v(beta) = epsilon - r(beta),

where r(beta) is a posterior summary of the discrepancy between the detailed
trajectory at beta and trajectories induced by reduced-posterior samples.
The retained region is v(beta) >= 0.

No model-specific ODE simulation code is included here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import erfc
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


ArrayLike = Union[np.ndarray, Sequence[float]]
ParamsDict = Dict[str, float]
TrajectoryFn = Callable[[ParamsDict], np.ndarray]
LogLikelihoodFn = Callable[[ParamsDict, Any], float]
DiscrepancyFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def map_parameters_to_gp_space(
    parameters: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    *,
    log_scale: bool = True,
) -> np.ndarray:
    """
    Map physical parameter values to normalized GP coordinates in [0, 1]^d.

    Parameters
    ----------
    parameters
        Shape (..., d).
    lower_bounds, upper_bounds
        Shape (d,).
    log_scale
        If True, normalize in log-parameter space, matching the gene-expression
        GP notebook. Set False for linearly parameterized domains.

    Returns
    -------
    X
        Shape (..., d).
    """
    theta = np.asarray(parameters, dtype=float)
    lo = np.asarray(lower_bounds, dtype=float)
    hi = np.asarray(upper_bounds, dtype=float)

    if lo.ndim != 1 or hi.ndim != 1 or lo.shape != hi.shape:
        raise ValueError("lower_bounds and upper_bounds must be 1D arrays with equal shape.")
    if theta.shape[-1] != lo.size:
        raise ValueError("Last parameter dimension does not match bounds.")
    if np.any(hi <= lo):
        raise ValueError("Each upper bound must be greater than its lower bound.")

    if log_scale:
        if np.any(theta <= 0) or np.any(lo <= 0) or np.any(hi <= 0):
            raise ValueError("log_scale=True requires strictly positive values and bounds.")
        theta_t = np.log(theta)
        lo_t = np.log(lo)
        hi_t = np.log(hi)
    else:
        theta_t = theta
        lo_t = lo
        hi_t = hi

    return (theta_t - lo_t) / (hi_t - lo_t)


def map_gp_space_to_parameters(
    X: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    *,
    log_scale: bool = True,
) -> np.ndarray:
    """
    Inverse of :func:`map_parameters_to_gp_space`.

    Parameters
    ----------
    X
        Shape (..., d), usually values in [0, 1].
    lower_bounds, upper_bounds
        Shape (d,).
    log_scale
        Must match the forward mapping.

    Returns
    -------
    parameters
        Shape (..., d).
    """
    X = np.asarray(X, dtype=float)
    lo = np.asarray(lower_bounds, dtype=float)
    hi = np.asarray(upper_bounds, dtype=float)

    if lo.ndim != 1 or hi.ndim != 1 or lo.shape != hi.shape:
        raise ValueError("lower_bounds and upper_bounds must be 1D arrays with equal shape.")
    if X.shape[-1] != lo.size:
        raise ValueError("Last GP-coordinate dimension does not match bounds.")
    if np.any(hi <= lo):
        raise ValueError("Each upper bound must be greater than its lower bound.")

    if log_scale:
        if np.any(lo <= 0) or np.any(hi <= 0):
            raise ValueError("log_scale=True requires strictly positive bounds.")
        lo_t = np.log(lo)
        hi_t = np.log(hi)
        return np.exp(lo_t + X * (hi_t - lo_t))

    return lo + X * (hi - lo)


# ---------------------------------------------------------------------------
# GP object and fitting
# ---------------------------------------------------------------------------

@dataclass
class GPValidityModel:
    """
    Frozen description of a learned behavioral-validity GP.

    The fitted sklearn GP is trained on scaled validity values

        y = v / validity_scale,

    so the sign of the GP mean is identical to the sign of v.

    ``custom_joint_prior`` returns ``np.inf`` for GP-rejected parameter tuples
    and ``0.0`` for GP-retained tuples, matching the custom-prior convention
    used in the attached GP-filtered Bioscrape inference script.
    """

    gp: GaussianProcessRegressor
    param_names: Tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    validity_scale: float
    log_scale: bool = True
    epsilon: Optional[float] = None
    posterior_quantile: Optional[float] = None
    gp_noise: float = 1e-8

    def parameters_to_gp(self, parameters: ArrayLike) -> np.ndarray:
        return map_parameters_to_gp_space(
            parameters,
            self.lower_bounds,
            self.upper_bounds,
            log_scale=self.log_scale,
        )

    def gp_to_parameters(self, X: ArrayLike) -> np.ndarray:
        return map_gp_space_to_parameters(
            X,
            self.lower_bounds,
            self.upper_bounds,
            log_scale=self.log_scale,
        )

    def _dict_to_vector(self, params_dict: Mapping[str, float]) -> np.ndarray:
        missing = [name for name in self.param_names if name not in params_dict]
        if missing:
            raise KeyError(f"Missing GP parameters: {missing}")
        return np.asarray([params_dict[name] for name in self.param_names], dtype=float)

    def predict(
        self,
        parameters: ArrayLike,
        *,
        return_std: bool = False,
        physical_validity_scale: bool = True,
    ):
        """
        Predict validity at physical parameter values.

        By default the returned mean and standard deviation are rescaled to the
        physical v(beta) units used to train the GP.
        """
        X = self.parameters_to_gp(parameters)
        X2 = np.atleast_2d(X)
        if return_std:
            mu, std = self.gp.predict(X2, return_std=True)
            if physical_validity_scale:
                mu = mu * self.validity_scale
                std = std * self.validity_scale
            return mu, std

        mu = self.gp.predict(X2)
        if physical_validity_scale:
            mu = mu * self.validity_scale
        return mu

    def sign_misclassification_probability(self, parameters: ArrayLike) -> np.ndarray:
        X = self.parameters_to_gp(parameters)
        return gp_sign_misclassification_probability(self.gp, np.atleast_2d(X))

    def retained_probability(self, parameters: ArrayLike) -> np.ndarray:
        X = self.parameters_to_gp(parameters)
        return gp_retained_probability(self.gp, np.atleast_2d(X))

    def custom_joint_prior(
        self,
        params_dict: Mapping[str, float],
        *,
        retained_probability_threshold: Optional[float] = None,
    ) -> float:
        """
        GP-based hard prior for use with :func:`py_inference`.

        Parameters
        ----------
        params_dict
            Dictionary of physical parameter values.
        retained_probability_threshold
            If None, retain when GP posterior mean >= 0, matching the attached
            ``exp2_gp_inference`` script.

            If supplied, retain when
                P_GP[v(beta) >= 0] >= threshold.

        Returns
        -------
        float
            0.0 if retained, np.inf if rejected.

        Notes
        -----
        This uses a *cost / negative-log-prior* convention for the custom prior:
        finite zero means "allowed" and +infinity means "reject".
        """
        beta = self._dict_to_vector(params_dict)

        if self.log_scale and np.any(beta <= 0):
            return np.inf
        if np.any(beta < self.lower_bounds) or np.any(beta > self.upper_bounds):
            return np.inf

        X = self.parameters_to_gp(beta).reshape(1, -1)

        if retained_probability_threshold is None:
            mu = float(self.gp.predict(X)[0])
            return 0.0 if mu >= 0.0 else np.inf

        if not (0.0 < retained_probability_threshold <= 1.0):
            raise ValueError("retained_probability_threshold must lie in (0, 1].")

        p_keep = float(gp_retained_probability(self.gp, X)[0])
        return 0.0 if p_keep >= retained_probability_threshold else np.inf


def fit_validity_gp(
    X_train: ArrayLike,
    validity_values: ArrayLike,
    *,
    initial_length_scale: Union[float, ArrayLike] = 0.25,
    length_scale_bounds: Tuple[float, float] = (0.01, 2.0),
    constant_value: float = 1.0,
    constant_value_bounds: Tuple[float, float] = (1e-3, 1e3),
    gp_noise: float = 1e-8,
    n_restarts_optimizer: int = 5,
    random_state: Optional[int] = 123,
    validity_scale: Optional[float] = None,
) -> Tuple[GaussianProcessRegressor, float]:
    """
    Fit the same ConstantKernel * RBF GP structure used in the GP-MPM notebook.

    Returns
    -------
    gp
        Fitted GaussianProcessRegressor.
    validity_scale
        Scale used to train the GP on validity_values / validity_scale.
    """
    X = np.asarray(X_train, dtype=float)
    v = np.asarray(validity_values, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X_train must have shape (n, d).")
    if X.shape[0] != v.size:
        raise ValueError("X_train and validity_values have inconsistent sample counts.")

    d = X.shape[1]

    if validity_scale is None:
        validity_scale = float(np.std(v))
        if validity_scale < 1e-12:
            validity_scale = 1.0
    else:
        validity_scale = float(validity_scale)
        if validity_scale <= 0:
            raise ValueError("validity_scale must be positive.")

    if np.isscalar(initial_length_scale):
        length_scale = np.full(d, float(initial_length_scale))
    else:
        length_scale = np.asarray(initial_length_scale, dtype=float)
        if length_scale.shape != (d,):
            raise ValueError("initial_length_scale array must have shape (d,).")

    kernel = ConstantKernel(
        constant_value,
        constant_value_bounds=constant_value_bounds,
    ) * RBF(
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds,
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(gp_noise),
        normalize_y=False,
        n_restarts_optimizer=int(n_restarts_optimizer),
        random_state=random_state,
    )
    gp.fit(X, v / validity_scale)
    return gp, validity_scale


def reconstruct_validity_gp(
    X_train: ArrayLike,
    validity_values: ArrayLike,
    *,
    validity_scale: float,
    kernel_constant: float,
    kernel_length_scale: ArrayLike,
    gp_noise: float = 1e-8,
) -> GaussianProcessRegressor:
    """
    Reconstruct a previously trained GP using fixed kernel hyperparameters.

    This matches the reconstruction pattern used before GP-filtered inference.
    """
    X = np.asarray(X_train, dtype=float)
    v = np.asarray(validity_values, dtype=float).reshape(-1)
    ls = np.asarray(kernel_length_scale, dtype=float)

    kernel = ConstantKernel(
        float(kernel_constant),
        constant_value_bounds="fixed",
    ) * RBF(
        ls,
        length_scale_bounds="fixed",
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(gp_noise),
        optimizer=None,
        normalize_y=False,
    )
    gp.fit(X, v / float(validity_scale))
    return gp


def gp_sign_misclassification_probability(
    gp: GaussianProcessRegressor,
    X: ArrayLike,
    *,
    min_std: float = 1e-12,
) -> np.ndarray:
    """
    Posterior probability that the GP sign classifier is wrong:

        p_e(x) = 0.5 erfc(|mu(x)| / (sqrt(2) sigma(x))).
    """
    X = np.asarray(X, dtype=float)
    mu, std = gp.predict(np.atleast_2d(X), return_std=True)
    std = np.maximum(std, min_std)
    return 0.5 * erfc(np.abs(mu) / (np.sqrt(2.0) * std))


def gp_retained_probability(
    gp: GaussianProcessRegressor,
    X: ArrayLike,
    *,
    min_std: float = 1e-12,
) -> np.ndarray:
    """
    GP posterior probability that v(x) >= 0:

        p_K(x) = Phi(mu(x) / sigma(x)).
    """
    X = np.asarray(X, dtype=float)
    mu, std = gp.predict(np.atleast_2d(X), return_std=True)
    std = np.maximum(std, min_std)
    return norm.cdf(mu / std)


# ---------------------------------------------------------------------------
# GP-MPM acquisition and diagnostics
# ---------------------------------------------------------------------------

def select_gp_mpm_candidate(
    gp: GaussianProcessRegressor,
    X_integration: ArrayLike,
    X_candidates: ArrayLike,
    *,
    mpm_alpha: float = 2.0,
    gp_noise: float = 1e-8,
    min_variance: float = 1e-12,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Select the candidate that maximizes the local expected reduction in GP
    sign-misclassification probability.

    This factors the main acquisition block from the attached GP-MPM notebook
    into one reusable method.

    Returns
    -------
    x_star
        Shape (d,).
    best_index
        Index into X_candidates.
    acquisition_scores
        Shape (n_candidates,).
    """
    X_int = np.asarray(X_integration, dtype=float)
    X_cand = np.asarray(X_candidates, dtype=float)

    if X_int.ndim != 2 or X_cand.ndim != 2:
        raise ValueError("X_integration and X_candidates must both be 2D.")
    if X_int.shape[1] != X_cand.shape[1]:
        raise ValueError("Integration and candidate points must have equal dimension.")
    if not hasattr(gp, "X_train_") or not hasattr(gp, "L_"):
        raise ValueError("gp must already be fitted.")

    mu_current, std_current = gp.predict(X_int, return_std=True)
    std_current = np.maximum(std_current, np.sqrt(min_variance))

    p_error_current = 0.5 * erfc(
        np.abs(mu_current) / (np.sqrt(2.0) * std_current)
    )

    K_int_train = gp.kernel_(X_int, gp.X_train_)
    A_int = solve_triangular(gp.L_, K_int_train.T, lower=True)

    try:
        current_length_scales = np.asarray(gp.kernel_.k2.length_scale, dtype=float)
    except AttributeError as exc:
        raise ValueError(
            "select_gp_mpm_candidate expects a fitted ConstantKernel * RBF GP."
        ) from exc

    if current_length_scales.ndim == 0:
        current_length_scales = np.full(X_int.shape[1], float(current_length_scales))

    acquisition_scores = np.zeros(X_cand.shape[0], dtype=float)
    var_current = std_current ** 2

    for q, x_candidate in enumerate(X_cand):
        x2 = x_candidate.reshape(1, -1)

        K_train_candidate = gp.kernel_(gp.X_train_, x2)
        A_candidate = solve_triangular(gp.L_, K_train_candidate, lower=True)

        k_int_candidate = gp.kernel_(X_int, x2).reshape(-1)
        cov_int_candidate = (
            k_int_candidate - (A_int.T @ A_candidate).reshape(-1)
        )

        k_candidate_candidate = float(gp.kernel_(x2, x2)[0, 0])
        var_candidate = k_candidate_candidate - float(np.sum(A_candidate ** 2))
        var_candidate = max(var_candidate, min_variance)

        var_after_sample = (
            var_current
            - cov_int_candidate ** 2 / (var_candidate + float(gp_noise))
        )
        var_after_sample = np.maximum(var_after_sample, min_variance)
        std_after_sample = np.sqrt(var_after_sample)

        p_error_after = 0.5 * erfc(
            np.abs(mu_current) / (np.sqrt(2.0) * std_after_sample)
        )

        scaled_distance = np.sqrt(
            np.sum(
                ((X_int - x_candidate) / current_length_scales) ** 2,
                axis=1,
            )
        )
        local_region = scaled_distance <= float(mpm_alpha)

        if np.any(local_region):
            reduction = p_error_current - p_error_after
            # Match the notebook: normalize by the total integration sample count.
            acquisition_scores[q] = (
                np.sum(reduction[local_region]) / X_int.shape[0]
            )

    best_index = int(np.argmax(acquisition_scores))
    return X_cand[best_index].copy(), best_index, acquisition_scores


def evaluate_gp_classifier(
    gp: GaussianProcessRegressor,
    X_test: ArrayLike,
    true_valid: ArrayLike,
) -> Dict[str, float]:
    """
    Evaluate the sign classifier against known retained/rejected labels.

    Returns the metrics used in the GP-MPM notebook.
    """
    X = np.asarray(X_test, dtype=float)
    truth = np.asarray(true_valid, dtype=bool).reshape(-1)

    mu, std = gp.predict(X, return_std=True)
    predicted = mu >= 0.0

    if predicted.size != truth.size:
        raise ValueError("X_test and true_valid have inconsistent sample counts.")

    overall = float(np.mean(predicted != truth))

    if np.any(truth):
        false_rejection = float(np.mean(~predicted[truth]))
    else:
        false_rejection = np.nan

    if np.any(~truth):
        false_acceptance = float(np.mean(predicted[~truth]))
    else:
        false_acceptance = np.nan

    balanced = float(np.nanmean([false_rejection, false_acceptance]))
    mean_perr = float(np.mean(
        0.5 * erfc(
            np.abs(mu) / (
                np.sqrt(2.0) * np.maximum(std, 1e-12)
            )
        )
    ))

    return {
        "misclassification": overall,
        "false_rejection": false_rejection,
        "false_acceptance": false_acceptance,
        "balanced_error": balanced,
        "mean_posterior_misclassification_probability": mean_perr,
    }


@dataclass
class GPMpmResult:
    validity_model: GPValidityModel
    X_train: np.ndarray
    parameters_train: np.ndarray
    discrepancy_train: np.ndarray
    validity_train: np.ndarray
    selected_discrepancy: np.ndarray
    acquisition_history: np.ndarray
    diagnostics_history: Tuple[Dict[str, float], ...] = field(default_factory=tuple)


def run_gp_mpm(
    evaluate_validity: Callable[[np.ndarray], Tuple[float, float]],
    *,
    param_names: Sequence[str],
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    log_scale: bool = True,
    n_initial: Optional[int] = None,
    n_steps: int = 100,
    n_integration_points: int = 2000,
    n_acquisition_candidates: int = 200,
    initial_length_scale: float = 0.25,
    length_scale_bounds: Tuple[float, float] = (0.01, 2.0),
    gp_noise: float = 1e-8,
    mpm_alpha: float = 2.0,
    n_restarts_optimizer: int = 5,
    random_seed: int = 1234,
    gp_random_state: int = 123,
    X_initial: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    true_test_valid: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
    posterior_quantile: Optional[float] = None,
) -> GPMpmResult:
    """
    Sequential GP-MPM level-set learning.

    ``evaluate_validity(beta)`` must return

        (r_beta, v_beta)

    for one *physical* detailed-parameter vector beta. Model simulation and the
    definition of r(beta) therefore remain outside this generic routine.
    """
    names = tuple(param_names)
    lo = np.asarray(lower_bounds, dtype=float)
    hi = np.asarray(upper_bounds, dtype=float)
    d = len(names)

    if lo.shape != (d,) or hi.shape != (d,):
        raise ValueError("Bounds must have shape (len(param_names),).")

    rng = np.random.default_rng(random_seed)
    if n_initial is None:
        n_initial = 2 * d

    if X_initial is None:
        X_train = rng.uniform(0.0, 1.0, size=(int(n_initial), d))
    else:
        X_train = np.asarray(X_initial, dtype=float).copy()
        if X_train.ndim != 2 or X_train.shape[1] != d:
            raise ValueError("X_initial must have shape (n_initial, d).")

    discrepancy = []
    validity = []

    for x in X_train:
        beta = map_gp_space_to_parameters(
            x, lo, hi, log_scale=log_scale
        )
        r_beta, v_beta = evaluate_validity(beta)
        discrepancy.append(float(r_beta))
        validity.append(float(v_beta))

    discrepancy = np.asarray(discrepancy, dtype=float)
    validity = np.asarray(validity, dtype=float)

    gp, validity_scale = fit_validity_gp(
        X_train,
        validity,
        initial_length_scale=initial_length_scale,
        length_scale_bounds=length_scale_bounds,
        gp_noise=gp_noise,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=gp_random_state,
    )

    X_integration = rng.uniform(
        0.0, 1.0, size=(int(n_integration_points), d)
    )

    selected_r = []
    acquisition_history = []
    diagnostics = []

    if X_test is not None and true_test_valid is not None:
        diagnostics.append(
            evaluate_gp_classifier(gp, X_test, true_test_valid)
        )

    for _ in range(int(n_steps)):
        X_candidates = rng.uniform(
            0.0, 1.0, size=(int(n_acquisition_candidates), d)
        )

        x_star, _, scores = select_gp_mpm_candidate(
            gp,
            X_integration,
            X_candidates,
            mpm_alpha=mpm_alpha,
            gp_noise=gp_noise,
        )
        best_score = float(np.max(scores))

        beta_star = map_gp_space_to_parameters(
            x_star, lo, hi, log_scale=log_scale
        )
        r_star, v_star = evaluate_validity(beta_star)

        X_train = np.vstack((X_train, x_star))
        discrepancy = np.append(discrepancy, float(r_star))
        validity = np.append(validity, float(v_star))

        # Preserve the initial validity scaling, as in the notebook.
        gp, _ = fit_validity_gp(
            X_train,
            validity,
            initial_length_scale=np.asarray(gp.kernel_.k2.length_scale, dtype=float),
            length_scale_bounds=length_scale_bounds,
            constant_value=float(gp.kernel_.k1.constant_value),
            gp_noise=gp_noise,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=gp_random_state,
            validity_scale=validity_scale,
        )

        selected_r.append(float(r_star))
        acquisition_history.append(best_score)

        if X_test is not None and true_test_valid is not None:
            diagnostics.append(
                evaluate_gp_classifier(gp, X_test, true_test_valid)
            )

    validity_model = GPValidityModel(
        gp=gp,
        param_names=names,
        lower_bounds=lo,
        upper_bounds=hi,
        validity_scale=validity_scale,
        log_scale=log_scale,
        epsilon=epsilon,
        posterior_quantile=posterior_quantile,
        gp_noise=gp_noise,
    )

    parameters_train = map_gp_space_to_parameters(
        X_train,
        lo,
        hi,
        log_scale=log_scale,
    )

    return GPMpmResult(
        validity_model=validity_model,
        X_train=X_train,
        parameters_train=parameters_train,
        discrepancy_train=discrepancy,
        validity_train=validity,
        selected_discrepancy=np.asarray(selected_r, dtype=float),
        acquisition_history=np.asarray(acquisition_history, dtype=float),
        diagnostics_history=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# Behavioral support / high-level GP construction from reduced posterior
# ---------------------------------------------------------------------------

def weighted_rmse_discrepancy(
    reference_trajectories: np.ndarray,
    candidate_trajectory: np.ndarray,
    condition_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Discrepancy used in the attached GP notebook.

    Parameters
    ----------
    reference_trajectories
        Shape (M, C, T), where M is the number of reduced-posterior samples,
        C the number of experimental conditions, and T the number of timepoints.
    candidate_trajectory
        Shape (C, T).
    condition_weights
        Shape (C,). Defaults to equal weights summing to one.

    Returns
    -------
    D_by_sample
        Shape (M,). Each value is the weighted sum over conditions of
        per-condition time-domain RMSE.
    """
    R = np.asarray(reference_trajectories, dtype=float)
    F = np.asarray(candidate_trajectory, dtype=float)

    if R.ndim == 2:
        R = R[:, None, :]
    if F.ndim == 1:
        F = F[None, :]

    if R.ndim != 3:
        raise ValueError("reference_trajectories must have shape (M,C,T) or (M,T).")
    if F.shape != R.shape[1:]:
        raise ValueError(
            f"candidate trajectory shape {F.shape} does not match "
            f"reference trajectory condition/time shape {R.shape[1:]}."
        )

    C = R.shape[1]
    if condition_weights is None:
        w = np.full(C, 1.0 / C)
    else:
        w = np.asarray(condition_weights, dtype=float)
        if w.shape != (C,):
            raise ValueError("condition_weights must have shape (C,).")
        if np.any(w < 0) or np.sum(w) <= 0:
            raise ValueError("condition_weights must be nonnegative with positive sum.")
        w = w / np.sum(w)

    diff = R - F[None, :, :]
    rmse_by_condition = np.sqrt(np.mean(diff ** 2, axis=2))
    return np.sum(rmse_by_condition * w[None, :], axis=1)


def _flatten_chain(
    chain: np.ndarray,
    *,
    discard: int = 0,
    thin: int = 1,
) -> np.ndarray:
    """Convert common emcee chain layouts to shape (n_samples, n_parameters)."""
    arr = np.asarray(chain, dtype=float)

    if arr.ndim == 2:
        if discard:
            arr = arr[discard:]
        return arr[::thin]

    if arr.ndim == 3:
        # Native emcee chain: (n_steps, n_walkers, n_dim)
        arr = arr[discard::thin]
        return arr.reshape(-1, arr.shape[-1])

    raise ValueError("chain must have shape (samples,d) or (steps,walkers,d).")


@dataclass
class BehavioralGPBuildResult:
    """
    Complete output of :func:`build_behavioral_gp_from_reduced_posterior`.
    """

    gp_mpm: GPMpmResult
    reduced_posterior_samples: np.ndarray
    reduced_posterior_trajectories: np.ndarray
    data_discrepancy_by_sample: np.ndarray
    epsilon: float
    data_discrepancy_threshold: float
    posterior_quantile: float
    data_discrepancy_quantile: float
    epsilon_multiplier: float
    condition_weights: np.ndarray

    @property
    def validity_model(self) -> GPValidityModel:
        return self.gp_mpm.validity_model

    @property
    def custom_joint_prior(self):
        """
        Convenience handle for

            py_inference(...,
                         custom_joint_prior=result.custom_joint_prior)
        """
        return self.validity_model.custom_joint_prior


def build_behavioral_gp_from_reduced_posterior(
    reduced_chain: np.ndarray,
    *,
    reduced_param_names: Sequence[str],
    reduced_trajectory_fn: TrajectoryFn,
    detailed_param_names: Sequence[str],
    detailed_lower_bounds: ArrayLike,
    detailed_upper_bounds: ArrayLike,
    detailed_trajectory_fn: TrajectoryFn,
    observed_trajectory: np.ndarray,
    condition_weights: Optional[ArrayLike] = None,
    chain_discard: int = 0,
    chain_thin: int = 1,
    max_reduced_samples: Optional[int] = None,
    reduced_sample_seed: int = 2026,
    posterior_quantile: float = 0.5,
    data_discrepancy_quantile: float = 0.95,
    epsilon_multiplier: float = 3.0,
    discrepancy_fn: DiscrepancyFn = weighted_rmse_discrepancy,
    detailed_log_scale: bool = True,
    n_initial: Optional[int] = None,
    n_mpm_steps: int = 100,
    n_integration_points: int = 2000,
    n_acquisition_candidates: int = 200,
    initial_length_scale: float = 0.25,
    length_scale_bounds: Tuple[float, float] = (0.01, 2.0),
    gp_noise: float = 1e-8,
    mpm_alpha: float = 2.0,
    n_restarts_optimizer: int = 5,
    random_seed: int = 1234,
    gp_random_state: int = 123,
    X_initial: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    true_test_valid: Optional[np.ndarray] = None,
) -> BehavioralGPBuildResult:
    """
    High-level wrapper that constructs the behavioral-validity GP directly from
    a reduced-model posterior chain.

    This is the model-agnostic counterpart of the workflow spread across the
    attached GP notebook.

    Required model-specific inputs are *callables*:
      - ``reduced_trajectory_fn(params_dict) -> trajectory``
      - ``detailed_trajectory_fn(params_dict) -> trajectory``

    The trajectories may have shape (T,) for one condition or (C,T) for
    multiple conditions.

    Workflow
    --------
    1. Flatten/subsample the reduced posterior chain.
    2. Simulate the reduced trajectory for each retained posterior sample using
       ``reduced_trajectory_fn``.
    3. Compute the reduced-posterior discrepancy from the observed data.
    4. Calibrate
           epsilon =
               epsilon_multiplier *
               quantile(D_data_reduced_posterior,
                        data_discrepancy_quantile).
    5. For each detailed beta queried by GP-MPM:
           r(beta) =
               quantile(D(reduced_posterior_trajectory, detailed(beta)),
                        posterior_quantile),
           v(beta) = epsilon - r(beta).
    6. Learn the zero level set of v(beta) with GP-MPM.
    7. Return a ``GPValidityModel`` whose ``custom_joint_prior`` can be passed
       directly to :func:`py_inference`.

    This function intentionally contains no model-specific simulation code.
    """
    if not (0.0 <= posterior_quantile <= 1.0):
        raise ValueError("posterior_quantile must lie in [0,1].")
    if not (0.0 <= data_discrepancy_quantile <= 1.0):
        raise ValueError("data_discrepancy_quantile must lie in [0,1].")
    if epsilon_multiplier <= 0:
        raise ValueError("epsilon_multiplier must be positive.")

    reduced_names = tuple(reduced_param_names)
    detailed_names = tuple(detailed_param_names)

    samples = _flatten_chain(
        reduced_chain,
        discard=int(chain_discard),
        thin=int(chain_thin),
    )
    if samples.shape[1] != len(reduced_names):
        raise ValueError(
            "Reduced chain parameter dimension does not match reduced_param_names."
        )

    if max_reduced_samples is not None and samples.shape[0] > max_reduced_samples:
        rng_sub = np.random.default_rng(reduced_sample_seed)
        idx = rng_sub.choice(
            samples.shape[0],
            size=int(max_reduced_samples),
            replace=False,
        )
        samples = samples[idx]

    reduced_trajectories = []
    for sample in samples:
        params = dict(zip(reduced_names, sample))
        traj = np.asarray(reduced_trajectory_fn(params), dtype=float)
        if traj.ndim == 1:
            traj = traj[None, :]
        if traj.ndim != 2:
            raise ValueError(
                "reduced_trajectory_fn must return shape (T,) or (C,T)."
            )
        reduced_trajectories.append(traj)

    reduced_trajectories = np.stack(reduced_trajectories, axis=0)

    observed = np.asarray(observed_trajectory, dtype=float)
    if observed.ndim == 1:
        observed = observed[None, :]
    if observed.shape != reduced_trajectories.shape[1:]:
        raise ValueError(
            "observed_trajectory shape must match reduced trajectory shape (C,T)."
        )

    C = observed.shape[0]
    if condition_weights is None:
        weights = np.full(C, 1.0 / C)
    else:
        weights = np.asarray(condition_weights, dtype=float)
        if weights.shape != (C,):
            raise ValueError("condition_weights must have shape (C,).")
        if np.any(weights < 0) or np.sum(weights) <= 0:
            raise ValueError("condition_weights must be nonnegative with positive sum.")
        weights = weights / np.sum(weights)

    # Treat the observed data as one candidate trajectory and compute one
    # discrepancy for every reduced-posterior sample.
    D_data = discrepancy_fn(
        reduced_trajectories,
        observed,
        weights,
    )
    D_data = np.asarray(D_data, dtype=float).reshape(-1)

    data_threshold = float(
        np.quantile(D_data, data_discrepancy_quantile)
    )
    epsilon = float(epsilon_multiplier * data_threshold)

    def evaluate_validity(beta_vector: np.ndarray) -> Tuple[float, float]:
        beta_dict = dict(zip(detailed_names, np.asarray(beta_vector, dtype=float)))
        detailed_traj = np.asarray(
            detailed_trajectory_fn(beta_dict),
            dtype=float,
        )
        if detailed_traj.ndim == 1:
            detailed_traj = detailed_traj[None, :]

        D_by_sample = discrepancy_fn(
            reduced_trajectories,
            detailed_traj,
            weights,
        )
        r_beta = float(
            np.quantile(
                np.asarray(D_by_sample, dtype=float),
                posterior_quantile,
            )
        )
        v_beta = float(epsilon - r_beta)
        return r_beta, v_beta

    gp_mpm = run_gp_mpm(
        evaluate_validity,
        param_names=detailed_names,
        lower_bounds=detailed_lower_bounds,
        upper_bounds=detailed_upper_bounds,
        log_scale=detailed_log_scale,
        n_initial=n_initial,
        n_steps=n_mpm_steps,
        n_integration_points=n_integration_points,
        n_acquisition_candidates=n_acquisition_candidates,
        initial_length_scale=initial_length_scale,
        length_scale_bounds=length_scale_bounds,
        gp_noise=gp_noise,
        mpm_alpha=mpm_alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        random_seed=random_seed,
        gp_random_state=gp_random_state,
        X_initial=X_initial,
        X_test=X_test,
        true_test_valid=true_test_valid,
        epsilon=epsilon,
        posterior_quantile=posterior_quantile,
    )

    return BehavioralGPBuildResult(
        gp_mpm=gp_mpm,
        reduced_posterior_samples=samples,
        reduced_posterior_trajectories=reduced_trajectories,
        data_discrepancy_by_sample=D_data,
        epsilon=epsilon,
        data_discrepancy_threshold=data_threshold,
        posterior_quantile=float(posterior_quantile),
        data_discrepancy_quantile=float(data_discrepancy_quantile),
        epsilon_multiplier=float(epsilon_multiplier),
        condition_weights=weights,
    )


# ---------------------------------------------------------------------------
# emcee-native py_inference
# ---------------------------------------------------------------------------

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
        # Match the convention used in the attached exp2_gp_inference script:
        # custom_joint_prior == 0 means allowed, +inf means reject. Finite
        # positive values act as an additional negative-log-prior penalty.
        penalty = float(custom_joint_prior(params))
        if not np.isfinite(penalty):
            return -np.inf
        logp -= penalty

    return float(logp)


@dataclass
class InferenceResult:
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
    pool: Any = None,
    moves: Any = None,
    vectorize: bool = False,
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
        Optional callable with the same convention used in the attached
        GP-filtered inference script:

            custom_joint_prior(params_dict) == 0.0    -> allowed
            custom_joint_prior(params_dict) == inf    -> reject
            finite positive value                     -> extra -log-prior penalty

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
    if pool is not None:
        sampler_kwargs["pool"] = pool
    if vectorize:
        sampler_kwargs["vectorize"] = True

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        **sampler_kwargs,
    )

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
            n_postburn = flat_samples.shape[0]
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


__all__ = [
    "BehavioralGPBuildResult",
    "GPMpmResult",
    "GPValidityModel",
    "InferenceResult",
    "build_behavioral_gp_from_reduced_posterior",
    "evaluate_gp_classifier",
    "fit_validity_gp",
    "gp_retained_probability",
    "gp_sign_misclassification_probability",
    "map_gp_space_to_parameters",
    "map_parameters_to_gp_space",
    "py_inference",
    "reconstruct_validity_gp",
    "run_gp_mpm",
    "select_gp_mpm_candidate",
    "weighted_rmse_discrepancy",
]
