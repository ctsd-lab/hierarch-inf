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

For a fixed finite-horizon compatibility tolerance ``epsilon`` and
posterior behavioral-support level-set boundary ``gamma``, the reduced-model
posterior induces

    b_epsilon(beta)
        = P_{theta_hat ~ pi_hat(. | y_bar)}
          [d(y(beta), y_hat(theta_hat)) <= epsilon].

The GP learns the sign of

    v_epsilon,gamma(beta) = b_epsilon(beta) - gamma.

The behaviorally retained detailed-parameter set is therefore

    K_epsilon,gamma = {beta : v_epsilon,gamma(beta) >= 0}.

With reduced-posterior MCMC samples, ``b_epsilon`` is estimated by the empirical
fraction of stored reduced trajectories whose discrepancy from the detailed
trajectory is at most ``epsilon``.

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
    Frozen GP approximation of the behavioral-support level set.

    For compatibility tolerance ``epsilon`` and behavioral-support level-set
    boundary ``posterior_level_set_boundary = gamma``, the GP target is

        v_epsilon,gamma(theta)
            = b_epsilon(theta) - gamma,

    where

        b_epsilon(theta)
            = P_{theta_hat ~ pi_hat(. | y_bar)}
              [d(y(theta), y_hat(theta_hat)) <= epsilon].

    The fitted sklearn GP is trained on the scaled target

        y_gp = v_epsilon,gamma / validity_scale,

    so the zero level set is unchanged by scaling.

    Threshold definitions
    ---------------------
    epsilon
        Finite-horizon *behavioral compatibility tolerance*. A detailed/reduced
        pair is behaviorally compatible when

            d(y(theta), y_hat(theta_hat)) <= epsilon.

    posterior_level_set_boundary
        The support level ``gamma in (0,1)`` defining the retained set

            K_epsilon,gamma
                = {theta : b_epsilon(theta) >= gamma}

        and therefore the behavioral-support level-set boundary

            b_epsilon(theta) = gamma.

    data_discrepancy_threshold
        The detailed-model data-fit tolerance ``eta`` defining

            F_eta
                = {theta : d(y(theta), y_bar) <= eta}.

        It is stored with the GP object because the retention theorem relates
        ``F_eta`` to ``K_epsilon,gamma`` through

            gamma <= m_pi_hat(epsilon - eta).

    gp_screening_level
        The GP screening level ``zeta``. During production detailed-model MCMC,
        a proposal is screened when

            p_K(theta) = Phi(mu(theta) / sigma(theta)) <= zeta.

    ``custom_joint_prior`` implements this fixed-GP screening rule using the
    same custom-prior convention as the attached GP-filtered inference script:
    ``0.0`` means allowed and ``np.inf`` means rejected.
    """

    gp: GaussianProcessRegressor
    param_names: Tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    validity_scale: float
    log_scale: bool = True

    epsilon: Optional[float] = None
    posterior_level_set_boundary: Optional[float] = None
    data_discrepancy_threshold: Optional[float] = None
    gp_screening_level: float = 0.01
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
        return np.asarray(
            [params_dict[name] for name in self.param_names],
            dtype=float,
        )

    def predict(
        self,
        parameters: ArrayLike,
        *,
        return_std: bool = False,
        physical_validity_scale: bool = True,
    ):
        """
        Predict ``v_epsilon,gamma(theta)`` at physical parameter values.

        By default, the returned mean and standard deviation are rescaled from
        the GP training scale to behavioral-support units. The zero level set is

            v_epsilon,gamma(theta) = 0

        which is equivalent to

            b_epsilon(theta) = gamma.
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

    def sign_misclassification_probability(
        self,
        parameters: ArrayLike,
    ) -> np.ndarray:
        """Return the GP posterior probability of misclassifying the sign of v."""
        X = self.parameters_to_gp(parameters)
        return gp_sign_misclassification_probability(
            self.gp,
            np.atleast_2d(X),
        )

    def retained_probability(self, parameters: ArrayLike) -> np.ndarray:
        """
        Return

            p_K(theta) = P_GP[v_epsilon,gamma(theta) >= 0]
                       = Phi(mu(theta) / sigma(theta)).
        """
        X = self.parameters_to_gp(parameters)
        return gp_retained_probability(
            self.gp,
            np.atleast_2d(X),
        )

    def custom_joint_prior(
        self,
        params_dict: Mapping[str, float],
        *,
        gp_screening_level: Optional[float] = None,
    ) -> float:
        """
        Fixed-GP hard behavioral prior for :func:`py_inference`.

        Parameters
        ----------
        params_dict
            Dictionary of physical detailed-model parameter values.

        gp_screening_level
            Optional override of the stored GP screening level ``zeta``.
            A proposal is rejected when

                p_K(theta) <= zeta,

            where

                p_K(theta) = Phi(mu(theta) / sigma(theta)).

            If omitted, ``self.gp_screening_level`` is used.

        Returns
        -------
        float
            ``0.0`` if the proposal passes behavioral screening and ``np.inf``
            if it is screened.

        Notes
        -----
        This follows the production-inference rule in the updated theory: the
        GP is trained and frozen before detailed-model MCMC, then evaluated at
        every proposal before the expensive detailed-model likelihood.
        """
        beta = self._dict_to_vector(params_dict)

        if self.log_scale and np.any(beta <= 0):
            return np.inf
        if np.any(beta < self.lower_bounds) or np.any(beta > self.upper_bounds):
            return np.inf

        zeta = (
            self.gp_screening_level
            if gp_screening_level is None
            else float(gp_screening_level)
        )
        if not (0.0 <= zeta < 1.0):
            raise ValueError("gp_screening_level must lie in [0, 1).")

        X = self.parameters_to_gp(beta).reshape(1, -1)
        p_keep = float(gp_retained_probability(self.gp, X)[0])

        return np.inf if p_keep <= zeta else 0.0


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
    """
    Results of sequential GP-MPM learning of the behavioral-support level set.
    """

    validity_model: GPValidityModel
    X_train: np.ndarray
    parameters_train: np.ndarray

    # b_hat_epsilon,M(theta) at every detailed point evaluated for GP training.
    behavioral_support_train: np.ndarray

    # v_hat_epsilon,gamma,M(theta) = b_hat_epsilon,M(theta) - gamma.
    validity_train: np.ndarray

    # Behavioral-support values acquired after the initial design.
    selected_behavioral_support: np.ndarray

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
    posterior_level_set_boundary: Optional[float] = None,
    data_discrepancy_threshold: Optional[float] = None,
    gp_screening_level: float = 0.01,
) -> GPMpmResult:
    """
    Sequential GP-MPM learning of

        b_epsilon(theta) = gamma.

    ``evaluate_validity(theta)`` must return

        (b_hat, v_hat)

    where

        b_hat
            is the reduced-posterior Monte Carlo estimate

                b_hat_epsilon,M(theta)
                    = (1/M) sum_j
                      1[d(y(theta), y_hat(theta_hat^(j))) <= epsilon],

        v_hat
            is the scalar GP target

                v_hat_epsilon,gamma,M(theta)
                    = b_hat_epsilon,M(theta) - gamma.

    The routine is intentionally unaware of the detailed and reduced model
    implementations. One detailed simulation occurs inside
    ``evaluate_validity`` whenever GP-MPM selects a new parameter tuple.

    Threshold definitions
    ---------------------
    posterior_level_set_boundary
        ``gamma``: the behavioral-support value defining the target level set
        ``b_epsilon(theta) = gamma``.

    data_discrepancy_threshold
        ``eta``: the accepted detailed-model data-fit tolerance defining
        ``F_eta = {theta : d(y(theta), y_bar) <= eta}``. It is metadata for the
        retention statement; GP-MPM itself learns the behavioral-support level
        set.

    epsilon
        Finite-horizon detailed/reduced compatibility tolerance used inside
        the indicator defining ``b_hat_epsilon,M``.

    gp_screening_level
        ``zeta``: after training, production MCMC screens a proposal when the
        frozen GP gives ``p_K(theta) <= zeta``.
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

    support_values = []
    validity_values = []

    for x in X_train:
        theta = map_gp_space_to_parameters(
            x,
            lo,
            hi,
            log_scale=log_scale,
        )
        b_hat, v_hat = evaluate_validity(theta)
        support_values.append(float(b_hat))
        validity_values.append(float(v_hat))

    support_values = np.asarray(support_values, dtype=float)
    validity_values = np.asarray(validity_values, dtype=float)

    gp, validity_scale = fit_validity_gp(
        X_train,
        validity_values,
        initial_length_scale=initial_length_scale,
        length_scale_bounds=length_scale_bounds,
        gp_noise=gp_noise,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=gp_random_state,
    )

    X_integration = rng.uniform(
        0.0,
        1.0,
        size=(int(n_integration_points), d),
    )

    selected_support = []
    acquisition_history = []
    diagnostics = []

    if X_test is not None and true_test_valid is not None:
        diagnostics.append(
            evaluate_gp_classifier(gp, X_test, true_test_valid)
        )

    for _ in range(int(n_steps)):
        X_candidates = rng.uniform(
            0.0,
            1.0,
            size=(int(n_acquisition_candidates), d),
        )

        x_star, _, scores = select_gp_mpm_candidate(
            gp,
            X_integration,
            X_candidates,
            mpm_alpha=mpm_alpha,
            gp_noise=gp_noise,
        )
        best_score = float(np.max(scores))

        theta_star = map_gp_space_to_parameters(
            x_star,
            lo,
            hi,
            log_scale=log_scale,
        )
        b_star, v_star = evaluate_validity(theta_star)

        X_train = np.vstack((X_train, x_star))
        support_values = np.append(support_values, float(b_star))
        validity_values = np.append(validity_values, float(v_star))

        # Preserve the original validity scaling while updating the GP.
        gp, _ = fit_validity_gp(
            X_train,
            validity_values,
            initial_length_scale=np.asarray(
                gp.kernel_.k2.length_scale,
                dtype=float,
            ),
            length_scale_bounds=length_scale_bounds,
            constant_value=float(gp.kernel_.k1.constant_value),
            gp_noise=gp_noise,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=gp_random_state,
            validity_scale=validity_scale,
        )

        selected_support.append(float(b_star))
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
        posterior_level_set_boundary=posterior_level_set_boundary,
        data_discrepancy_threshold=data_discrepancy_threshold,
        gp_screening_level=float(gp_screening_level),
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
        behavioral_support_train=support_values,
        validity_train=validity_values,
        selected_behavioral_support=np.asarray(
            selected_support,
            dtype=float,
        ),
        acquisition_history=np.asarray(
            acquisition_history,
            dtype=float,
        ),
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

    Threshold definitions
    ---------------------
    posterior_level_set_boundary
        ``gamma``: the reduced-posterior behavioral-support level defining

            K_epsilon,gamma
                = {theta : b_epsilon(theta) >= gamma}

        with boundary ``b_epsilon(theta) = gamma``.

    data_discrepancy_threshold
        ``eta``: the accepted detailed-model discrepancy from the measured
        trajectory, defining

            F_eta = {theta : d(y(theta), y_bar) <= eta}.

    epsilon
        The finite-horizon behavioral compatibility tolerance between detailed
        and reduced outputs. It appears in

            b_epsilon(theta)
                = P[d(y(theta), y_hat(theta_hat)) <= epsilon].

        The updated retention theorem uses the three thresholds through

            gamma <= m_pi_hat(epsilon - eta)

        to guarantee ``F_eta subset K_epsilon,gamma``.
    """

    gp_mpm: GPMpmResult
    reduced_posterior_samples: np.ndarray
    reduced_posterior_trajectories: np.ndarray

    # d(y_hat(theta_hat^(j)), y_bar), one value per reduced posterior sample.
    data_discrepancy_by_sample: np.ndarray

    epsilon: float
    posterior_level_set_boundary: float
    data_discrepancy_threshold: float
    gp_screening_level: float

    # Empirical reduced-posterior mass m_pi_hat(epsilon - eta).
    reduced_posterior_fit_mass: float

    condition_weights: np.ndarray

    @property
    def validity_model(self) -> GPValidityModel:
        return self.gp_mpm.validity_model

    @property
    def custom_joint_prior(self):
        """
        Convenience handle for behaviorally guided detailed-model inference:

            py_inference(
                ...,
                custom_joint_prior=result.custom_joint_prior,
            )
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

    epsilon: float,
    posterior_level_set_boundary: float,
    data_discrepancy_threshold: float,
    gp_screening_level: float = 0.01,

    condition_weights: Optional[ArrayLike] = None,
    chain_discard: int = 0,
    chain_thin: int = 1,
    max_reduced_samples: Optional[int] = None,
    reduced_sample_seed: int = 2026,
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
    Build the behavioral-support GP directly from a reduced-model MCMC chain.

    This wrapper implements the updated theory language and training target.

    Required model-specific inputs
    ------------------------------
    reduced_trajectory_fn
        ``reduced_trajectory_fn(params_dict) -> trajectory``.

    detailed_trajectory_fn
        ``detailed_trajectory_fn(params_dict) -> trajectory``.

    observed_trajectory
        Measured output ``y_bar`` in the same trajectory representation used by
        the detailed and reduced trajectory functions.

    Theory thresholds
    -----------------
    posterior_level_set_boundary
        The behavioral-support level ``gamma in (0,1)``. This is the
        *posterior level-set boundary*. The GP learns

            b_epsilon(theta) = gamma,

        or equivalently the zero level set of

            v_epsilon,gamma(theta)
                = b_epsilon(theta) - gamma.

        Detailed parameters with ``b_epsilon(theta) >= gamma`` belong to the
        behaviorally retained set ``K_epsilon,gamma``.

    data_discrepancy_threshold
        The detailed-model data-fit tolerance ``eta > 0``. It defines the
        accepted data-fit set

            F_eta
                = {theta : d(y(theta), y_bar) <= eta}.

        The GP training target does not use ``eta`` directly. ``eta`` enters
        the retention condition

            gamma <= m_pi_hat(epsilon - eta),

        which guarantees that the data-fit region is contained in the
        behaviorally retained region.

    epsilon
        The finite-horizon behavioral compatibility tolerance ``epsilon > 0``.
        A reduced-posterior sample ``theta_hat^(j)`` supports a detailed
        parameter ``theta`` when

            d(
                y(theta),
                y_hat(theta_hat^(j))
            ) <= epsilon.

        Thus the Monte Carlo behavioral-support estimate used for GP training
        is

            b_hat_epsilon,M(theta)
                = (1/M) sum_j
                  1[
                    d(y(theta), y_hat(theta_hat^(j))) <= epsilon
                  ].

    gp_screening_level
        The GP screening level ``zeta``. Once GP-MPM training is complete, the
        GP is frozen. During detailed-model MCMC, the custom prior rejects a
        proposal when

            p_K(theta) = Phi(mu(theta) / sigma(theta)) <= zeta.

    Workflow
    --------
    1. Flatten/subsample the reduced posterior chain.
    2. Simulate and store one reduced trajectory for each retained posterior
       sample.
    3. Compute each stored reduced trajectory's discrepancy from the measured
       output.
    4. Compute the empirical reduced-posterior fit mass

           m_hat_pi(epsilon - eta)
               = fraction_j[
                   d(y_hat(theta_hat^(j)), y_bar)
                   <= epsilon - eta
                 ].

    5. At every detailed parameter selected by GP-MPM, simulate the detailed
       model once and compare that trajectory with every stored reduced
       posterior trajectory.
    6. Estimate ``b_hat_epsilon,M(theta)`` as the fraction of those
       discrepancies that are at most ``epsilon``.
    7. Train the GP on

           v_hat_epsilon,gamma,M(theta)
               = b_hat_epsilon,M(theta) - gamma.

    8. Return a frozen ``GPValidityModel`` whose ``custom_joint_prior`` can be
       supplied directly to :func:`py_inference`.

    Notes
    -----
    ``discrepancy_fn`` remains configurable so the notebook can supply the same
    finite-horizon output metric used by the model experiment.
    """
    epsilon = float(epsilon)
    gamma = float(posterior_level_set_boundary)
    eta = float(data_discrepancy_threshold)
    zeta = float(gp_screening_level)

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not (0.0 < gamma < 1.0):
        raise ValueError(
            "posterior_level_set_boundary must lie in (0, 1)."
        )
    if eta < 0:
        raise ValueError("data_discrepancy_threshold must be nonnegative.")
    if not (0.0 <= zeta < 1.0):
        raise ValueError("gp_screening_level must lie in [0, 1).")

    reduced_names = tuple(reduced_param_names)
    detailed_names = tuple(detailed_param_names)

    samples = _flatten_chain(
        reduced_chain,
        discard=int(chain_discard),
        thin=int(chain_thin),
    )
    if samples.shape[1] != len(reduced_names):
        raise ValueError(
            "Reduced chain parameter dimension does not match "
            "reduced_param_names."
        )

    if (
        max_reduced_samples is not None
        and samples.shape[0] > max_reduced_samples
    ):
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
        traj = np.asarray(
            reduced_trajectory_fn(params),
            dtype=float,
        )
        if traj.ndim == 1:
            traj = traj[None, :]
        if traj.ndim != 2:
            raise ValueError(
                "reduced_trajectory_fn must return shape (T,) or (C,T)."
            )
        reduced_trajectories.append(traj)

    reduced_trajectories = np.stack(
        reduced_trajectories,
        axis=0,
    )

    observed = np.asarray(observed_trajectory, dtype=float)
    if observed.ndim == 1:
        observed = observed[None, :]
    if observed.shape != reduced_trajectories.shape[1:]:
        raise ValueError(
            "observed_trajectory shape must match reduced trajectory "
            "shape (C,T)."
        )

    C = observed.shape[0]
    if condition_weights is None:
        weights = np.full(C, 1.0 / C)
    else:
        weights = np.asarray(condition_weights, dtype=float)
        if weights.shape != (C,):
            raise ValueError("condition_weights must have shape (C,).")
        if np.any(weights < 0) or np.sum(weights) <= 0:
            raise ValueError(
                "condition_weights must be nonnegative with positive sum."
            )
        weights = weights / np.sum(weights)

    # Reduced posterior distance from measured data:
    # d(y_hat(theta_hat^(j)), y_bar).
    D_data = np.asarray(
        discrepancy_fn(
            reduced_trajectories,
            observed,
            weights,
        ),
        dtype=float,
    ).reshape(-1)

    # Empirical counterpart of m_pi_hat(epsilon - eta).
    retention_radius = epsilon - eta
    if retention_radius < 0:
        reduced_posterior_fit_mass = 0.0
    else:
        reduced_posterior_fit_mass = float(
            np.mean(D_data <= retention_radius)
        )

    def evaluate_validity(
        theta_vector: np.ndarray,
    ) -> Tuple[float, float]:
        theta_dict = dict(
            zip(
                detailed_names,
                np.asarray(theta_vector, dtype=float),
            )
        )

        detailed_traj = np.asarray(
            detailed_trajectory_fn(theta_dict),
            dtype=float,
        )
        if detailed_traj.ndim == 1:
            detailed_traj = detailed_traj[None, :]

        D_by_sample = np.asarray(
            discrepancy_fn(
                reduced_trajectories,
                detailed_traj,
                weights,
            ),
            dtype=float,
        ).reshape(-1)

        # Monte Carlo behavioral support:
        # b_hat_epsilon,M(theta)
        b_hat = float(np.mean(D_by_sample <= epsilon))

        # GP target:
        # v_hat_epsilon,gamma,M(theta)
        v_hat = float(b_hat - gamma)

        return b_hat, v_hat

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
        posterior_level_set_boundary=gamma,
        data_discrepancy_threshold=eta,
        gp_screening_level=zeta,
    )

    return BehavioralGPBuildResult(
        gp_mpm=gp_mpm,
        reduced_posterior_samples=samples,
        reduced_posterior_trajectories=reduced_trajectories,
        data_discrepancy_by_sample=D_data,
        epsilon=epsilon,
        posterior_level_set_boundary=gamma,
        data_discrepancy_threshold=eta,
        gp_screening_level=zeta,
        reduced_posterior_fit_mass=reduced_posterior_fit_mass,
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
