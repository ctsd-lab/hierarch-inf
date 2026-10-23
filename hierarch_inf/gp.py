"""Gaussian-process learning and fixed behavioral-support filtering.

Model-specific simulations and likelihoods are supplied by the caller.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import erfc
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


ArrayLike = Union[np.ndarray, Sequence[float]]


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


@dataclass
class GPValidityModel:
    """
    Trained GP approximation of the behavioral-support level set.

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
        """Map physical parameters to the coordinates used by this GP."""
        return map_parameters_to_gp_space(
            parameters,
            self.lower_bounds,
            self.upper_bounds,
            log_scale=self.log_scale,
        )

    def _dict_to_vector(self, params_dict: Mapping[str, float]) -> np.ndarray:
        """Order a parameter mapping using the stored parameter names."""
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
        GP is trained and held fixed before detailed-model MCMC, then evaluated at
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

    if np.ndim(initial_length_scale) == 0:
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
        trained GP gives ``p_K(theta) <= zeta``.
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
