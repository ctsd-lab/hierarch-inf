"""Build behavioral compatibility from reduced-posterior trajectories.

Model-specific simulations and likelihoods are supplied by the caller.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from .gp import GPValidityModel, GPMpmResult, run_gp_mpm


ArrayLike = Union[np.ndarray, Sequence[float]]
ParamsDict = Dict[str, float]
TrajectoryFn = Callable[[ParamsDict], np.ndarray]
DiscrepancyFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]


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
        """Return the fitted, fixed GP used to screen detailed-model proposals."""
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
        GP is held fixed. During detailed-model MCMC, the custom prior rejects a
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

    8. Return a trained ``GPValidityModel`` whose ``custom_joint_prior`` can be
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
        """Evaluate compatibility against the fixed reduced-posterior trajectories."""
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
