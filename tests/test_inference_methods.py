import numpy as np

from hierarch_inf import (
    build_behavioral_gp_from_reduced_posterior,
    map_gp_space_to_parameters,
    map_parameters_to_gp_space,
    py_inference,
)


def test_log_parameter_coordinates_round_trip():
    lower = np.array([0.1, 0.001])
    upper = np.array([1000.0, 10.0])
    physical = np.array([[0.1, 0.001], [10.0, 0.1], [1000.0, 10.0]])
    normalized = map_parameters_to_gp_space(physical, lower, upper, log_scale=True)
    np.testing.assert_allclose(normalized, [[0, 0], [0.5, 0.5], [1, 1]])
    np.testing.assert_allclose(
        map_gp_space_to_parameters(normalized, lower, upper, log_scale=True), physical
    )


def test_inference_seed_controls_proposals_and_preserves_global_rng():
    def fit():
        return py_inference(
            likelihood=lambda parameters, _: -0.5 * parameters["theta"]**2,
            data=None,
            params_to_estimate=["theta"],
            prior={"theta": ["normal", 0.0, 1.0]},
            nwalkers=8, nsteps=24, discard=4,
            random_seed=42, compute_diagnostics=False,
        )

    np.random.seed(5)
    before = np.random.get_state()
    first = fit()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.random.seed(9876)
    second = fit()
    np.testing.assert_array_equal(first.chain, second.chain)


def test_thinning_does_not_spuriously_halve_ess():
    def fit(thin):
        return py_inference(
            likelihood=lambda parameters, _: -0.5 * parameters["theta"]**2,
            data=None,
            params_to_estimate=["theta"],
            prior={"theta": ["normal", 0.0, 1.0]},
            nwalkers=8, nsteps=2048, discard=256, thin=thin,
            random_seed=42, compute_diagnostics=True,
        )

    unthinned = fit(1)
    thinned = fit(2)
    np.testing.assert_array_equal(unthinned.chain, thinned.chain)
    assert thinned.flat_samples.shape[0] == unthinned.flat_samples.shape[0] // 2
    assert unthinned.ess is not None and thinned.ess is not None
    # Estimating autocorrelation from every other iteration can change the
    # estimate slightly, but should not introduce a factor-of-two decrease.
    np.testing.assert_allclose(thinned.ess, unthinned.ess, rtol=0.1)


def test_behavioral_screen_precedes_expensive_likelihood():
    evaluated = []

    def likelihood(parameters, _):
        assert parameters["theta"] >= 0.0
        evaluated.append(parameters["theta"])
        return -parameters["theta"]**2

    result = py_inference(
        likelihood=likelihood, data=None,
        params_to_estimate=["theta"], prior={"theta": ["uniform", -1.0, 1.0]},
        custom_joint_prior=lambda parameters: np.inf if parameters["theta"] < 0 else 0.0,
        nwalkers=8, nsteps=24, discard=4,
        random_seed=8, compute_diagnostics=False,
    )
    assert evaluated
    assert (result.chain >= 0.0).all()


def test_finite_joint_log_prior_recovers_analytical_posterior():
    # N(0, 1) prior, log likelihood -theta^2/2, and an additive 2*theta
    # log-prior contribution give a N(1, 1/2) posterior.
    result = py_inference(
        likelihood=lambda parameters, _: -0.5 * parameters["theta"]**2,
        data=None,
        params_to_estimate=["theta"],
        prior={"theta": ["normal", 0.0, 1.0]},
        custom_joint_prior=lambda parameters: 2.0 * parameters["theta"],
        nwalkers=16, nsteps=1000, discard=200,
        random_seed=42, compute_diagnostics=False,
    )
    samples = result.flat_samples[:, 0]
    assert abs(samples.mean() - 1.0) < 0.15
    assert abs(samples.var() - 0.5) < 0.15


def test_support_estimate_and_retention_diagnostic_have_known_values():
    reduced_samples = np.array([[0.4], [0.5], [0.6]])
    result = build_behavioral_gp_from_reduced_posterior(
        reduced_samples,
        reduced_param_names=["alpha"],
        reduced_trajectory_fn=lambda parameters: np.full(3, parameters["alpha"]),
        detailed_param_names=["theta"],
        detailed_lower_bounds=[0.0], detailed_upper_bounds=[1.0],
        detailed_trajectory_fn=lambda parameters: np.full(3, parameters["theta"]),
        observed_trajectory=np.full(3, 0.5),
        epsilon=0.15, posterior_level_set_boundary=0.5, data_discrepancy_threshold=0.1,
        detailed_log_scale=False, n_initial=4, n_mpm_steps=1,
        n_integration_points=16, n_acquisition_candidates=8,
        n_restarts_optimizer=0, random_seed=11,
    )
    theta = result.gp_mpm.parameters_train[:, 0]
    expected_support = (np.abs(theta[:, None] - reduced_samples[:, 0]) <= 0.15).mean(axis=1)
    np.testing.assert_allclose(result.gp_mpm.behavioral_support_train, expected_support)
    np.testing.assert_allclose(result.gp_mpm.validity_train, expected_support - 0.5)
    assert result.reduced_posterior_fit_mass == 1.0 / 3.0
    frozen = result.validity_model
    before = frozen.gp.X_train_.copy()
    assert frozen.custom_joint_prior({"theta": -0.1}) == np.inf
    assert frozen.custom_joint_prior({"theta": 1.1}) == np.inf
    frozen.custom_joint_prior({"theta": 0.5})
    np.testing.assert_array_equal(frozen.gp.X_train_, before)
