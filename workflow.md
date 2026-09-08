A workflow for using implementation_methods.py to learn a Gaussian Process and use it for Bayesian parameter inference:

```
gp_result = build_behavioral_gp_from_reduced_posterior(
    reduced_chain,
    reduced_param_names=...,
    reduced_trajectory_fn=...,
    detailed_param_names=...,
    detailed_lower_bounds=...,
    detailed_upper_bounds=...,
    detailed_trajectory_fn=...,
    observed_trajectory=...,
    ...
)

result = py_inference(
    likelihood=log_likelihood,
    data=data,
    params_to_estimate=detailed_param_names,
    prior=prior,
    custom_joint_prior=gp_result.custom_joint_prior,
    ...
)
```
