"""Methods accompanying the hierarchical Bayesian inference paper."""

from .inference import InferenceResult, py_inference
from .behavioral import (
    BehavioralGPBuildResult,
    build_behavioral_gp_from_reduced_posterior,
    weighted_rmse_discrepancy,
)
from .gp import (
    GPMpmResult,
    GPValidityModel,
    evaluate_gp_classifier,
    fit_validity_gp,
    gp_retained_probability,
    map_gp_space_to_parameters,
    map_parameters_to_gp_space,
    run_gp_mpm,
    select_gp_mpm_candidate,
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
    "map_gp_space_to_parameters",
    "map_parameters_to_gp_space",
    "py_inference",
    "run_gp_mpm",
    "select_gp_mpm_candidate",
    "weighted_rmse_discrepancy",
]
