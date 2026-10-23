"""Prepare the recorded GP-based initial ensemble for mechanistic inference.

Initialization settings are module constants. Generated files are stored under
examples/data/.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import hashlib

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = EXAMPLES_DIR / "data"

GP_MPM_FILE = (EXAMPLES_DIR / "results/gp_mpm_exp3_result_eps0p5_1020.npz")
OUTPUT_FILE = (DATA_DIR / "exp3_1_initial_walkers_eps0p5_1020.npz")
NWALKERS = 256
RANDOM_SEED = 2026
HALF_WIDTH_LOG10 = 0.02
N_CANDIDATES = 10000
INITIAL_SCREENING_LEVEL = 0.8


def main():
    """Prepare the recorded GP-based initial ensemble for mechanistic inference."""
    with np.load(GP_MPM_FILE, allow_pickle=False) as record:
        X = record["X_gp"]
        targets = record["v_gp"]
        discrepancies = record["rho_gp"]
        param_names = record["detailed_param_names"]
        lower = record["lower_bounds_detailed"]
        upper = record["upper_bounds_detailed"]
        centre_index = int(np.argmin(discrepancies))
        centre_x = X[centre_index]
        centre = np.exp(np.log(lower) + centre_x * np.log(upper / lower))
        centre_discrepancy = float(discrepancies[centre_index])
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(float(record["gp_kernel_constant"]), "fixed")
            * RBF(record["gp_kernel_length_scale"], "fixed"),
            alpha=float(record["gp_noise"]), optimizer=None, normalize_y=False,
        )
        gp.fit(X, targets / float(record["validity_scale"]))

    rng = np.random.default_rng(RANDOM_SEED)
    candidates = 10 ** (
        np.log10(centre)
        + rng.uniform(-HALF_WIDTH_LOG10, HALF_WIDTH_LOG10,
                      size=(N_CANDIDATES, len(param_names)))
    )
    within_prior = np.all((candidates >= lower) & (candidates <= upper), axis=1)
    candidates = candidates[within_prior]
    normalized = np.log(candidates / lower) / np.log(upper / lower)
    mean, sd = gp.predict(normalized, return_std=True)
    probabilities = ndtr(mean / np.maximum(sd, 1e-12))
    passing = np.flatnonzero(probabilities > INITIAL_SCREENING_LEVEL)
    if len(passing) < NWALKERS:
        raise RuntimeError(f"Only {len(passing)} initialization candidates pass the GP.")
    selected = rng.choice(passing, size=NWALKERS, replace=False)
    initial_walkers = candidates[selected]

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        initial_walkers=initial_walkers,
        param_names=param_names,
        centre=centre,
        centre_training_index=centre_index,
        centre_discrepancy=centre_discrepancy,
        centre_selection=np.array("minimum measured reduced-posterior discrepancy among GP training tuples"),
        half_width_log10=HALF_WIDTH_LOG10,
        initial_screening_level=INITIAL_SCREENING_LEVEL,
        initial_gp_probabilities=probabilities[selected],
        candidates_generated=N_CANDIDATES,
        candidates_retained=len(passing),
        nwalkers=NWALKERS,
        random_seed=RANDOM_SEED,
        lower_bounds=lower,
        upper_bounds=upper,
        gp_file=np.array(str(GP_MPM_FILE)),
        gp_SHA256=np.array(hashlib.sha256(GP_MPM_FILE.read_bytes()).hexdigest()),
        concentration_unit=np.array("uM"),
        time_unit=np.array("min"),
    )
    print("Centre selected from GP training row:", centre_index)
    print("Centre trajectory discrepancy (uM):", centre_discrepancy)
    print("Centre parameters:", dict(zip(param_names, centre)))
    print("GP-passing initialization candidates:", len(passing), "/", N_CANDIDATES)
    print("Saved shared starting ensemble for the matched 40k runs:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
