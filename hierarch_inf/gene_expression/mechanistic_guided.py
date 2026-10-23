"""Run the 38-state inference using the fixed GP as a batched joint prior.

Sampling settings are module constants. Generated files are stored under
examples/ and measurements are loaded from examples/data/degfp61_data_tidy.csv.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import time
import json
import hashlib

import numpy as np
import pandas as pd

from bioscrape.types import Model
from bioscrape.inference import py_inference
from bioscrape.pid_interfaces import PIDInterface
import bioscrape


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = EXAMPLES_DIR / "data"

CLEAN_SBML = (EXAMPLES_DIR / "models/exp3_gfp_bioscrape.xml")
DATA_FILE = DATA_DIR / "degfp61_data_tidy.csv"
RESULTS_DIR = EXAMPLES_DIR / "results"
RUN_TAG = "1020_2"
INITIAL_WALKERS_FILE = DATA_DIR / "exp3_1_initial_walkers_eps0p5_1020.npz"
GP_MPM_FILE = RESULTS_DIR / "gp_mpm_exp3_result_eps0p5_1020.npz"
NWALKERS = 256
NSTEPS = 40000
BURN = 5000
RANDOM_SEED = 2026
N_PROCESSES = 32
PARALLEL_METHOD = "batched"
GP_SCREENING_LEVEL = 0.8
OUTPUT_FILE = RESULTS_DIR / f"exp3_1_init_gp_inference_zeta_{GP_SCREENING_LEVEL:g}_40k_eps0p5_batched_{RUN_TAG}.npz"
DNA_NM_TO_UM = 1e-3

GFP_RFU_PER_UM = 5542.4
TIME_MAX_H = 3.0
TRUNCATE_BEFORE_H = 0.25

params_to_estimate = ['kb_tx', 'ku_tx', 'kb_tl', 'ku_tl', 'ktx', 'ktl', 'kdil']

prior = {'kb_tx': ['log-uniform', 0.1, 100000.0],
 'ku_tx': ['log-uniform', 0.001, 1000.0],
 'kb_tl': ['log-uniform', 0.01, 10000.0],
 'ku_tl': ['log-uniform', 0.001, 1000.0],
 'ktx': ['log-uniform', 0.01, 10000.0],
 'ktl': ['log-uniform', 0.001, 100.0],
 'kdil': ['log-uniform', 0.00001, 10.0]}

FIXED_PARAMETERS = {'kb_deg': 10.0,
 'ku_deg': 1.0,
 'kdeg': 0.1,
 'kiso_tx': 10.0,
 'kiso_tl': 10.0,
 'kmat': 0.12121212121212122}

order_magnitude_params = {'kb_tx': 10.0,
 'ku_tx': 1.0,
 'kb_tl': 10.0,
 'ku_tl': 1.0,
 'ktx': 1.0,
 'ktl': 1.0,
 'kdil': 0.1,
 'kiso_tx': 10.0,
 'kiso_tl': 10.0,
 'kmat': 0.12121212121212122,
 'kb_deg': 10.0,
 'ku_deg': 1.0,
 'kdeg': 0.1}

def main():
    """Run the 38-state inference using the fixed GP as a batched joint prior."""
    if not hasattr(PIDInterface, "get_log_prior_batch"):
        raise RuntimeError("Install the updated Bioscrape with batch joint-prior evaluation.")
    # The GP and its callbacks remain in the parent process.
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    from scipy.special import ndtr

    gp_data = np.load(GP_MPM_FILE)
    assert str(gp_data["concentration_unit"]) == "uM"
    assert float(gp_data["DNA_NM_TO_UM"]) == DNA_NM_TO_UM

    X_gp = gp_data["X_gp"]
    v_gp = gp_data["v_gp"]
    validity_scale = float(gp_data["validity_scale"])
    lower_bounds_detailed = gp_data["lower_bounds_detailed"]
    upper_bounds_detailed = gp_data["upper_bounds_detailed"]
    gp_param_names = gp_data["detailed_param_names"].tolist()
    gp_kernel_constant = float(gp_data["gp_kernel_constant"])
    gp_kernel_length_scale = gp_data["gp_kernel_length_scale"]
    GP_NOISE = float(gp_data["gp_noise"])
    gp_epsilon = float(gp_data["epsilon"])
    gp_gamma = float(gp_data["gamma"])
    gp_training_seconds = float(gp_data["gp_training_seconds"])
    gp_region_centres = gp_data["region_centres"]
    gp_region_lower = gp_data["region_lower"]
    gp_region_upper = gp_data["region_upper"]
    gp_SHA256 = hashlib.sha256(GP_MPM_FILE.read_bytes()).hexdigest()
    assert json.loads(gp_data["fixed_detailed_parameters_json"].item()) == FIXED_PARAMETERS
    assert str(gp_data["detailed_model_SHA256"]) == hashlib.sha256(CLEAN_SBML.read_bytes()).hexdigest()
    reduced_inference_seconds = float(gp_data["reduced_inference_seconds"])
    reduced_gp_preparation_seconds = float(gp_data["reduced_gp_preparation_seconds"])
    gp_training_seconds_scope = str(gp_data["gp_training_seconds_scope"])
    gp_data.close()

    assert gp_param_names == params_to_estimate

    prior_lower_bounds = np.array([prior[name][1] for name in params_to_estimate], dtype=float)
    prior_upper_bounds = np.array([prior[name][2] for name in params_to_estimate], dtype=float)

    assert np.allclose(lower_bounds_detailed, prior_lower_bounds)
    assert np.allclose(upper_bounds_detailed, prior_upper_bounds)

    log_lower_bounds = np.log(lower_bounds_detailed)
    log_upper_bounds = np.log(upper_bounds_detailed)

    kernel_trained = (
        ConstantKernel(
            gp_kernel_constant,
            constant_value_bounds="fixed",
        )
        * RBF(
            gp_kernel_length_scale,
            length_scale_bounds="fixed",
        )
    )

    gp_validity = GaussianProcessRegressor(
        kernel=kernel_trained,
        alpha=GP_NOISE,
        optimizer=None,
        normalize_y=False,
    )

    gp_validity.fit(X_gp, v_gp / validity_scale)


    def custom_joint_prior(params_dict):
        """Apply the fixed GP filter to one physical parameter mapping."""
        beta = np.array(
            [params_dict[name] for name in params_to_estimate],
            dtype=float,
        )

        if np.any(beta <= 0):
            return np.inf

        x = (np.log(beta) - log_lower_bounds) / (log_upper_bounds - log_lower_bounds)
        gp_mean, gp_std = gp_validity.predict(x.reshape(1, -1), return_std=True)
        gp_mean = float(gp_mean[0])
        gp_std = max(float(gp_std[0]), 1e-12)
        gp_retained_probability = float(ndtr(gp_mean / gp_std))

        if gp_retained_probability <= GP_SCREENING_LEVEL:
            return np.inf

        return 0.0

    gp_screening_stats = {"batches": 0, "proposals": 0, "rejected": 0, "seconds": 0.0}

    def custom_joint_prior_batch(params_dict):
        """Apply the fixed GP filter to a batch of parameter tuples."""
        screening_start = time.perf_counter()
        beta = np.column_stack([
            params_dict[name] for name in params_to_estimate
        ])
        x = (np.log(beta) - log_lower_bounds) / (log_upper_bounds - log_lower_bounds)
        gp_mean, gp_std = gp_validity.predict(x, return_std=True)
        gp_retained_probability = ndtr(gp_mean / np.maximum(gp_std, 1e-12))
        retained = gp_retained_probability > GP_SCREENING_LEVEL
        gp_screening_stats["batches"] += 1
        gp_screening_stats["proposals"] += len(retained)
        gp_screening_stats["rejected"] += int(np.count_nonzero(~retained))
        gp_screening_stats["seconds"] += time.perf_counter() - screening_start
        return np.where(retained, 0.0, -np.inf)

    RESULTS_DIR.mkdir(exist_ok=True)
    M = Model(sbml_filename=str(CLEAN_SBML))
    M.set_params(order_magnitude_params)
    M.set_params(FIXED_PARAMETERS)
    species_defaults = M.get_species_dictionary()
    enzyme_totals = {name: species_defaults[name] for name in ("RNAP", "Ribo", "RNase")}

    with np.load(INITIAL_WALKERS_FILE, allow_pickle=False) as record:
        initialization = {key: record[key] for key in record.files}
    initial_walkers = initialization["initial_walkers"]
    assert initial_walkers.shape == (NWALKERS, len(params_to_estimate))
    assert initialization["param_names"].tolist() == params_to_estimate
    assert np.all(initial_walkers >= np.array([prior[p][1] for p in params_to_estimate]))
    assert np.all(initial_walkers <= np.array([prior[p][2] for p in params_to_estimate]))
    initialization_SHA256 = hashlib.sha256(INITIAL_WALKERS_FILE.read_bytes()).hexdigest()
    print("GP-based initialization:", INITIAL_WALKERS_FILE)
    print("Centre:", dict(zip(params_to_estimate, initialization["centre"])))
    print("Initialization half-width (log10):", float(initialization["half_width_log10"]))
    assert initialization["gp_SHA256"].item() == gp_SHA256

    df = pd.read_csv(DATA_FILE)

    for column in ["na", "construct", "repeat"]:
        df[column] = df[column].astype(str).str.strip()

    for column in ["Time", "value", "na_conc"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[
        (df["na"].str.lower() == "dna")
        & (df["construct"] == "MGapt-deGFP")
    ].dropna(
        subset=[
            "Time",
            "value",
            "na_conc",
            "repeat",
        ]
    ).copy()

    df["time_h"] = df["Time"] / 60.0
    df = df[
        (df["time_h"] >= TRUNCATE_BEFORE_H)
        & (df["time_h"] <= TIME_MAX_H)
    ].copy()

    time_offset_min = df["Time"].min()
    df["Time"] = df["Time"] - time_offset_min
    df["time_h"] = df["Time"] / 60.0

    df["dna_uM"] = df["na_conc"] * DNA_NM_TO_UM
    dna_concs = np.sort(df["dna_uM"].unique())[::-1]
    replicates = sorted(df["repeat"].unique())

    exp_data = []
    trajectory_initial_conditions = []

    for dna in dna_concs:
        for rep in replicates:
            block = df[
                (df["dna_uM"] == dna)
                & (df["repeat"] == rep)
            ].sort_values("Time")

            exp_data.append(
                pd.DataFrame(
                    {
                        "time": block["Time"].to_numpy(dtype=float),
                        "GFP": block["value"].to_numpy(dtype=float) / GFP_RFU_PER_UM,
                    }
                )
            )

            trajectory_initial_conditions.append({"DNA": float(dna)})


    np.random.seed(RANDOM_SEED)
    
    print("Experiment 3.1 GP-guided inference: GP-centre initialization")
    print("Walkers:", NWALKERS)
    print("Steps:", NSTEPS)
    print("Burn:", BURN)
    print("Processes:", N_PROCESSES)
    print("Parallel method:", PARALLEL_METHOD)
    print("Joint prior evaluation: batch")
    print("Trajectories:", len(exp_data))
    print("Parameters:", params_to_estimate)
    print("Loaded GP kernel:", gp_validity.kernel_)
    print("GP checkpoint:", GP_MPM_FILE)
    print("GP epsilon (uM):", gp_epsilon, "; gamma:", gp_gamma)
    print("GP screening level (zeta):", GP_SCREENING_LEVEL)
    print("GP training evaluations:", len(X_gp))
    print("DNA inputs (uM):", dna_concs)
    print("Enzyme pools inherited from SBML (uM):", enzyme_totals)


    start_time = time.perf_counter()

    sampler, pid = py_inference(
        Model=M,
        exp_data=exp_data,
        measurements=["GFP"],
        time_column=["time"],
        initial_conditions=trajectory_initial_conditions,
        params_to_estimate=params_to_estimate,
        prior=prior,
        custom_joint_prior=custom_joint_prior,
        custom_joint_prior_batch=custom_joint_prior_batch,
        sim_type="deterministic",
        nwalkers=NWALKERS,
        nsteps=NSTEPS,
        init_seed=initial_walkers,
        discard=BURN,
        parallel=True,
        n_processes=N_PROCESSES,
        parallel_method=PARALLEL_METHOD,
        plot_show=False,
        filename_csv=str(DATA_DIR / f"exp3_1_init_gp_mcmc_zeta_{GP_SCREENING_LEVEL:g}_40k_eps0p5_batched_{RUN_TAG}.csv"),
        filename_txt=str(DATA_DIR / f"exp3_1_init_gp_mcmc_zeta_{GP_SCREENING_LEVEL:g}_40k_eps0p5_batched_{RUN_TAG}.txt"),
    )

    runtime_seconds = time.perf_counter() - start_time
    n_proposals_evaluated = len(pid.cost_progress)
    n_parameter_prior_rejected = n_proposals_evaluated - gp_screening_stats["proposals"]
    n_likelihood_evaluations = gp_screening_stats["proposals"] - gp_screening_stats["rejected"]
    print("GP screening:", gp_screening_stats)
    print("Ordinary-prior rejections:", n_parameter_prior_rejected)
    print("Detailed likelihood evaluations:", n_likelihood_evaluations)

    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    try:
        tau = sampler.get_autocorr_time(discard=BURN, tol=0)
        n_postburn_samples = NWALKERS * (NSTEPS - BURN)
        ess = n_postburn_samples / tau
    except Exception as error:
        print("Autocorrelation diagnostic unavailable:", error)
        tau = np.full(len(params_to_estimate), np.nan)
        ess = np.full(len(params_to_estimate), np.nan)
    acceptance_fraction = sampler.acceptance_fraction

    print("Runtime:", runtime_seconds / 60.0, "minutes")
    print("Chain shape:", chain.shape)
    print("Autocorrelation time:", tau)
    print("Effective sample size:", ess)
    print("Mean acceptance fraction:", np.mean(acceptance_fraction))

    np.savez_compressed(
        OUTPUT_FILE,
        concentration_unit=np.array("uM"), DNA_NM_TO_UM=DNA_NM_TO_UM,
        dna_concs=dna_concs, dna_concs_nM=dna_concs / DNA_NM_TO_UM,
        timepoints=exp_data[0]["time"].to_numpy(dtype=float),
        time_offset_minutes=float(time_offset_min),
        n_processes=N_PROCESSES, prior_json=np.array(json.dumps(prior)),
        parallel_method=np.array(PARALLEL_METHOD),
        joint_prior_evaluation=np.array("batch"),
        n_proposals_evaluated=np.array(n_proposals_evaluated),
        n_parameter_prior_rejected=np.array(n_parameter_prior_rejected),
        n_likelihood_evaluations=np.array(n_likelihood_evaluations),
        gp_screening_batches=np.array(gp_screening_stats["batches"]),
        gp_screening_proposals=np.array(gp_screening_stats["proposals"]),
        gp_screening_rejected=np.array(gp_screening_stats["rejected"]),
        gp_screening_seconds=np.array(gp_screening_stats["seconds"]),
        bioscrape_version=np.array(bioscrape.__version__),
        fixed_parameters_json=np.array(json.dumps(FIXED_PARAMETERS)),
        run_tag=np.array(RUN_TAG),
        enzyme_totals_json=np.array(json.dumps(enzyme_totals)),
        initialization_mode=np.array("single GP-training centre with uniform log10 perturbations"),
        initial_walkers=initial_walkers,
        initialization_file=np.array(str(INITIAL_WALKERS_FILE)),
        initialization_SHA256=np.array(initialization_SHA256),
        initialization_center=initialization["centre"],
        initialization_center_training_index=initialization["centre_training_index"],
        initialization_center_discrepancy=initialization["centre_discrepancy"],
        initialization_center_selection=initialization["centre_selection"],
        initialization_half_width_log10=initialization["half_width_log10"],
        initialization_screening_level=initialization["initial_screening_level"],
        initialization_gp_probabilities=initialization["initial_gp_probabilities"],
        initialization_gp_SHA256=initialization["gp_SHA256"],
        parameter_units=np.array(["uM^-1 min^-1" if name.startswith("kb_") else "min^-1"
                                  for name in params_to_estimate]),
        time_unit=np.array("min"),
        chain=chain,
        log_prob=log_prob,
        runtime_seconds=np.array(runtime_seconds),
        tau=tau,
        ess=ess,
        acceptance_fraction=acceptance_fraction,
        param_names=np.array(params_to_estimate, dtype=str),
        nwalkers=np.array(NWALKERS),
        nsteps=np.array(NSTEPS),
        burn=np.array(BURN),
        random_seed=np.array(RANDOM_SEED),
        gp_kernel_constant=np.array(gp_kernel_constant),
        gp_kernel_length_scale=np.asarray(gp_kernel_length_scale),
        gp_noise=np.array(GP_NOISE),
        gp_screening_level=np.array(GP_SCREENING_LEVEL),
        gp_epsilon=np.array(gp_epsilon),
        gp_gamma=np.array(gp_gamma),
        gp_validity_scale=np.array(validity_scale),
        gp_training_X=X_gp,
        gp_training_targets=v_gp,
        gp_lower_bounds=lower_bounds_detailed,
        gp_upper_bounds=upper_bounds_detailed,
        gp_region_centres=gp_region_centres,
        gp_region_lower=gp_region_lower,
        gp_region_upper=gp_region_upper,
        gp_training_seconds=np.array(gp_training_seconds),
        gp_training_seconds_scope=np.array(gp_training_seconds_scope),
        reduced_inference_seconds=np.array(reduced_inference_seconds),
        reduced_gp_preparation_seconds=np.array(reduced_gp_preparation_seconds),
        gp_file=np.array(str(GP_MPM_FILE)),
        gp_reused_from_exp2=np.array(False),
        gp_SHA256=np.array(gp_SHA256),
        data_SHA256=np.array(hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()),
        model_SHA256=np.array(hashlib.sha256(CLEAN_SBML.read_bytes()).hexdigest()),
        tx_max_occupancy=5, tl_max_occupancy=10,
        n_states=len(M.get_species_dictionary()),
        script_SHA256=np.array(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    )

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
