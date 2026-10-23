"""Run the nine-state baseline inference with the recorded paper settings.

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
import bioscrape


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = EXAMPLES_DIR / "data"

CLEAN_SBML = (EXAMPLES_DIR / "models/gfp_expression_bioscrape.xml")
DATA_FILE = DATA_DIR / "degfp61_data_tidy.csv"
RESULTS_DIR = EXAMPLES_DIR / "results"
RUN_TAG = "1020"
INITIALIZATION_RELATIVE_SD = 0.05

NWALKERS = 256
NSTEPS = 40000
BURN = 5000
RANDOM_SEED = 2026
N_PROCESSES = 32
PARALLEL_METHOD = "batched"

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

FIXED_PARAMETERS = {"kb_deg": 10.0, "ku_deg": 1.0, "kdeg": 0.1}

order_magnitude_params = {
    "kb_tx": 10.0, "ku_tx": 1.0, "kb_tl": 10.0, "ku_tl": 1.0,
    "ktx": 1.0, "ktl": 1.0, "kdil": 0.1, **FIXED_PARAMETERS,
}


def make_initial_walkers():
    """Draw a Gaussian ball around the nominal rates, inside the prior bounds."""
    centre = np.array([order_magnitude_params[name] for name in params_to_estimate])
    standard_deviation = INITIALIZATION_RELATIVE_SD * centre
    lower = np.array([prior[name][1] for name in params_to_estimate])
    upper = np.array([prior[name][2] for name in params_to_estimate])
    rng = np.random.default_rng(RANDOM_SEED)
    walkers = rng.normal(centre, standard_deviation, size=(NWALKERS, len(centre)))
    outside = np.any((walkers < lower) | (walkers > upper), axis=1)
    while np.any(outside):
        walkers[outside] = rng.normal(
            centre, standard_deviation, size=(np.count_nonzero(outside), len(centre))
        )
        outside = np.any((walkers < lower) | (walkers > upper), axis=1)
    return walkers, centre, standard_deviation


def main():
    """Run the nine-state baseline inference with the recorded paper settings."""
    RESULTS_DIR.mkdir(exist_ok=True)
    M = Model(sbml_filename=str(CLEAN_SBML))
    M.set_params(order_magnitude_params)
    M.set_params(FIXED_PARAMETERS)
    species_defaults = M.get_species_dictionary()
    enzyme_totals = {name: species_defaults[name] for name in ("RNAP", "Ribo", "RNase")}

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

    # Source DNA conditions are nM; all model concentration states use µM.
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

            trajectory_initial_conditions.append(
                {
                    "DNA": float(dna),
                }
            )

    print("Phenomenological baseline: nine-state model")
    print("Walkers:", NWALKERS)
    print("Steps:", NSTEPS)
    print("Burn:", BURN)
    print("Processes:", N_PROCESSES)
    print("Parallel method:", PARALLEL_METHOD)
    print("Trajectories:", len(exp_data))
    print("Parameters:", params_to_estimate)
    print("DNA inputs (µM):", dna_concs)
    print("Priors:", prior)
    print("Fixed parameters:", FIXED_PARAMETERS)
    print("Enzyme pools inherited from SBML (µM):", enzyme_totals)

    timepoints = exp_data[0]["time"].to_numpy(dtype=float)
    np.random.seed(RANDOM_SEED)
    initial_walkers, initialization_center, initialization_sd = make_initial_walkers()
    print("Initialization: Gaussian ball around nominal model rates")
    print("Centre:", dict(zip(params_to_estimate, initialization_center)))
    print("Initialization relative SD:", INITIALIZATION_RELATIVE_SD)

    start_time = time.perf_counter()

    sampler, pid = py_inference(
        Model=M,
        exp_data=exp_data,
        measurements=["GFP"],
        time_column=["time"],
        initial_conditions=trajectory_initial_conditions,
        params_to_estimate=params_to_estimate,
        prior=prior,
        sim_type="deterministic",
        nwalkers=NWALKERS,
        nsteps=NSTEPS,
        init_seed=initial_walkers,
        discard=BURN,
        parallel=True,
        n_processes=N_PROCESSES,
        parallel_method=PARALLEL_METHOD,
        plot_show=False,
        filename_csv=str(DATA_DIR / f"p_base_mcmc_{RUN_TAG}.csv"),
        filename_txt=str(DATA_DIR / f"p_base_mcmc_{RUN_TAG}.txt"),
    )

    runtime_seconds = time.perf_counter() - start_time

    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    # A diagnostic failure must not prevent saving an expensive completed run.
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
        RESULTS_DIR / f"p_base_{RUN_TAG}.npz",
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
        concentration_unit=np.array("uM"), time_unit=np.array("min"),
        DNA_NM_TO_UM=DNA_NM_TO_UM, dna_concs=dna_concs,
        dna_concs_nM=dna_concs / DNA_NM_TO_UM, timepoints=timepoints,
        time_offset_minutes=float(time_offset_min),
        prior_json=np.array(json.dumps(prior)),
        fixed_parameters_json=np.array(json.dumps(FIXED_PARAMETERS)),
        run_tag=np.array(RUN_TAG),
        enzyme_totals_json=np.array(json.dumps(enzyme_totals)),
        parameter_units=np.array([
            "uM^-1 min^-1" if name.startswith("kb_") else "min^-1"
            for name in params_to_estimate
        ]),
        n_processes=N_PROCESSES,
        parallel_method=np.array(PARALLEL_METHOD),
        bioscrape_version=np.array(bioscrape.__version__),
        initialization_mode=np.array("Gaussian ball around nominal model rates, conditioned on prior bounds"),
        initial_walkers=initial_walkers,
        initialization_center=initialization_center,
        initialization_standard_deviation=initialization_sd,
        initialization_relative_sd=np.array(INITIALIZATION_RELATIVE_SD),
        initialization_coordinates=np.array("physical parameter values"),
        initialization_seed=np.array(RANDOM_SEED),
        data_SHA256=np.array(hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()),
        model_SHA256=np.array(hashlib.sha256(CLEAN_SBML.read_bytes()).hexdigest()),
        model_level=np.array("phenomenological"),
        n_states=len(M.get_species_dictionary()),
        script_SHA256=np.array(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    )

    print("Saved:", RESULTS_DIR / f"p_base_{RUN_TAG}.npz")


if __name__ == "__main__":
    main()
