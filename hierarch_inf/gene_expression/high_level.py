"""Run inference of the two-state high-level model.

Sampling settings are module constants. Generated files are stored under
examples/ and measurements are loaded from examples/data/degfp61_data_tidy.csv.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from bioscrape.types import Model
from bioscrape.inference import py_inference


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
DATA_DIR = EXAMPLES_DIR / "data"

REDUCED_SBML = (EXAMPLES_DIR / "models/gfp_reduced_TX.xml")
DATA_FILE = DATA_DIR / "degfp61_data_tidy.csv"
RESULTS_DIR = EXAMPLES_DIR / "results"
RUN_TAG = "1020"
OUTPUT_FILE = RESULTS_DIR / f"reduced_model_inference_10k_{RUN_TAG}.npz"

NWALKERS = 64
NSTEPS = 10000
BURN = 5000
N_PROCESSES = 32
RANDOM_SEED = 2026

DNA_NM_TO_UM = 1e-3
GFP_RFU_PER_UM = 5542.4
TIME_MAX_H = 3.0
TRUNCATE_BEFORE_H = 0.25

FIXED_RNA_PARAMETERS = {"kb_deg": 10.0, "ku_deg": 1.0, "kdeg": 0.1}
fixed_parameters = {
    "K_2": (FIXED_RNA_PARAMETERS["ku_deg"] + FIXED_RNA_PARAMETERS["kdeg"])
           / FIXED_RNA_PARAMETERS["kb_deg"],
    "kdeg": FIXED_RNA_PARAMETERS["kdeg"],
}

params_to_estimate = ["ktx", "K_1", "ktl", "kdil", "K_0"]
prior = {
    "ktx": ["log-uniform", 0.1, 1000.0],
    "K_1": ["log-uniform", 0.001, 10.0],
    "ktl": ["log-uniform", 0.01, 10.0],
    "kdil": ["log-uniform", 0.0001, 1.0],
    "K_0": ["log-uniform", 0.001, 10.0],
}


def main():
    """Run inference of the two-state high-level model."""
    RESULTS_DIR.mkdir(exist_ok=True)
    model = Model(sbml_filename=str(REDUCED_SBML))
    model.set_params(fixed_parameters)
    model_parameters = model.get_parameter_dictionary()
    enzyme_totals = {name: model_parameters[name] for name in ("P_tot", "R_tot", "E_tot")}

    df = pd.read_csv(DATA_FILE)
    for column in ("na", "construct", "repeat"):
        df[column] = df[column].astype(str).str.strip()
    for column in ("Time", "value", "na_conc"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[(df["na"].str.lower() == "dna") & (df["construct"] == "MGapt-deGFP")].dropna(
        subset=["Time", "value", "na_conc", "repeat"]
    ).copy()
    df = df[df["Time"].between(TRUNCATE_BEFORE_H * 60, TIME_MAX_H * 60)].copy()
    time_offset_min = float(df["Time"].min())
    df["Time"] -= time_offset_min
    df["dna_uM"] = df["na_conc"] * DNA_NM_TO_UM
    dna_concs = np.sort(df["dna_uM"].unique())[::-1]
    replicates = sorted(df["repeat"].unique())

    exp_data = []
    initial_conditions = []
    parameter_conditions = []
    for dna in dna_concs:
        for rep in replicates:
            block = df[(df["dna_uM"] == dna) & (df["repeat"] == rep)].sort_values("Time")
            exp_data.append(pd.DataFrame({
                "time": block["Time"].to_numpy(dtype=float),
                "GFP": block["value"].to_numpy(dtype=float) / GFP_RFU_PER_UM,
            }))
            initial_conditions.append({"mRNA": 0.0, "GFP": 0.0})
            parameter_conditions.append({"DNA": float(dna)})

    timepoints = exp_data[0]["time"].to_numpy(dtype=float)
    experimental_mean_trajectories = np.array([
        np.mean([
            exp_data[c * len(replicates) + j]["GFP"].to_numpy(dtype=float)
            for j in range(len(replicates))
        ], axis=0)
        for c in range(len(dna_concs))
    ])

    np.random.seed(RANDOM_SEED)
    initial_walkers = np.column_stack([
        np.exp(np.random.uniform(np.log(prior[name][1]), np.log(prior[name][2]), NWALKERS))
        for name in params_to_estimate
    ])
    print("Experiment 2 reduced-model inference", flush=True)
    print("Walkers:", NWALKERS, "Steps:", NSTEPS, "Burn:", BURN, "Processes:", N_PROCESSES)
    print("Priors:", prior)
    print("Fixed RNA degradation parameters:", fixed_parameters)
    print("DNA inputs (uM):", dna_concs)
    print("Resource pools (uM):", enzyme_totals, flush=True)

    start = time.perf_counter()
    sampler, pid = py_inference(
        Model=model,
        exp_data=exp_data,
        measurements=["GFP"],
        time_column="time",
        initial_conditions=initial_conditions,
        parameter_conditions=parameter_conditions,
        params_to_estimate=params_to_estimate,
        prior=prior,
        sim_type="deterministic",
        nwalkers=NWALKERS,
        nsteps=NSTEPS,
        init_seed=initial_walkers,
        discard=BURN,
        parallel=True,
        n_processes=N_PROCESSES,
        plot_show=False,
        filename_csv=str(DATA_DIR / f"reduced_mcmc_10k_{RUN_TAG}.csv"),
        filename_txt=str(DATA_DIR / f"reduced_mcmc_10k_{RUN_TAG}.txt"),
    )
    runtime_seconds = time.perf_counter() - start
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    # Write the completed chain before computing autocorrelation diagnostics.
    np.savez_compressed(
        OUTPUT_FILE,
        chain=chain, log_prob=log_prob,
        runtime_seconds=runtime_seconds,
        acceptance_fraction=sampler.acceptance_fraction,
        param_names=np.array(params_to_estimate),
        parameter_units=np.array(["uM" if name.startswith("K_") else "min^-1"
                                  for name in params_to_estimate]),
        nwalkers=NWALKERS, nsteps=NSTEPS, burn=BURN,
        random_seed=RANDOM_SEED, n_processes=N_PROCESSES,
        initial_walkers=initial_walkers, initialization_mode=np.array("log-uniform"),
        prior_json=np.array(json.dumps(prior)),
        fixed_parameters_json=np.array(json.dumps(fixed_parameters)),
        fixed_detailed_parameters_json=np.array(json.dumps(FIXED_RNA_PARAMETERS)),
        concentration_unit=np.array("uM"), time_unit=np.array("min"),
        run_tag=np.array(RUN_TAG),
        DNA_NM_TO_UM=DNA_NM_TO_UM, GFP_RFU_PER_UM=GFP_RFU_PER_UM,
        dna_concs=dna_concs, dna_concs_nM=dna_concs / DNA_NM_TO_UM,
        timepoints=timepoints, time_offset_minutes=time_offset_min,
        experimental_mean_trajectories=experimental_mean_trajectories,
        enzyme_totals_json=np.array(json.dumps(enzyme_totals)),
        data_SHA256=np.array(hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()),
        model_SHA256=np.array(hashlib.sha256(REDUCED_SBML.read_bytes()).hexdigest()),
        script_SHA256=np.array(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    )
    print("Saved:", OUTPUT_FILE, flush=True)
    print("Runtime (minutes):", runtime_seconds / 60)
    print("Mean acceptance fraction:", sampler.acceptance_fraction.mean())
    tau = sampler.get_autocorr_time(discard=BURN, tol=0)
    ess = NWALKERS * (NSTEPS - BURN) / tau
    print("Autocorrelation times (steps):", tau)
    print("Effective sample sizes:", ess, flush=True)


if __name__ == "__main__":
    main()
