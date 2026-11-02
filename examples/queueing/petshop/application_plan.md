# PetShop application: mean response-time preservation under changing operating conditions

## Verified data

Source: https://github.com/amazon-science/petshop-root-cause-analysis (main tree retrieved 2026-11-02). Paper: https://proceedings.mlr.press/v236/hardt24a/hardt24a.pdf . Data license CC-BY-4.0; code Apache-2.0. Published 2024, actual checked timestamps in 2023.

CSV uses three header rows `(microservice, metric, statistic)` and a Unix-seconds index. PetSite exposes `(latency, Average)`, `(latency,p50/p90/p95/p99)`, `(requests,Sum)`, and `(availability,Average)`. Cadence is exactly 300 seconds in inspected normal runs. Availability values use percent. Request count divided by 300 gives the average observed request rate, not the within-window arrival process.

Normal windows: low 589; high 589; temporal 1652. Temporal_traffic1 and temporal_traffic2 contain the same baseline values/timestamps after sorting columns: deduplicate before splitting. Graphs contain 42/44/41 nodes respectively, reflecting observed service-map variation; paper describes 41 components. Call adjacency does not specify queue disciplines, routing probabilities, server counts, synchronous/asynchronous composition or service distributions.

All 68 issue files contain 5 time windows. Most have 2 timestamps before injection and 3 after. These short cases are useful for paired stress checks, not independent estimation of a full response curve. Fault metadata identifies injection timestamp, target metric, root service and command. There are 36 latency-targeted and 32 availability-targeted cases. Six database-overload cases are all labeled test and involve petInfo; retain them as application holdouts.

Unit issue: PetSite raw normal latency means are 0.0949286 (low), 0.0981784 (high), 0.1004184 (temporal). Paper Table 5 labels 9.49, 9.82, 10.04 as milliseconds. Its factor 100 conversion disagrees with conventional seconds→milliseconds. Source CSVs carry no unit. Do not silently convert. Dimensionless relative errors avoid this scale ambiguity pending confirmation; absolute units need resolution.

## Recommended application

Use PetShop as an application of behavior-preserving aggregate inference under normal traffic and service degradation. Start with the empirical mean response-time curve for the normal temporal run. Infer a calibrated aggregate performance model with explicit modeling assumptions; compare full parameter posterior inference against contract-restricted inference on the same model.

Define a workload bin's reference mean by total-request weighting of its five-minute average latencies: sum(n_t*y_t)/sum(n_t). Keep this separate from the mean across windows. Use chronological held-out day blocks, not randomly interleaved windows; deduplicate both temporal baselines and reserve evaluation data from contract construction. Track availability so response-time changes caused by failing requests are visible.

A two-level candidate hierarchy is a detailed service/subsystem performance model and an effective aggregate PetSite model. Any queueing-network realization needs declared assumptions for call composition and service discipline. Available aggregates alone do not identify request-level event histories. Contract bounds must be derived for the declared model; an observed data fit is empirical validation, not a sound model-to-reality guarantee.

First experiment: normal-traffic behavior preservation and computational cost. Fit/load-calibrate on temporal day blocks, validate mean response at held-out blocks/load bins, and inspect transfer to low/high runs separately (different collection dates/configuration graphs). Report retained posterior mass/acceptance, uniform response-curve error over evaluated loads, violation rate, effective sample size and wall time including contract construction. No speedup is measured by this inspection.

Second experiment: six held-out database-overload cases. Compare pre/post mean-response observations against the normal-regime contract and quantify rejection/constraint violations when its environmental assumptions no longer hold. Injected database traffic is external to the PetSite request count, so this is a disturbance/regime-change test, not a larger input-load point on the normal PetSite curve. Avoid treating successful fault detection as a causal identification result.

Normal data show a nearly flat response curve over much of the observed load range: temporal bins with >= 20 windows span rates approximately 1.20–4.40 requests/s and raw means approximately 0.0994–0.1037. GAMMA is consequently the stronger primary workload-response experiment; PetShop adds operational disturbance stress testing.

## Reproducible artifacts

`inspect_petshop.py` reads downloaded source CSVs and writes `summary.json`, `issues.csv`, `descriptive_curves.csv` in the selected output directory, and per-scenario PetSite extraction CSVs. The code preserves latency raw units and converts only requests/window to requests/second. Outputs are descriptive measured-data summaries, not inferred parameters or posterior/speedup results. The source manifest pins the dataset revision and input file hashes.
