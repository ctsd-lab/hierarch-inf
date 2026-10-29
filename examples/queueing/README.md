# Sutton–Jordan queueing application

Initial setup for applying behavioral inference to the applications in
[Sutton and Jordan (2011), arXiv:1001.3355v3](https://arxiv.org/abs/1001.3355v3).
The event primitives are implemented and tested; experiment reproduction and
hierarchical inference remain to be developed.

## Paper application inventory

[`paper_experiments.json`](paper_experiments.json) records reported settings;
its unresolved entries remain `null`. It is an inventory, not a runnable config.

| Application | Baseline | Pending |
| --- | --- | --- |
| §6 Cloudstone extrapolation | Complete-data maximum likelihood; queueing and regression comparisons | Original trace and workflow |
| §7 Cloudstone diagnosis | PS network; stochastic EM with slice-sampled missing events | Original trace and missing-event sampler |
| §8 Missing-queue selection | Synthetic bottleneck detection; Bayesian parameter resampling | Queue discipline and Figure 5(b)/5(c) generator discrepancy |

The §7 reference is the **complete-data reconstruction**, rather than measured
service-time ground truth. Under §3.1, hidden tasks have unknown arrivals and
paths: running the transforms on only logged jobs would give wrong occupancies.

## Implemented

- `fcfs_departures`, `fcfs_service_times`: single-server equation (1).
- `ps_service_times`: complete-event PS integral, equation (7).
- `ps_log_jacobian`: reciprocal-occupancy factor, equation (8). Service,
  arrival, and routing densities would still be needed for a full likelihood.

Import these from `hierarch_inf.queueing`. Supply all relevant events for one
queue as paired arrays in consistent units; omission cannot be detected from
arrays alone. FCFS uses arrival order (input order for ties). PS preserves job
order. Empty arrays and zero-duration events are supported; the PS integral
accepts ties, but its Jacobian rejects them as potentially nondifferentiable.
Nonfinite or missing times are rejected. Tests use analytical unit fixtures.

```bash
# Repository root, Python 3.12
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_queueing_events.py
python -m pytest
```

## Data and source checks

The original Cloudstone trace was not located in the inspected history of the
[authors' code](https://github.com/casutton/bayes-qnet/tree/0fff4b38394e47b02c0b4f0a4e433049b78fa09c)
and is not included here; its `rails3`/YAML fixtures are only tests. A supplied
trace needs provenance and schema checks before an importer can be built.

Two conventions need care: equation (9) uses an exponential **rate**, while the
authors' class uses **mean/scale**. Also, printed equation (11) disagrees with
(8): `QueuePS.likelihoodDelta` uses reciprocal occupancy, but `allEventLik`
returns before its correction. Our Jacobian follows (8), independently checked
against a finite-difference determinant.

## Design decisions pending

Discuss the contract's information, the additional low-cost model, the detailed
model's role, and the behavioral observable/discrepancy before integration.
Existing callable interfaces in `build_behavioral_gp_from_reduced_posterior`
and `run_gp_mpm` provide extension points; a contract has not been equated with
a rejection prior. The existing `py_inference` ensemble sampler is distinct
from Sutton–Jordan's latent-event slice sampler and stochastic EM. Network
simulation, experiment inference, and performance comparisons follow once the
inputs and model roles are established.
