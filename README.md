# hierarch-inf [![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.30683-b31b1b.svg)](https://arxiv.org/abs/2610.30683)

Code accompanying [**Behavioral Mapping of Hierarchical Dynamical Models
Accelerates Bayesian Parameter Inference**](https://arxiv.org/abs/2610.30683), by Jingsong Bai and Ayush Pandey
(2026).

The paper is based on the following key idea: high-level models in a hierarchy of models can be used to inform the posterior learning of a detailed model. We do this by learning a behavioral
compatibility region for the detailed model. A fixed Gaussian process filters
detailed model MCMC proposals before the expensive likelihood evaluation.

## Examples

| Notebook | Contents |
| --- | --- |
| [Linear system](examples/Linear%20system.ipynb) | Demonstration of analytical behavioral support, and Illustration of GP-MPM learning. |
| [Hierarchical inference of a phenomenological model](examples/Hierarchical%20inference%20of%20a%20phenomenological%20model.ipynb) | High-level to nine-state inference, GP (high-level to phenomenological), and the associated paper figures. |
| [Hierarchical inference of a mechanistic model](examples/Hierarchical%20inference%20of%20a%20mechanistic%20model.ipynb) | High-level to 38-state inference, GP_hm (high-level to mechanistic), and the associated paper figures. |
| [Model construction](examples/Model%20construction.ipynb) | BioCRNpyler construction and AutoReduce reduction of all three gene expression model levels, with SBML export. |

## Installation

Use Python 3.12. From the repository root:

```bash
git lfs install
git lfs pull
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install -r requirements-gene-expression.txt
python -m ipykernel install --user --name hierarch-inf --display-name "Python (hierarch-inf)"
jupyter lab
```

For just the shared methods and linear example, install `requirements.txt`.
Conda users can instead run `conda env create -f environment.yml` followed by
`conda activate hierarch-inf`. Select the environment's kernel in Jupyter or VS Code.

`requirements-gene-expression.txt` installs Bioscrape from the
[`paralle-inference-fixes` branch](https://github.com/biocircuits/bioscrape/tree/paralle-inference-fixes). (apologies for the typo on the branch name!)

The gene expression workflows calls `bioscrape.inference.py_inference` with the
parallel sampling and joint-prior settings used for the paper. These are not live on Bioscrape main yet (hopefully soon!)

## Experimental measurements

Obtain the measurements from
[zjuradoq/PURE_CRN_models](https://github.com/zjuradoq/PURE_CRN_models).
The file used here is
[`Data_files/2024.01.26_PURExpress_DNAconcs/degfp61_data_tidy.csv`](https://github.com/zjuradoq/PURE_CRN_models/blob/main/Data_files/2024.01.26_PURExpress_DNAconcs/degfp61_data_tidy.csv).

Download the file to `examples/data/degfp61_data_tidy.csv` before running the
notebooks. The notebooks and inference modules use this location by default;
edit `DATA_FILE` to use another copy. The downloaded CSV is ignored by Git.


## Methods and files

The importable package is `hierarch_inf/`:

- `inference.py`: `py_inference`, prior evaluation, and MCMC results.
- `behavioral.py`: discrepancies and behavioral GP construction from a reduced posterior.
- `gp.py`: parameter coordinates, GP-MPM acquisition and learning, and fixed-GP filtering.
- `models.py`: the existing mechanistic CRN construction and SBML export helpers.
- `gene_expression/`: the existing high-level, phenomenological, and mechanistic
  inference entry points, including guided initialization.

```python
from hierarch_inf import py_inference, build_behavioral_gp_from_reduced_posterior
```

Model simulations and likelihoods are supplied as Python callables. The linear
notebook demonstrates the shared interface. The gene expression methods use the Bioscrape implementations used for the final runs.

`examples/data/` holds the locally downloaded measurements and generated GP warm-start observations. New sampler
CSV/TXT exports and GP construction intermediates also go there.
`examples/results/` contains only these 16 generated files used by the figure
workflows; historical runs and development artifacts live in the research folder.

| File in `examples/results/` | Figure input |
| --- | --- |
| `reduced_model_inference_10k_1020.npz` | Shared high-level posterior, fit, and setup time. |
| `p_base_1020.npz` | Phenomenological baseline. |
| `p_gp_z0.8_1020.npz` | Phenomenological GP-guided run. |
| `exp3_1_init_baseline_inference_40k_gaussian_batched_1020_2.npz` | Mechanistic baseline. |
| `exp3_1_init_gp_inference_zeta_0.8_40k_eps0p5_batched_1020_2.npz` | Mechanistic GP-guided run. |
| `exp3_1_init_baseline_inference_1020.npz` | Saved ensemble used for the mechanistic proposal-filter illustration. |
| `gp_hierarchid_inputs_1020.npz` | Shared reduced-posterior trajectories and figure source record. |
| `gp_hp_1020.npz` | Trained GP_hp. |
| `gp_hp_rand_1020.npz` | Phenomenological random-sampling comparison. |
| `gp_hp_history_1020.npz` | Phenomenological learning curves. |
| `gp_hp_test_1020.npz` | Phenomenological independent test observations. |
| `gp_hp_timing_1020.json` | Phenomenological GP construction time. |
| `gp_mpm_exp3_result_eps0p5_1020.npz` | Trained GP_hm. |
| `gp_random_exp3_result_eps0p5_1020.npz` | Mechanistic random-sampling comparison. |
| `gp_exp3_focused_history_eps0p5_1020.npz` | Mechanistic learning curves. |
| `gp_exp3_focused_timing_eps0p5_1020.json` | Mechanistic GP construction time. |


## Existing method tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## AI Use Statement

The authors used several AI tools for brainstorming, assistance with the
computational implementations, data visualizations, and writing of the manuscript.
AI generated outputs were reviewed, vetted, and edited by the authors for accuracy,
and the authors take full responsibility for the content of the manuscript.

## Citation

```bibtex
@misc{bai2026behavioral,
  title  = {Behavioral Mapping of Hierarchical Dynamical Models Accelerates Bayesian Parameter Inference},
  author = {Bai, Jingsong and Pandey, Ayush},
  year   = {2026},
  eprint = {2610.30683},
  archivePrefix = {arXiv},
  primaryClass = {stat.ME},
  doi    = {10.48550/arXiv.2610.30683},
  url    = {https://arxiv.org/abs/2610.30683}
}
```

Citation metadata is in [CITATION.cff](CITATION.cff).
Released under the [BSD 3-Clause License](LICENSE). Copyright CTSD Lab.
