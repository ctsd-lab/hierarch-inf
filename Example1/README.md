# Example 1

## Installation

A working Conda installation (Miniconda or Anaconda) and internet access are
required. Keep `example1.ipynb`, `implementation_methods.py`, and
`environment.yml` in the same project directory.

### 1. Open the project directory

Replace the path below with your local project directory:

```bash
cd "/path/to/simphsi"
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate simphsi
```

If the environment already exists, update it instead:

```bash
conda env update -n simphsi -f environment.yml
conda activate simphsi
```

The environment installs Python 3.12, NumPy, SciPy, Matplotlib,
python-control, scikit-learn, emcee, and the Jupyter tools.
`solve_ivp` is included in SciPy and needs no separate installation.

### 3. Register the notebook kernel

```bash
python -m ipykernel install --user --name simphsi --display-name "Python (simphsi)"
```

### 4. Open and run the notebook

```bash
jupyter lab example1.ipynb
```

Select **Python (simphsi)** and run the cells from top to bottom. For a fresh
run, use **Restart Kernel and Run All Cells**. No external dataset is needed.

Alternatively, open `example1.ipynb` in VS Code with the Python and Jupyter
extensions installed, and select the `simphsi` environment as its kernel.

## Experiment parameters

| Parameter | Notebook variable | Value | Meaning |
| --- | --- | --- | --- |
| $\epsilon$ | `epsilon` | 0.20 | Compatibility tolerance between detailed and reduced trajectories. |
| $\eta$ | `ita` | 0.15 | Data-discrepancy tolerance used in the reduced-posterior fit-mass diagnostic. |
| $\gamma$ | `gamma` | 0.80 | Behavioral-support threshold defining the retained set $b_\epsilon(\theta)\ge\gamma$. |
| $\zeta$ | `zeta` | 0.01 | GP screening cutoff: within the GP domain, reject when the predicted probability of $b_\epsilon(\theta)-\gamma\ge0$ is at most $\zeta$. |

The notebook sets these values and their corresponding function arguments as:

```python
epsilon = 0.20
ita = 0.15
gamma = 0.80
zeta = 0.01
```

After changing these parameters, rerun their assignments, the GP construction
cell, guided MCMC, and the visualization cell. Changing a variable alone does
not update an already fitted GP.
