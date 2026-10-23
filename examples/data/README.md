# Example data

This folder holds locally downloaded measurements, generated inputs, and intermediates for the inference examples.
The two supplied archives are:

- `gp_mpm_exp3_result_1018.npz`
- `gp_exp3_focused_evaluations_1018.npz`

New sampler CSV/TXT exports, mechanistic initial walkers, and GP search
intermediates are written here. The figure workflows read their final inputs from
`../results/`.

Obtain experimental measurements from
[zjuradoq/PURE_CRN_models](https://github.com/zjuradoq/PURE_CRN_models) and save
`degfp61_data_tidy.csv` in this folder. The notebooks and inference modules use
`examples/data/degfp61_data_tidy.csv` by default. The downloaded CSV is ignored by
Git and is not distributed with the repository.
