# PetShop measured-data inspection

This is a schema/data assessment and descriptive application plan. It does not fit a queueing model or measure inference speedup.

Source: https://github.com/amazon-science/petshop-root-cause-analysis/tree/2e96f937c4c044b8b4aad03217592cd52e66db5d

Paper: Hardt et al. (2024), The PetShop Dataset — Finding Causes of Performance Issues across Microservices, PMLR236:957–978. https://proceedings.mlr.press/v236/hardt24a.html

Data license: CC-BY-4.0. The manifest pins 144 dataset files totaling 18,609,770 bytes. Data are downloaded separately and must not be committed. Each file is verified against its Git blob SHA-1 and byte count. Source files already present with different contents cause an error; they are not silently overwritten.

Requires Python 3, pandas and numpy. From this directory:

```
python download_petshop.py --data-dir /tmp/petshop-data
python inspect_petshop.py --data-dir /tmp/petshop-data --output-dir /tmp/petshop-results
```

To verify cached inputs without downloading:

```
python download_petshop.py --data-dir /tmp/petshop-data --verify-only
```

`summary.json` is a small measured-data summary reproduced by the above commands. `application_plan.md` discusses appropriate inference experiments and assumptions. The generated extracted CSVs retain raw latency units. The paper's stated millisecond values disagree with the conventional conversion of the raw numerical values, so no latency conversion is performed.

Temporal1/2 normal files have different column ordering but identical values and timestamps after sorting columns; deduplicate them before train/test splitting. The normal temporal curve is nearly flat over much of the measured workload. All 68 fault cases contain only 5 windows, so they support short disturbance checks rather than independent curve estimation.
