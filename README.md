# get-shredded

Minimal starter repository for experimenting with **SHRED (Shallow Recurrent Decoder)** on the cylinder-vortex dataset used in the course material.

## What this includes

- POD/SVD preprocessing for high-dimensional flow snapshots
- Sequence windowing for autoregressive forecasting
- A shallow recurrent decoder (`GRU` or `LSTM`, one layer)
- A baseline training script for `CYLINDER_ALL.mat`
- Hydra-based experiment configs for the baseline and comparison runs

## Project layout

```
get-shredded/
  data/
  configs/
    cylinder_baseline.yaml
    compare_shred_vs_dmd.yaml
  scripts/
    run_cylinder_baseline.py
    compare_shred_vs_dmd.py
  src/get_shredded/
    data.py
    model.py
    train.py
```

## Quick start

```bash
uv sync
uv run python scripts/run_cylinder_baseline.py
uv run python scripts/compare_shred_vs_dmd.py
```

The checked-in dataset is expected at `data/CYLINDER_ALL.mat`.

Hydra overrides work as usual, for example `python scripts/run_cylinder_baseline.py model.rank=40 train.epochs=500`.
