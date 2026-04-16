# get-shredded

Minimal starter repository for experimenting with **SHRED (Shallow Recurrent Decoder)** on the cylinder-vortex dataset used in the course material.

## What this includes

- POD/SVD preprocessing for high-dimensional flow snapshots
- Sequence windowing for autoregressive forecasting
- A shallow recurrent decoder (`GRU` or `LSTM`, one layer)
- A baseline training script for `CYLINDER_ALL.mat`

## Project layout

```
get-shredded/
  configs/
    cylinder_baseline.yaml
  scripts/
    run_cylinder_baseline.py
  src/get_shredded/
    data.py
    model.py
    train.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_cylinder_baseline.py --data-mat ../../DATA/FLUIDS/CYLINDER_ALL.mat
```

When this repo is used as a submodule under `Project/get-shredded` in your exercises repo, the default `--data-mat` path above is valid.
