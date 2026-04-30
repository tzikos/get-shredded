# get-shredded

Reimplementation of **SHRED (SHallow REcurrent Decoder)** from Williams, Zahn, Kutz (2024), *"Sensing with shallow recurrent decoder networks"*, applied to the cylinder-vortex dataset.

The repo trains SHRED end-to-end to reconstruct the full vorticity field from a handful of point sensor measurements over a `lags`-long time window, and compares it against two baselines used in the paper:

- **SDN** — static shallow decoder (same MLP, no recurrence; takes only the most recent sensor snapshot)
- **QR/POD** — linear gappy-POD reconstruction with QR-pivoted sensor placement: `x̂ = U_r (C U_r)^{-1} y`

## Project layout

```
get-shredded/
  data/                                # place CYLINDER_ALL.mat here
  configs/
    cylinder_baseline.yaml             # single-run config
    sweep_num_sensors.yaml             # error-vs-num_sensors sweep config
    default.yaml                       # Hydra plumbing
  scripts/
    run_cylinder_baseline.py           # train SHRED + SDN + QR/POD, save plots & GIF
    sweep_num_sensors.py               # sweep num_sensors, save aggregate plot
  src/get_shredded/
    model.py                           # SHRED, SDN, fit(), forecast()
    data.py                            # cylinder loader, qr_place, sensor windowing
    baseline.py                        # QR/POD baseline wrapper
    experiment.py                      # reusable single-run pipeline → RunResult
    plotting.py                        # panel/animation/curve/sweep plot helpers
  outputs/                             # all results land here (created on first run)
```

## Setup

Requires Python ≥ 3.10. Using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

Place `CYLINDER_ALL.mat` (the standard Brunton/Kutz cylinder-vortex dataset, contains a `VORTALL` array of shape `(m, T)`) at `data/CYLINDER_ALL.mat`.

## Single run: train + evaluate + plot

```bash
uv run python scripts/run_cylinder_baseline.py
```

This trains SHRED and SDN with early stopping, computes the QR/POD baseline, prints relative L2 test errors for all three, and writes plots under `outputs/cylinder/`:

- **`reconstructions/panel.png`** — 3 test snapshots × 4 columns (truth+sensor positions, SHRED, SDN, QR/POD). Sensor locations overlaid as lime dots on the truth column.
- **`reconstructions/comparison.gif`** — animated side-by-side reconstruction across the entire test set.
- **`curves/per_snapshot_error.png`** — relative L2 error per test snapshot, one line per method.
- **`curves/training_curves.png`** — log-scale validation error vs epoch for SHRED and SDN.

Hydra overrides work as usual:

```bash
# more sensors, random placement
uv run python scripts/run_cylinder_baseline.py model.num_sensors=10 model.placement=random

# longer history window, smaller LSTM
uv run python scripts/run_cylinder_baseline.py model.lags=20 model.hidden_size=32

# different output directory
uv run python scripts/run_cylinder_baseline.py outputs.root=outputs/experiment_2
```

Key knobs in [configs/cylinder_baseline.yaml](configs/cylinder_baseline.yaml):

| Setting | Default | Meaning |
|---|---|---|
| `model.num_sensors` | 3 | Number of point sensors |
| `model.lags` | 10 | Length of sensor history window fed to the LSTM |
| `model.placement` | `QR` | `QR` (greedy QR-pivot) or `random` |
| `model.hidden_size` / `hidden_layers` | 64 / 2 | LSTM hidden size and stack depth |
| `model.l1` / `l2` | 350 / 400 | Decoder MLP widths |
| `data.test_size` | 10 | Last N windows held out for testing |
| `data.val_size` | 20 | Validation windows preceding the test set |
| `train.epochs` / `patience` | 1000 / 5 | Max epochs + patience (× 20 epochs of no improvement) |

## Sensor noise model (per sensor)

You can inject sensor corruption with a dedicated Hydra config group under `configs/noise/`.

- `noise=disabled` (default): all sensors return true measurements.
- `noise=per_sensor`: per-sensor modes are applied from `noise.modes`.

Supported per-sensor modes:

- `"true"` — return true sensor value.
- `"white"` — add Gaussian white noise (`noise.white_std`).
- `"none"` — sensor still returns a noisy reading; currently this uses the same Gaussian noise path as `"white"`.

Example (three sensors: true / white / dead):

```bash
uv run python scripts/run_cylinder_baseline.py noise=per_sensor
```

Or override inline:

```bash
uv run python scripts/run_cylinder_baseline.py \
  noise.enabled=true \
  noise.modes='["true","white","none"]' \
  noise.white_std=0.02 \
  noise.none_fill_value=0.0
```

Notes:

- Noise is applied to SHRED/SDN sensor inputs and to QR/POD sensor measurements.
- `noise.auto_extend=true` lets short mode lists be padded with `noise.default_mode`.
- `noise.seed` controls noise reproducibility (falls back to main `seed` when null).

## Sweep: error vs number of sensors × placement

Reproduces the paper's Fig 2B / 3B / 4B style plot — relative L2 error vs sensor count, with **both QR-pivoted and random placement**, for SHRED, SDN, and the QR/POD baseline.

```bash
uv run python scripts/sweep_num_sensors.py
```

Writes to `outputs/cylinder/sweep/`:

- **`error_vs_num_sensors.png`** — median test error vs `num_sensors`. Up to 5 lines: SHRED (QR), SHRED (random), SDN (QR), SDN (random), and QR/POD (linear baseline; only QR placement is plotted since the linear inverse becomes ill-conditioned with random sensors).
- **`sweep_results.npz`** — raw error arrays of shape `(sensor_count, seed)` for every `(method, placement)` combination, keyed as e.g. `shred_QR`, `sdn_random`, etc.

Defaults sweep `num_sensors ∈ {1, 2, 3, 5, 8, 12, 20}` × placements `{QR, random}` × seeds `{0, 1, 2}`. Tune in [configs/sweep_num_sensors.yaml](configs/sweep_num_sensors.yaml):

```bash
# faster (single seed, fewer sensor counts)
uv run python scripts/sweep_num_sensors.py sweep.seeds=[0] sweep.sensor_counts=[1,3,10,20]

# QR placement only (skip the random comparison)
uv run python scripts/sweep_num_sensors.py sweep.placements=[QR]
```

The sweep is the long-running job — runtime scales as `len(sensor_counts) × len(placements) × len(seeds) × 2 networks × min(epochs, patience-stopped)`. Three nested tqdm bars show progress: `sensor sweep` → `n_sensors=N <placement>` → `train SHRED/SDN`. The training bar shows running `loss`, `val` (relative L2 every 20 epochs), `best`, and `patience` counter.

## Implementation notes

- The split is **sequential** over sliding windows (last `test_size` for test, preceding `val_size` for val, rest for train) — chosen for the small cylinder dataset (~150 snapshots). Paper experiments on SST/turbulence use random interleaved splits since they have many more frames.
- `MinMaxScaler` is fit on training rows only and applied globally (matches paper).
- `fit()` validates every 20 epochs, restores best parameters on early stopping (matches paper's `models.fit`).
- QR/POD uses the *unscaled* training POD basis and reconstructs from unscaled sensor measurements at the test timestamps.
- SHRED architecture defaults (`hidden_size=64, hidden_layers=2, l1=350, l2=400`) match paper's `models.SHRED`.

## Reference

Williams, J. P., Zahn, O., & Kutz, J. N. (2024). *Sensing with shallow recurrent decoder networks.* arXiv:2301.12011. Original code: [github.com/JanWilliams/pyshred](https://github.com/JanWilliams/pyshred).
