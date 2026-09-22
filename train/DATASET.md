# Offline expert dataset

From the repository root:

```bash
python -m multi_scale.train.generate_dataset --samples 10000 --seed 7 \
  --output multi_scale/data/expert_dataset.npz
```

The generator samples speed and CPU/GPU settings from `configs/robot.yaml`,
complexity factors from 1.0/0.9/0.8, and target FPS uniformly from 15–40.
Use `--target-fps-min`, `--target-fps-max`, `--dt`, or `--config` to override.
Existing output files are not overwritten.

The NPZ contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `state` | N × 6 | Normalized power, speed, FPS, target FPS, complexity |
| `battery_history` | N × window_size × 3 | Synthetic voltage/current/temperature histories |
| `target_action` | N × 3 | Runtime-compatible normalized expert labels |
| `expert_action` | N × 3 | Speed (m/s), CPU (kHz), GPU (Hz) |
| `expert_metrics` | N × 3 | Predicted FPS, total watts, expert cost |
| `operating_action` | N × 3 | Sampled operating speed and frequencies |
| `battery_initial` | N × 4 | Initial SOC, SOH, temperature, resistance |
| `battery_final_soc` | N | Final modeled SOC |
| `target_fps`, `complexity` | N | Unnormalized operating requirements |
| `metadata` | scalar JSON string | Schema, seed, units, config and model assumptions |

`--encode-health` additionally creates `health_feature` (N × encoder_dim) using
the configured ONNX encoder and its pickled channel scalers. This requires ONNX
Runtime and the scaler dependencies (including scikit-learn). Without this flag,
raw histories are saved; they must be encoded before use with the model's
`health_feature` input. No placeholder health features are generated.

```python
import numpy as np
with np.load('multi_scale/data/expert_dataset.npz', allow_pickle=False) as data:
    states = data['state']
    histories = data['battery_history']
    labels = data['target_action']
```

Each history is an independent synthetic single-cell ECM trajectory with
piecewise loads, varying SOC/SOH, temperature, and resistance. Histories are
sampled independently of the current operating action, not measured Jetson
battery telemetry. The power/FPS regressions match the expert in `train_msc.py`,
not the different compute-power approximation in `RobotEnv`.

The expert exhaustively minimizes FPS shortfall plus energy per metre. Battery
history does not enter this objective: these labels cannot teach a battery-aware
policy. Frequency labels use runtime index-bin centres, rather than frequency
ratios, to match `MultiScaleController`'s index decoder. Expert metrics describe
the winning candidate. `train_msc.py` consumes these saved labels directly; it does not recompute the expert during training.


## Train the controller

```bash
python -m multi_scale.train.train_msc \
  --dataset multi_scale/data/expert_dataset.npz \
  --output-dir multi_scale/checkpoints/msc_training \
  --epochs 100 --batch-size 128 --device cpu
```

The trainer validates dataset shapes, finite values and action encoding. If
`health_feature` is absent, it runs the pretrained encoder once over the saved
histories; ONNX Runtime, scikit-learn and the configured encoder/scalers must be
available. It never substitutes zero features. Existing health features bypass
encoding.

A seeded sample split reserves 20% for validation. This is appropriate for the
generator's independent histories; datasets with overlapping trajectory windows
need a trajectory-level split before using this trainer. Validation measures
imitation of the synthetic expert, not hardware performance or battery benefit.

Training uses AdamW, mini-batches, gradient clipping and early stopping (15 epochs
without improved validation MSE). The architecture matches the runtime controller.
The output directory must be new and contains:

- `best.pt`: best-validation plain state dictionary, loadable by `MultiScaleController`.
- `metrics.jsonl`: train/validation MSE, per-action MSE, decoded speed MAE and CPU/GPU accuracy.
- `split.npz`: exact training and validation row indices.
- `config.json`: training options, architecture and dataset metadata.
- `summary.json`: best epoch, validation loss and sample counts.

The existing deployment checkpoint is not overwritten. Select the new model with
`MultiScaleController(robot, checkpoint_path=".../best.pt")`, or pass
`--checkpoint .../best.pt` to the ROS controller launcher. Match the dataset's
motor limits, frequency tables and normalization settings to the deployment config.
Seeded CPU runs are reproducible in the same environment; exact results across
hardware and library versions are not guaranteed.
