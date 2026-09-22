#!/usr/bin/env python3
"""Sample synthetic operating states, battery histories, and exhaustive expert labels.

python -m multi_scale.train.generate_dataset --samples 10000 --output multi_scale/data/expert.npz
Use --encode-health to also run the configured ONNX battery encoder/scalers.
No ROS, Jetson hardware, YOLO inference, or training is started.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import yaml

if __package__:
    from .expert_policy import find_best_action, predict_fps, predict_mechanical_power, predict_compute_power
else:
    from expert_policy import find_best_action, predict_fps, predict_mechanical_power, predict_compute_power

ROOT = Path(__file__).resolve().parents[1]


def sample_histories(rng, count, window, dt):
    """Synthetic single-cell ECM trajectories; columns are volts, amps, Celsius.

    Random piecewise loads create temporally correlated histories. SOH and SOC
    are sampled independently per history; these are modeled, not measured data.
    """
    soc = rng.uniform(0.4, 1.0, count)
    soh = rng.uniform(0.65, 1.0, count)
    ambient = rng.uniform(15, 35, count)
    temperature = ambient + rng.uniform(0, 8, count)
    resistance = rng.uniform(0.025, 0.075, count) * (2 - soh)
    capacity = 2.0 * soh
    initial = np.column_stack([soc, soh, temperature, resistance])
    rc1, rc2 = np.zeros(count), np.zeros(count)
    history = np.empty((count, window, 3), dtype=np.float32)
    current = rng.uniform(0.2, 4.5, count)
    a1, a2 = np.exp(-dt / (0.015 * 1800)), np.exp(-dt / (0.008 * 6000))
    for t in range(window):
        if t % 10 == 0:
            load = rng.uniform(0.2, 4.5, count)
        current = np.clip(0.85 * current + 0.15 * load + rng.normal(0, 0.03, count), 0.05, 5)
        soc = np.clip(soc - current * dt / (3600 * capacity), 0, 1)
        rc1 = a1 * rc1 + 0.015 * (1-a1) * current
        rc2 = a2 * rc2 + 0.008 * (1-a2) * current
        ocv = 3 + 1.18 * soc + 0.05 * np.tanh(8 * (soc - 0.5))
        voltage = np.clip(ocv - current * resistance - rc1 - rc2, 2.8, 4.2)
        temperature += (current**2 * (resistance + 0.023) - 0.02 * (temperature - ambient)) * dt * 0.005
        history[:, t] = np.column_stack([voltage, current, temperature])
    return history, initial.astype(np.float32), soc.astype(np.float32)


def generate_dataset(cfg, samples=10000, seed=7, dt=1.0, target_range=(15.0, 40.0)):
    if samples <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('samples and dt must be positive')
    if not 0 < target_range[0] <= target_range[1] or not np.all(np.isfinite(target_range)):
        raise ValueError('Invalid target FPS range')
    rng = np.random.default_rng(seed)
    motor, board = cfg['motor'], cfg['jetson_orin_nx_16g']
    if not 0 < motor['v_min'] <= motor['v_max'] or motor['v_step'] <= 0:
        raise ValueError('Invalid motor action range')
    speeds = np.arange(motor['v_min'], motor['v_max'] + 1e-9, motor['v_step'])
    if not np.isclose(speeds[-1], motor['v_max']):
        raise ValueError('Motor maximum must lie on the speed grid')
    cpus, gpus = np.asarray(board['cpu_freqs']), np.asarray(board['gpu_freqs'])
    window = int(cfg['battery']['window_size'])
    if window <= 0 or min(cfg['controller']['max_mech_power'], cfg['controller']['max_comp_power']) <= 0:
        raise ValueError('Window and power normalizers must be positive')
    speed, cpu, gpu = rng.choice(speeds, samples), rng.choice(cpus, samples), rng.choice(gpus, samples)
    complexity = rng.choice([1.0, 0.9, 0.8], samples)
    target = rng.uniform(*target_range, size=samples)
    fps = predict_fps(cpu, gpu, complexity)
    state = np.column_stack([
        predict_mechanical_power(speed) / cfg['controller']['max_mech_power'],
        predict_compute_power(cpu, gpu) / cfg['controller']['max_comp_power'],
        speed / motor['v_max'], fps / 45, target / 45, complexity,
    ]).astype(np.float32)
    histories, initial, final_soc = sample_histories(rng, samples, window, dt)
    teachers = [find_best_action(t, c, speeds, cpus, gpus) for t, c in zip(target, complexity)]
    return {
        'state': state,
        'battery_history': histories,
        'battery_initial': initial,
        'battery_final_soc': final_soc,
        'operating_action': np.column_stack([speed, cpu, gpu]),
        'target_action': np.stack([x['action'] for x in teachers]),
        'expert_action': np.array([[x['speed'], x['cpu'], x['gpu']] for x in teachers]),
        'expert_metrics': np.array([[x['fps'], x['power'], x['cost']] for x in teachers]),
        'target_fps': target,
        'complexity': complexity,
    }


def encode_health(history, cfg):
    import pickle
    import onnxruntime as ort

    def resolve(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p

    scaled = history.copy()
    for channel, key in enumerate(('encoder_x0_scaler', 'encoder_x1_scaler', 'encoder_x2_scaler')):
        with resolve(cfg['battery'][key]).open('rb') as stream:
            scaler = pickle.load(stream)
        scaled[:, :, channel] = scaler.transform(history[:, :, channel])
    session = ort.InferenceSession(str(resolve(cfg['battery']['encoder_path'])), providers=['CPUExecutionProvider'])
    # Single samples also support encoder exports with a fixed batch size of one.
    features = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0]
                               for row in scaled], axis=0)
    if features.shape != (len(history), cfg['battery']['encoder_dim']) or not np.all(np.isfinite(features)):
        raise ValueError('Unexpected battery encoder output')
    return features.astype(np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dt', type=float, default=1)
    p.add_argument('--target-fps-min', type=float, default=15)
    p.add_argument('--target-fps-max', type=float, default=40)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/robot.yaml')
    p.add_argument('--output', type=Path, default=ROOT / 'data/expert_dataset.npz')
    p.add_argument('--encode-health', action='store_true')
    args = p.parse_args(argv)
    with args.config.expanduser().open() as stream:
        cfg = yaml.safe_load(stream)
    output = args.output.expanduser()
    if output.exists():
        p.error(f'Output already exists: {output}; choose another path')
    data = generate_dataset(cfg, args.samples, args.seed, args.dt, (args.target_fps_min, args.target_fps_max))
    if args.encode_health:
        data['health_feature'] = encode_health(data['battery_history'], cfg)
    metadata = {
        'schema_version': 1, 'seed': args.seed, 'samples': args.samples, 'dt_s': args.dt,
        'state_columns': ['p_mech/max_mech_power', 'p_comp/max_comp_power', 'speed/max_speed',
                          'fps/45', 'target_fps/45', 'complexity'],
        'battery_history_columns': ['voltage_V', 'current_A', 'temperature_C'],
        'battery_initial_columns': ['soc', 'soh', 'temperature_C', 'resistance_ohm'],
        'action_columns': ['speed_mps', 'cpu_kHz', 'gpu_Hz'],
        'expert_metrics_columns': ['fps', 'power_W', 'cost'],
        'target_encoding': 'speed/max_speed; CPU/GPU runtime index-bin centres',
        'expert_cost': 'max(target_fps-predicted_fps,0) + total_power/speed',
        'battery_affects_expert': False,
        'history_source': 'Synthetic single-cell ECM; sampled independently of current operating action',
        'power_source': 'Linear compute and cubic motor regressions from train_msc.py expert',
        'config': cfg,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, **data, metadata=np.array(json.dumps(metadata)))
    print(f'Saved {len(data["state"])} samples to {output}')
    print('Arrays:', ', '.join(f'{k}{v.shape}' for k, v in data.items()))


if __name__ == '__main__':
    main()
