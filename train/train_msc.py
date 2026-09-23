#!/usr/bin/env python3
"""Train the runtime multi-scale controller from generate_dataset.py NPZ files.

python -m multi_scale.train.train_msc --dataset multi_scale/data/expert_dataset.npz
Raw battery histories are encoded once with the configured pretrained encoder.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from methods.multi_scale_control import EnergyEfficientMultiScaleController
except ImportError:  # pragma: no cover
    from multi_scale.methods.multi_scale_control import EnergyEfficientMultiScaleController


def encode_health(history, cfg):
    """Encode raw battery telemetry into the 32-dimensional health feature used by the controller.

    This matches the preprocessing used by BatteryMonitor in envs/robot_env.py.
    """
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 3:
        raise ValueError('Battery history must have shape (N, window, features)')
    if history.shape[1:] != (cfg['battery']['window_size'], 3):
        raise ValueError(f'Expected battery history shape (N, {cfg["battery"]["window_size"]}, 3); got {history.shape}')

    enc_path = Path(cfg['battery']['encoder_path'])
    if not enc_path.is_absolute():
        enc_path = (ROOT / enc_path).resolve()
    session = ort.InferenceSession(str(enc_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    scaled = history.copy()
    for channel_idx, scaler_name in enumerate(('encoder_x0_scaler', 'encoder_x1_scaler', 'encoder_x2_scaler')):
        scaler_path = Path(cfg['battery'][scaler_name])
        if not scaler_path.is_absolute():
            scaler_path = (ROOT / scaler_path).resolve()
        with scaler_path.open('rb') as fh:
            scaler = pickle.load(fh)
        scaled[:, :, channel_idx] = scaler.transform(scaled[:, :, channel_idx])

    encoded = session.run(None, {input_name: scaled.astype(np.float32)})[0]
    encoded = np.asarray(encoded, dtype=np.float32)
    if encoded.shape[1] != 32:
        raise ValueError(f'Expected 32-dim health feature, but got shape {encoded.shape}')
    return encoded

# Must match methods/controller.py:MultiScaleController for direct deployment.
MODEL_CONFIG = dict(state_dim=6, adapter_dim=32, hidden_dim=32,
                    n_heads=4, num_decoder_layers=2, action_dim=3)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        for key in ('state', 'target_action', 'metadata'):
            if key not in archive:
                raise ValueError(f'Dataset is missing {key}')
        metadata = json.loads(str(archive['metadata']))
        if metadata.get('schema_version') != 1 or metadata.get('target_encoding') != 'speed/max_speed; CPU/GPU runtime index-bin centres':
            raise ValueError('Unsupported schema or action encoding; regenerate with generate_dataset.py')
        cfg = metadata['config']
        if cfg['battery']['encoder_dim'] != 32:
            raise ValueError('Runtime controller requires 32-dimensional health features')
        state = np.asarray(archive['state'], dtype=np.float32)
        target = np.asarray(archive['target_action'], dtype=np.float32)
        count = len(state)
        if count < 2 or state.shape != (count, 6) or target.shape != (count, 3):
            raise ValueError('Expected at least two samples with state (N,6) and target_action (N,3)')
        if not np.isfinite(state).all() or not np.isfinite(target).all():
            raise ValueError('Dataset contains non-finite states or labels')
        if np.any((target < 0) | (target > 1)):
            raise ValueError('Action labels must be in [0,1]')
        if 'health_feature' in archive:
            health = np.asarray(archive['health_feature'], dtype=np.float32)
        else:
            if 'battery_history' not in archive:
                raise ValueError('Dataset needs health_feature or battery_history')
            history = np.asarray(archive['battery_history'], dtype=np.float32)
            if history.shape != (count, cfg['battery']['window_size'], 3) or not np.isfinite(history).all():
                raise ValueError('Invalid battery history shape or values')
            print('Encoding battery histories with the pretrained ONNX encoder...', flush=True)
            try:
                health = encode_health(history, cfg)
            except (ImportError, FileNotFoundError) as exc:
                raise RuntimeError('Battery encoding requires ONNX Runtime, scaler dependencies '
                                   '(including scikit-learn), and configured model files. '
                                   'Alternatively supply an NPZ containing health_feature.') from exc
        if health.shape != (count, 32) or not np.isfinite(health).all():
            raise ValueError('Expected finite health_feature with shape (N,32)')
    return TensorDataset(*(torch.from_numpy(x.copy()) for x in (state, health, target))), metadata


def run_epoch(model, loader, device, cfg, optimizer=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    squared_error = np.zeros(3, dtype=np.float64)
    speed_error, cpu_correct, gpu_correct, count = 0.0, 0, 0, 0
    counts = [len(cfg['jetson_orin_nx_16g'][key]) for key in ('cpu_freqs', 'gpu_freqs')]
    with torch.set_grad_enabled(training):
        for state, health, target in loader:
            state, health, target = (x.to(device) for x in (state, health, target))
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(state=state, health_feature=health)
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training/validation loss')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            squared_error += ((prediction.detach()-target)**2).sum(0).cpu().numpy()
            decoded = prediction.detach().clamp(0, 1)
            speed_error += float((decoded[:, 0]-target[:, 0]).abs().sum()) * cfg['motor']['v_max']
            cpu_correct += int(((decoded[:, 1]*(counts[0]-1)).long() == (target[:, 1]*(counts[0]-1)).long()).sum())
            gpu_correct += int(((decoded[:, 2]*(counts[1]-1)).long() == (target[:, 2]*(counts[1]-1)).long()).sum())
            count += len(state)
    return dict(mse=float(squared_error.sum()/(count*3)),
                per_action_mse=(squared_error/count).tolist(), speed_mae_mps=speed_error/count,
                cpu_accuracy=cpu_correct/count, gpu_accuracy=gpu_correct/count)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'train'/'dataset'/'expert_dataset.npz',
                   help='Path to the training dataset NPZ file.')
    p.add_argument('--output-dir', type=Path, default=ROOT/'checkpoints'/'msc_training',
                   help='Directory where checkpoints and logs are written.')
    p.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    p.add_argument('--batch-size', type=int, default=128, help='Mini-batch size.')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    p.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay.')
    p.add_argument('--val-fraction', type=float, default=0.2, help='Validation split fraction.')
    p.add_argument('--patience', type=int, default=15, help='Early stopping epochs; 0 disables.')
    p.add_argument('--grad-clip', type=float, default=1, help='Gradient-norm clipping value.')
    p.add_argument('--seed', type=int, default=7, help='Random seed.')
    p.add_argument('--threads', type=int, default=1, help='PyTorch CPU thread count.')
    p.add_argument('--device', default='cpu', help='cpu, cuda, or cuda:N')
    args = p.parse_args(argv)
    for key in ('epochs', 'batch_size', 'threads', 'lr', 'grad_clip'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f'{key} must be finite and positive')
    if (not 0 < args.val_fraction < 1 or not math.isfinite(args.weight_decay)
            or args.weight_decay < 0 or args.patience < 0 or not 0 <= args.seed < 2**32):
        p.error('Invalid split, weight decay, patience, or seed')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.dataset = args.dataset.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        raise FileExistsError(f'Choose a new --output-dir; already exists: {args.output_dir}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    dataset, metadata = load_dataset(args.dataset)
    if metadata.get('battery_affects_expert') is False:
        print('Note: expert labels do not depend on battery history; validation measures imitation only.')
    order = np.random.default_rng(args.seed).permutation(len(dataset))
    val_count = max(1, min(len(dataset)-1, round(len(dataset)*args.val_fraction)))
    val_indices, train_indices = order[:val_count], order[val_count:]
    train_loader = DataLoader(Subset(dataset, train_indices.tolist()), batch_size=args.batch_size,
                              shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(Subset(dataset, val_indices.tolist()), batch_size=args.batch_size)
    model = EnergyEfficientMultiScaleController(**MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True)
    np.savez(args.output_dir/'split.npz', train_indices=train_indices, val_indices=val_indices)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir/'config.json').write_text(json.dumps(dict(arguments=settings, architecture=MODEL_CONFIG,
                                                             dataset_metadata=metadata), indent=2)+'\n')
    best_loss, best_epoch, stale_epochs = float('inf'), 0, 0
    with (args.output_dir/'metrics.jsonl').open('w') as log:
        for epoch in range(1, args.epochs+1):
            train_metrics = run_epoch(model, train_loader, device, metadata['config'], optimizer, args.grad_clip)
            val_metrics = run_epoch(model, val_loader, device, metadata['config'])
            log.write(json.dumps(dict(epoch=epoch, train=train_metrics, validation=val_metrics))+'\n')
            log.flush()
            if val_metrics['mse'] < best_loss:
                best_loss, best_epoch, stale_epochs = val_metrics['mse'], epoch, 0
                # Plain state_dict is directly accepted by MultiScaleController.
                weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
                temp = args.output_dir/'best.tmp'
                torch.save(weights, temp)
                temp.replace(args.output_dir/'best.pt')
            else:
                stale_epochs += 1
            print(f'Epoch {epoch:03d}: train={train_metrics["mse"]:.6f} val={val_metrics["mse"]:.6f} '
                  f'speed_MAE={val_metrics["speed_mae_mps"]:.3f}m/s '
                  f'CPU_acc={val_metrics["cpu_accuracy"]:.1%} GPU_acc={val_metrics["gpu_accuracy"]:.1%}', flush=True)
            if args.patience and stale_epochs >= args.patience:
                print('Early stopping.')
                break
    (args.output_dir/'summary.json').write_text(json.dumps(dict(best_epoch=best_epoch,
        best_validation_mse=best_loss, epochs_completed=epoch, train_samples=len(train_indices),
        validation_samples=len(val_indices)), indent=2)+'\n')
    print(f'Best checkpoint: {args.output_dir / "best.pt"} (epoch {best_epoch})')


if __name__ == '__main__':
    main()
