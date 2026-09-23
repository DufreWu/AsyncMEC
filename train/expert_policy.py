"""Offline exhaustive QoS/energy expert. CPU frequencies are kHz, GPU Hz.

The cost matches train_msc.py: FPS shortfall + energy per travelled metre.
Battery history does not affect this expert's decisions.
"""
import numpy as np

SPEED_LEVELS = np.arange(0.5, 5.01, 0.25)
CPU_LEVELS = [422400, 576000, 729600, 883200, 1036800, 1190400,
              1344000, 1497600, 1651200, 1804800, 1958400, 1984000]
GPU_LEVELS = [306000000, 408000000, 510000000, 612000000, 714000000, 816000000, 918000000]


def predict_mechanical_power(speed):
    return 1.14005 * speed + 0.10893 * speed**3 + 1.1625


def predict_compute_power(cpu_freq, gpu_freq):
    return 3.7645 + 2.180e-6 * cpu_freq + 1.644e-9 * gpu_freq


def predict_total_power(speed, cpu_freq, gpu_freq):
    return predict_mechanical_power(speed) + predict_compute_power(cpu_freq, gpu_freq)


def predict_fps(cpu_freq, gpu_freq, complexity_value):
    return (3.6869 + 1.92172226e-5 * cpu_freq + 2.82835822e-9 * gpu_freq) * complexity_value


def evaluate_cost(fps, target_fps, power, speed):
    return np.maximum(target_fps - fps, 0) + power / np.maximum(speed, 1e-3)


def _index_label(index, count):
    # Bin centres survive float32 rounding with runtime int(ratio * (N - 1)).
    return 0.0 if count == 1 else (1.0 if index == count - 1 else (index + 0.5) / (count - 1))


def find_best_action(target_fps, complexity_value, speed_levels=SPEED_LEVELS,
                     cpu_levels=CPU_LEVELS, gpu_levels=GPU_LEVELS):
    if not np.isfinite(target_fps) or target_fps <= 0 or not 0 < complexity_value <= 1:
        raise ValueError('Expected positive target FPS and complexity in (0, 1]')
    levels = [np.asarray(x, dtype=float) for x in (speed_levels, cpu_levels, gpu_levels)]
    for values in levels:
        if (values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values))
                or np.any(values <= 0) or np.any(np.diff(values) <= 0)):
            raise ValueError('Action levels must be finite, positive and strictly increasing')
    speed, cpu, gpu = np.meshgrid(*levels, indexing='ij')
    fps = predict_fps(cpu, gpu, complexity_value)
    power = predict_total_power(speed, cpu, gpu)
    cost = evaluate_cost(fps, target_fps, power, speed)
    index = np.unravel_index(np.argmin(cost), cost.shape)
    return {
        'action': np.array([speed[index] / levels[0][-1],
                            _index_label(index[1], len(levels[1])),
                            _index_label(index[2], len(levels[2]))], dtype=np.float32),
        'speed': float(speed[index]), 'cpu': int(cpu[index]), 'gpu': int(gpu[index]),
        'fps': float(fps[index]), 'power': float(power[index]), 'cost': float(cost[index]),
    }
