import yaml
import torch
import joblib
import os
import numpy as np
from tensorflow import keras
from robot_env import RobotEnv


# =========================================================
# Base Controller
# =========================================================
class RuntimeController:
    def __init__(self, robot: RobotEnv):
        self.robot = robot

    def step(self, state):
        raise NotImplementedError

# =========================================================
# Max Performance Strategy
# =========================================================
class MaxPerformanceController(RuntimeController):
    def step(self, state):
        return {
            "cpu_freq": max(self.robot.board.cpu_freqs),
            "gpu_freq": max(self.robot.board.gpu_freqs),
            "speed": 2.0
        }


# =========================================================
# Adaptive Strategy
# =========================================================
class AdaptiveController(RuntimeController):
    def __init__(self, robot):
        self.robot = robot
        self.robot.motor.set_speed(self.robot.motor.max_speed)

    def step(self, state):
        fps = state["fps"]
        self.target_fps = state["target_fps"]
        # ----------------------------------------
        # Current frequencies
        # ----------------------------------------
        curr_speed = self.robot.motor.get_speed()
        curr_cpu_freq = self.robot.fc_list[0]
        curr_gpu_freq = self.robot.fg

        cpu_freqs = self.robot.board.cpu_freqs
        gpu_freqs = self.robot.board.gpu_freqs

        # ----------------------------------------
        # Current indices
        # ----------------------------------------
        cpu_idx = cpu_freqs.index(curr_cpu_freq)
        gpu_idx = gpu_freqs.index(curr_gpu_freq)

        next_cpu_freq = curr_cpu_freq
        next_gpu_freq = curr_gpu_freq
        next_speed = curr_speed

        # ----------------------------------------
        # Adaptive policy
        # ----------------------------------------
        if fps < self.target_fps:
            # increase frequency gradually
            next_cpu_idx = min(
                cpu_idx + 1,
                len(cpu_freqs) - 1
            )

            next_gpu_idx = min(
                gpu_idx + 1,
                len(gpu_freqs) - 1
            )

            next_cpu_freq = cpu_freqs[next_cpu_idx]
            next_gpu_freq = gpu_freqs[next_gpu_idx]
            next_speed = max(self.robot.motor.min_speed, curr_speed - 0.2)

        elif fps > self.target_fps + 5:
            # reduce power gradually
            next_cpu_idx = max(
                cpu_idx - 1,
                0
            )

            next_gpu_idx = max(
                gpu_idx - 1,
                0
            )
            print(next_cpu_idx, next_gpu_idx)

            next_cpu_freq = cpu_freqs[next_cpu_idx]
            next_gpu_freq = gpu_freqs[next_gpu_idx]
            next_speed = min(self.robot.motor.max_speed, curr_speed + 0.2)

        return {
            "cpu_freq": next_cpu_freq,
            "gpu_freq": next_gpu_freq,
            "speed": next_speed
        }

class PEOController(RuntimeController):
    def __init__(self, robot):
        self.robot = robot
        self.robot.motor.set_speed(self.robot.motor.max_speed)

    def step(self, state):
        fps = state["fps"]
        self.target_fps = state["target_fps"]
        # ----------------------------------------
        # Current frequencies
        # ----------------------------------------
        complexity_id = state["complexity_id"]
        cpu_freq = self.robot.board.cpu_freqs[5]
        gpu_freq = self.robot.board.gpu_freqs[4]
        speed = 3.0
        if complexity_id <= 0.91: # medium
            cpu_freq = self.robot.board.cpu_freqs[6]
            gpu_freq = self.robot.board.gpu_freqs[5]
            speed = 2.7
        if complexity_id <= 0.81: # high
            cpu_freq = self.robot.board.cpu_freqs[-3]
            gpu_freq = self.robot.board.gpu_freqs[-1]
            speed = 2.5

        return {
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq,
            "speed": speed
        }

class LongTermBatteryAwareController:
    def __init__(self, robot):
        self.robot = robot
        self.control_period = 10.0

        self.cpu_freqs = self.robot.board.cpu_freqs
        self.gpu_freqs = self.robot.board.gpu_freqs

        self.action = {
            "cpu_freq": self.cpu_freqs[0],
            "gpu_freq": self.gpu_freqs[0],
            "speed": 1.0
        }
    
    def compute_health_score(self, battery_state):
        temp = battery_state["temperature"]
        current = abs(battery_state["current"])

        # Normalization
        temp_norm = min(temp / 45.0, 1.0)
        current_norm = min(current / 5.0, 1.0)

        # Temperature dominates
        health = (
            0.7 * temp_norm +
            0.3 * current_norm
        )

        return health

    def step(self, yolo_data):
        battery_state = self.robot.battery.read()

        avg_cpu_idx = len(self.cpu_freqs) // 2
        avg_gpu_idx = len(self.gpu_freqs) // 2

        health = self.compute_health_score(
            battery_state
        )

        if health > 0.8:
            self.action = {
                "cpu_freq": self.cpu_freqs[avg_cpu_idx],
                "gpu_freq": self.gpu_freqs[avg_gpu_idx],
                "speed": 2.0
            }

        elif health > 0.60:
            self.action = {
                "cpu_freq": self.cpu_freqs[avg_cpu_idx],
                "gpu_freq": self.gpu_freqs[avg_gpu_idx],
                "speed": 1.75
            }

        else:
            self.action = {
                "cpu_freq": self.cpu_freqs[avg_cpu_idx],
                "gpu_freq": self.gpu_freqs[avg_gpu_idx],
                "speed": 1.5
            }

        return self.action

from rrl_control import DQNController
class RRLController(RuntimeController):
    def __init__(
        self,
        robot: RobotEnv,
        device="cpu"
    ):
        self.device = device
        self.model = DQNController(robot)
        self.model.agent.load("./checkpoints/rrl_model.pt")
    
    def step(self, yolo_data):
        return self.model.step(yolo_data)

from multi_scale_control import EnergyEfficientMultiScaleController
class MultiScaleController(RuntimeController):

    def __init__(self, robot: RobotEnv, device="cpu"):
        self.device = device
        self.robot = robot

        self.model = (
            EnergyEfficientMultiScaleController(
                state_dim=6,
                adapter_dim=32,
                hidden_dim=32,
                n_heads=4,
                num_decoder_layers=2,
                action_dim=3
            ).to(device)
        )

        checkpoint_path = (
            "./checkpoints/multi_scale_model.pt"
        )

        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device
            )
        )

        self.model.eval()
        print("MultiScaleController loaded.")


    # ========================================================
    # Runtime step
    # ========================================================
    def step(self, yolo_data):

        # ----------------------------------------------------
        # Build runtime robot state
        # ----------------------------------------------------
        robot_state = torch.tensor(
            [[
                self.robot.get_mechanical_power() / self.robot.max_mech_power,
                self.robot.get_computational_power() / self.robot.max_comp_power,
                self.robot.get_speed() / self.robot.motor.max_speed,
                yolo_data["fps"] / 45,
                yolo_data["target_fps"] / 45,
                yolo_data["complexity_id"]
            ]],
            dtype=torch.float32,
            device=self.device
        )

        # ----------------------------------------------------
        # Battery health feature
        # ----------------------------------------------------
        health_feature = self.robot.battery.get_health_feature()
        health_feature = torch.from_numpy(health_feature).float().to(self.device)
        if health_feature.ndim == 1:
            health_feature = health_feature.unsqueeze(0)

        # ----------------------------------------------------
        # Controller inference
        # ----------------------------------------------------
        with torch.no_grad():

            pred_action = self.model(
                state=robot_state,
                health_feature=health_feature
            )

        pred_action = pred_action.cpu().numpy()[0]

        # ====================================================
        # Decode actions
        # ====================================================

        # ----------------------------------------------------
        # Speed
        # ----------------------------------------------------
        speed = np.clip(pred_action[0], 0.0, 1.0)
        speed *= self.robot.motor.max_speed

        cpu_ratio = np.clip(pred_action[1], 0.0, 1.0)
        cpu_idx = int(cpu_ratio * (len(self.robot.board.cpu_freqs) - 1))
        cpu_freq = self.robot.board.cpu_freqs[cpu_idx]

        gpu_ratio = np.clip(pred_action[2], 0.0, 1.0)
        gpu_idx = int(gpu_ratio * (len(self.robot.board.gpu_freqs) - 1))
        gpu_freq = self.robot.board.gpu_freqs[gpu_idx]

        # ----------------------------------------------------
        # Runtime action
        # ----------------------------------------------------
        return {
            "speed": float(speed),
            "cpu_freq": cpu_freq,
            "gpu_freq": gpu_freq
        }

# =========================================================
# Example
# =========================================================

with open("./configs/robot.yaml", "r") as f:
    cfg = yaml.safe_load(f)

from robot_env import RobotEnv
if __name__ == "__main__":
    robot = RobotEnv(cfg, is_sim=True)

    warm_action = {
        "speed": 0.5,
        "cpu_freq": robot.board.cpu_freqs[2],
        "gpu_freq": robot.board.gpu_freqs[2]
    }
    print("Warm up battery window...")

    for _ in range(robot.battery.window_size):
        robot.step(warm_action)

    print("Battery window ready.\n")

    controllers = {
        # "MaxPerf": MaxPerformanceController(robot),
        # "Adaptive": AdaptiveController(robot),
        "MultiScale": MultiScaleController(robot)
    }

    for name, controller in controllers.items():
        print("=" * 60)
        print(name)
        print("=" * 60)

        yolo_data = {
            "fps": 15,
            "target_fps": 30
        }
        action = controller.step(yolo_data)
        print(f"Controller output: {action}")
        print(f"Mechanical Power : {robot.get_mechanical_power():.2f} W")
        print(f"Compute Power    : {robot.get_computational_power():.2f} W")
        print(f"Speed            : {robot.get_speed():.2f} m/s")
        print(f"Battery: {robot.battery.read()}")