import numpy as np
import torch

SPEED_LEVELS = np.arange(0.5, 5.0 + 0.5, 0.5)
CPU_LEVELS = [422400, 576000, 729600, 883200, 1036800, 1190400, 1344000, 1497600, 1651200, 1804800, 1958400, 1984000]
GPU_LEVELS = [306000000, 408000000, 510000000, 612000000, 714000000, 816000000, 918000000]

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

def predict_total_power(speed, cpu_freq, gpu_freq):
    return (
        3.7645
        + 2.180e-6 * cpu_freq
        + 1.644e-9 * gpu_freq
        + 1.14005 * speed
        + 0.10893 * speed**3
        + 1.1625
    )

def predict_fps(cpu_freq, gpu_freq, complexity_value):
    """
    Linear FPS prediction model.

    Args:
        cpu_freq : Hz
        gpu_freq : Hz

    Returns:
        fps
    """

    fps = (
        3.6869
        + 1.92172226e-5 * cpu_freq
        + 2.82835822e-9 * gpu_freq
    )

    fps = fps * complexity_value

    return fps

def evaluate_cost(fps, target_fps, power, speed): 
    """ Smaller is better. """ 
    # ---------------------------- 
    # QoS penalty 
    # ---------------------------- 
    qos = max( target_fps - fps, 0.0 ) 
    
    # ---------------------------- 
    # Energy per distance 
    # ---------------------------- 
    energy = power / max(speed, 1e-3) 
    total = (1.0*qos + 1.0*energy) 
    
    return total

def find_best_action(target_fps, complexity_value):
    best_cost = float("inf")
    best_action = torch.tensor(
        [[
            SPEED_LEVELS[0] / SPEED_LEVELS[-1],
            CPU_LEVELS[0] / CPU_LEVELS[-1],
            GPU_LEVELS[0] / GPU_LEVELS[-1],
        ]],
        dtype=torch.float32,
    )

    for speed in SPEED_LEVELS:
        for cpu in CPU_LEVELS:
            for gpu in GPU_LEVELS:

                fps = predict_fps(cpu, gpu, complexity_value)
                power = predict_total_power(
                    speed,
                    cpu,
                    gpu
                )
                cost = evaluate_cost(
                    fps=fps,
                    target_fps=target_fps,
                    power=power,
                    speed=speed,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_action = torch.tensor( 
                        [[ speed / SPEED_LEVELS[-1], 
                           cpu/CPU_LEVELS[-1], 
                           gpu/GPU_LEVELS[-1] ]
                        ], 
                        dtype=torch.float32, 
                        device=device
                    )

    return {
        "action": best_action,
        "speed": speed,
        "cpu": cpu,
        "gpu": gpu,
        "fps": fps,
        "power": power,
        "cost": cost
    }

from multi_scale_control import EnergyEfficientMultiScaleController
from robot_env import RobotEnv
from yolo_manager import YOLOManager

# ==========================================
# Load configs
# ==========================================
import yaml
import random
with open("configs/mission.yaml", "r") as f:
    mission_cfg = yaml.safe_load(f)

with open("configs/robot.yaml", "r") as f:
    robot_cfg = yaml.safe_load(f)

control_period = mission_cfg["control_period"]

robot_env = RobotEnv(
    robot_cfg,
    is_sim=False
)

appl_yolo = YOLOManager(    
    is_sim=True,
    robot=robot_env
)

controller = EnergyEfficientMultiScaleController(
    state_dim=robot_cfg["controller"]["state_dim"],
    adapter_dim=32,
    hidden_dim=32,
    n_heads=4,
    num_decoder_layers=2,
    action_dim=robot_cfg["controller"]["action_dim"],
).to(device)

print("Controller initialized.")
controller.train()

TARGET_FPS = 30
COMPLEXITY_LEVELS = {
    "low": 1.0,
    "medium": 0.9,
    "high": 0.8
}
num_epochs = 100
num_steps = 100

import torch.optim as optim
import torch.nn as nn
optimizer = optim.Adam(
    controller.parameters(),
    lr=1e-4
)
for epoch in range(num_epochs):

    epoch_loss = 0
    appl_yolo.set_enabled(True)

    for step in range(num_steps):
        complexity_key = random.choice(list(COMPLEXITY_LEVELS.keys()))
        complexity_value = COMPLEXITY_LEVELS[complexity_key]
        appl_yolo.set_scene_complexity(complexity_key)

        # ------------------------------------
        # Random robot action
        # (collect different states)
        # ------------------------------------
        action = {
            "speed": random.choice(SPEED_LEVELS),
            "cpu_freq": random.choice(CPU_LEVELS),
            "gpu_freq": random.choice(GPU_LEVELS)
        }

        state = robot_env.step(action)
        yolo_data = appl_yolo.run(action["cpu_freq"], action["gpu_freq"])

        # ------------------------------------
        # Battery feature
        # ------------------------------------
        health_feature = torch.from_numpy(
            robot_env.battery.get_health_feature()
        ).float().unsqueeze(0).to(device)

        # ------------------------------------
        # Robot state
        # ------------------------------------
        robot_state = torch.tensor(
            [[
                state["p_mech"] / robot_env.max_mech_power,
                state["p_comp"] / robot_env.max_comp_power,
                state["speed"] / robot_env.motor.max_speed,
                yolo_data["fps"] / 45,
                TARGET_FPS / 45,
                complexity_value,
            ]],
            dtype=torch.float32
        )

        # ------------------------------------
        # Expert label
        # ------------------------------------
        teacher = find_best_action(TARGET_FPS, complexity_value)
        target_action = teacher["action"].to(device)

        pred_action = controller(
            state=robot_state,
            health_feature=health_feature
        )

        # ------------------------------------
        # Loss
        # ------------------------------------
        loss = nn.functional.mse_loss(
            pred_action,
            target_action
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch+1:03d} | "
        f"Loss = {epoch_loss/num_steps:.5f} | "
        f"Pred Action = {pred_action} | "
        f"Target Action = {target_action}"
    )

torch.save(
    controller.state_dict(), "./checkpoints/multi_scale_model.pt"
)