import yaml
import time
import cv2
import numpy as np
from pathlib import Path

from robot_env import RobotEnv
from yolo_manager import YOLOManager
from video_player import VideoPlayer

from controller import (
    MaxPerformanceController,
    AdaptiveController,
    LongTermBatteryAwareController,
    PEOController,
    # SynchronousMultiScaleController,
    MultiScaleController
)


# ==========================================
# Load configs
# ==========================================
with open("configs/mission.yaml", "r") as f:
    mission_config = yaml.safe_load(f)

with open("configs/robot.yaml", "r") as f:
    robot_config = yaml.safe_load(f)

control_period = mission_config["control_period"]


# ==========================================
# Controllers
# ==========================================

controllers = {
    "ASMF": MaxPerformanceController,
    "RRL": PEOController,
    "LTBA": LongTermBatteryAwareController,
    # "SMS": SynchronousMultiScaleController,
    "OURS": MultiScaleController
}

MISSION = [
    {
        "complexity": "low",
        "video": "videos/people_detection.mp4",
        "frame_size": 416,
        "target_fps": 30,
        "duration": 20
    },
    {
        "complexity": "medium",
        "video": "videos/bottle_detection.mp4",
        "frame_size": 640,
        "target_fps": 30,
        "duration": 20
    },
    {
        "complexity": "high",
        "video": "videos/fruit_and_vegetable_detection.mp4",
        "frame_size": 960,
        "target_fps": 30,
        "duration": 20
    },
    {
        "complexity": "medium",
        "video": "videos/bottle_detection.mp4",
        "frame_size": 640,
        "target_fps": 30,
        "duration": 20
    },
    {
        "complexity": "low",
        "video": "videos/people_detection.mp4",
        "frame_size": 416,
        "target_fps": 30,
        "duration": 20
    },
]

MISSION_DISTANCE = 1000.0      # meters
EOL_SOH = 80.0

summary_log = open("logs/results_mission_cnt.log", "a")
summary_log.write("\n=============================================\n")
summary_log.write(f"Experiment started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
summary_log.write("=============================================\n")

def run_one_mission(controller, workload, appl, mission_dist=1000.0):
    control_period = mission_config["control_period"]

    player = VideoPlayer(workload["video"], reference_speed=1.0)
    appl.set_scene_complexity(workload["complexity"])

    distance = 0.0
    mission_time = 0.0
    total_energy = 0.0
    qos_count = 0
    total_steps = 0
    last_speed = 1.0

    yolo_data = {
        "fps": 0,
        "target_fps": workload["target_fps"],
        "object_count": 0,
        "complexity_id": 1.0
    }

    step_times = []

    while distance < mission_dist:
        frame = player.get_frame(speed=last_speed, control_period=control_period)

        start = time.perf_counter()
        action = controller.step(yolo_data)
        step_times.append(time.perf_counter() - start)
        last_speed = action["speed"]

        robot_state = robot.step(
            action,
            control_period
        )

        yolo_data = yolo.run(
            action["cpu_freq"],
            action["gpu_freq"],
            input_data=frame,
            frame_size=workload["frame_size"]
        )

        if yolo_data["fps"] >= workload["target_fps"]:
            qos_count += 1

        p_total = robot_state["p_comp"] + robot_state["p_mech"]
        total_energy += p_total * control_period
        distance += action["speed"] * control_period

        mission_time += control_period
        total_steps += 1

    player.release()

    return {
        "distance": distance,
        "time": mission_time,
        "energy": total_energy,
        "energy_per_distance": total_energy / distance,
        "qos": 100 * qos_count / total_steps,
        "avg_step_time": np.mean(step_times),
        "estimated_soh": robot.battery.estimated_soh,
        "capacity": robot.battery.capacity_ah,
        "temperature": robot.battery.temperature,
    }
    

# ==========================================
# Run experiments
# ==========================================

for controller_name, ControllerClass in controllers.items():

    print("\n================================================")
    print(f"Running {controller_name}")
    print("================================================")

    robot = RobotEnv(robot_config, is_sim=False)

    yolo = YOLOManager(
        is_sim=False,
        robot=robot,
        model_path="./checkpoints/yolov8n.pt"
    )
    yolo.set_enabled(True)

    controller = ControllerClass(robot)
    # robot.battery.reset(initial_soc=1.0)
    # print(robot.battery.read())

    mission_count = 0

    for mission in MISSION:
        # while robot.battery.estimated_soh > 0.8:
        while mission_count < 100:

            result = run_one_mission(
                controller,
                mission,
                yolo,
                mission_dist=MISSION_DISTANCE,
            )

            mission_count += 1

            print(
                f"Mission {mission_count:4d} | "
                f"SOH={result['estimated_soh']:6f}"
            )

        final_soh = result['estimated_soh']
        final_count = 100 * (0.2 / (1.0 - final_soh))

        print(
            f"\nBattery EOL after "
            f"{mission_count} missions."
            f"Final SOH:{final_soh}:.5f "
            f"Final count:{final_count}:.2f"
        )
    
        summary_log.write(
            f"{controller_name:8s} | "
            f"Final SOH={final_soh:5f}% | "
            f"Final count:{final_count}:.2f\n"
        )
        summary_log.flush()

print("\nAll Experiments Finished")
summary_log.close()