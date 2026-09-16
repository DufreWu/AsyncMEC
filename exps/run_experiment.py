import yaml
import time
import cv2
import numpy as np
from pathlib import Path

from robot_env import RobotEnv
from yolo_manager import YOLOManager
from segformer_manager import SegFormerManager
from video_player import VideoPlayer

from controller import (
    MaxPerformanceController,
    AdaptiveController,
    PEOController,
    LongTermBatteryAwareController,
    RRLController,
    MultiScaleController
)

from logger import ExperimentLogger

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
    # "ASMF": MaxPerformanceController,
    # "RRL": AdaptiveController,
    # "RRL": PEOController,
    # "LTBA": LongTermBatteryAwareController,
    # "RRL": RRLController,
    # "SMS": SynchronousMultiScaleController,
    "OURS": MultiScaleController
}

MISSION = [
    {
        "name": "low",
        "video": "videos/people_detection.mp4",
        "frame_size": 416,
        "target_fps": 30,
        "duration": 20
    },
    {
        "name": "medium",
        "video": "videos/bottle_detection.mp4",
        "frame_size": 640,
        "target_fps": 30,
        "duration": 20
    },
    {
        "name": "high",
        "video": "videos/fruit_and_vegetable_detection.mp4",
        "frame_size": 960,
        "target_fps": 30,
        "duration": 20
    },
    {
        "name": "medium",
        "video": "videos/bottle_detection.mp4",
        "frame_size": 640,
        "target_fps": 30,
        "duration": 20
    },
    {
        "name": "low",
        "video": "videos/people_detection.mp4",
        "frame_size": 416,
        "target_fps": 30,
        "duration": 20
    },
]

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
        model_path="./checkpoints/yolov8n.pt")
    yolo.set_enabled(True)

    segformer = SegFormerManager(
        model_path="./checkpoints/segformer-b0-finetuned-ade-512-512",
        max_fps=45,
    )

    # ---------------------------------------
    # Controller
    # ---------------------------------------       
    controller = ControllerClass(robot)

    logger = ExperimentLogger(
        f"logs/{controller_name}.csv"
    )

    global_time = 0.0
    distance = 0.0
    peak_current = 0.
    prev_action = None
    step_times = []

    yolo_data = {
        "fps": 0,
        "target_fps": 1,
        "object_count": 0,
        "complexity_id": 1.0
    }

    for stage in MISSION:
        print("\n------------------------------------")
        print(f"Stage : {stage['name']}")
        print("------------------------------------")

        player = VideoPlayer(stage["video"], reference_speed=1.0)

        steps = int(stage["duration"] / control_period)

        for _ in range(steps):

            frame = player.get_frame(speed=1.0, control_period=control_period)

            # ----------------------------------
            # Controller
            # ----------------------------------
            start = time.perf_counter()
            action = controller.step(yolo_data)
            step_times.append(time.perf_counter() - start)

            # ----------------------------------
            # Robot update
            # ----------------------------------
            print(action)
            robot_state = robot.step(
                action,
                control_period
            )
            prev_action = action
            # ----------------------------------
            # YOLO stage
            # ----------------------------------
            yolo_data = yolo.run(
                action["cpu_freq"],
                action["gpu_freq"],
                input_data=frame,
                frame_size=stage["frame_size"],
                complexity=stage["name"]
            )
            print(
                f"FPS={yolo_data['fps']:.2f} "
                f"Latency={yolo_data['latency_ms']:.2f} ms "
                f"Objects={yolo_data['object_count']}"
            )
            # seg_data = segformer.process(frame)
            # print(
            #     f"FPS={seg_data['fps']}"
            # )

            # ----------------------------------
            # Derived metrics
            # ----------------------------------
            p_comp = robot_state["p_comp"]
            # patch
            if stage["name"] == "medium":
                p_comp = p_comp * 1.1
            elif stage["name"] == "high":
                p_comp = p_comp * 1.2
            p_mech = robot_state["p_mech"]
            p_total = p_comp + p_mech
            eng_dist = p_total / action["speed"]

            current = robot_state["battery"]["current"]
            peak_current = max(
                peak_current,
                current
            )

            distance += (
                action["speed"] *
                control_period
            )

            # ----------------------------------
            # Log
            # ----------------------------------

            logger.log([
                # time
                global_time,

                # actions
                action["cpu_freq"],
                action["gpu_freq"],
                action["speed"],

                # workload
                stage["name"],
                stage["target_fps"],
                yolo_data["fps"],

                # power/energy
                p_comp,
                p_mech,
                p_total,
                eng_dist,

                # mission
                distance,

                # battery
                robot_state["battery"]["voltage"],
                current,
                robot_state["battery"]["temperature"],
                robot_state["battery"]["soc"],
                robot_state["battery"]["estimated_soh"],
                robot_state["battery"]["capacity_prediction"]
            ])

            # ==================================
            # Console
            # ==================================

            print(
                f"[{global_time:5.1f}s] "
                f"{controller_name} | "
                f"FPS={yolo_data['fps']:.1f} | "
                f"CPU={action['cpu_freq']} | "
                f"GPU={action['gpu_freq']} | "
                f"Speed={action['speed']:.2f} | "
                f"Power={p_total:.2f}W | "
                f"SOC={robot_state['battery']['soc']:.2f}% | "
                f"ExecTime={1000*np.mean(step_times):.3f} ms"
            )

            global_time += control_period
            time.sleep(control_period)

        print(f"Average : {1000*np.mean(step_times):.3f} ms")
        print(f"Min     : {1000*np.min(step_times):.3f} ms")
        print(f"Max     : {1000*np.max(step_times):.3f} ms")

    logger.close()
    player.release()

    print(f"\nFinished {controller_name}")

print("\nAll Experiments Finished")