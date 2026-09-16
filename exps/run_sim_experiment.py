import yaml
import time
import cv2
from pathlib import Path

from robot_env import RobotEnv
from yolo_manager import YOLOManager

from controller import (
    MaxPerformanceController,
    AdaptiveController,
    LongTermBatteryAwareController,
    # SynchronousMultiScaleController,
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
    "ASMF": MaxPerformanceController,
    # "RRL": AdaptiveController,
    "LTBA": LongTermBatteryAwareController,
    # "SMS": SynchronousMultiScaleController,
    # "OURS": MultiScaleController
}


# ==========================================
# Run experiments
# ==========================================

for controller_name, ControllerClass in controllers.items():

    print("\n================================================")
    print(f"Running {controller_name}")
    print("================================================")

    robot = RobotEnv(robot_config, is_sim=True)

    # ---------------------------------------
    # Real YOLO Example
    # ---------------------------------------
    yolo = YOLOManager(
        is_sim=True,
        robot=robot,
        model_path="./checkpoints/yolov8n.pt")
    yolo.set_enabled(True)

    # ---------------------------------------
    # Controller
    # ---------------------------------------       
    controller = ControllerClass(robot)

    logger = ExperimentLogger(
        f"logs/{controller_name}.csv"
    )

    global_time = 0.0
    distance = 0.0
    peak_current = 0.0

    yolo_data = {
        "fps": 0,
        "object_count": 0
    }

    # ==========================================
    # Mission loop
    # ==========================================

    for phase in mission_config["phases"]:
        print(f"\nStarting phase: {phase['name']}")
        duration = phase["duration"]
        object_count = phase["objects"]
        yolo.set_scene_complexity(object_count)
        steps = int(duration / control_period)

        for _ in range(steps):
            # ==================================
            # Controller
            # ==================================
            action = controller.step(yolo_data)

            # ==================================
            # Robot update
            # ==================================
            robot_state = robot.step(
                action,
                control_period
            )

            # ==================================
            # YOLO workload
            # ==================================
            yolo_data = yolo.run(
                action["cpu_freq"],
                action["gpu_freq"],
            )
            print(
                f"FPS={yolo_data['fps']:.2f} "
                # f"Latency={yolo_data['latency_ms']:.2f} ms "
                f"Objects={yolo_data['object_count']}"
            )

            # ==================================
            # Derived metrics
            # ==================================
            p_comp = robot_state["p_comp"]
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

            # ==================================
            # Health score
            # ==================================

            health_score = (
                0.4 * current +
                0.4 * peak_current +
                0.2 * robot_state["battery"]["temperature"]
            )

            # ==================================
            # Log
            # ==================================

            logger.log([
                # time
                global_time,

                # phase
                phase["name"],

                # actions
                action["cpu_freq"],
                action["gpu_freq"],
                action["speed"],

                # workload
                yolo_data["object_count"],
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
                peak_current,
                robot_state["battery"]["temperature"],
                robot_state["battery"]["soc"],

                # health
                health_score

            ])

            # ==================================
            # Console
            # ==================================

            print(
                f"[{global_time:5.1f}s] "
                f"{controller_name} | "
                f"{phase['name']} | "
                f"Obj={object_count} | "
                f"FPS={yolo_data['fps']:.1f} | "
                f"CPU={action['cpu_freq']} | "
                f"GPU={action['gpu_freq']} | "
                f"Speed={action['speed']:.2f} | "
                f"Power={p_total:.2f}W | "
                f"SOC={robot_state['battery']['soc']:.2f}%"
            )

            global_time += control_period
            time.sleep(control_period)

    logger.close()

    print(f"\nFinished {controller_name}")

print("\nAll Experiments Finished")