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
    # "ASMF": MaxPerformanceController,
    # "RRL": PEOController,
    # "LTBA": LongTermBatteryAwareController,
    # "SMS": SynchronousMultiScaleController,
    "OURS": MultiScaleController
}

MISSION_DISTANCE = 500.0      # meters
SOH_LIST = [100]

summary_log = open("logs/results_fixed_distance.log", "a")
summary_log.write("\n=============================================\n")
summary_log.write(f"Experiment started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
summary_log.write("=============================================\n")


# ==========================================
# Run experiments
# ==========================================

for controller_name, ControllerClass in controllers.items():

    print("\n================================================")
    print(f"Running {controller_name}")
    print("================================================")

    for soh in SOH_LIST:

        print(f"\n========== SOH = {soh}% ==========")

        # ==========================================
        # Mission loop
        # ==========================================
        test_videos = [
            {
                "name": "low",
                "path": "videos/person_bicycle_car_detection.mp4",
                "frame_size": 416,
                "target_fps": 30,
                "complexity": 2
            },
            # {
            #     "name": "medium",
            #     "path": "videos/fruit_and_vegetable_detection.mp4.mp4",
            #     "frame_size": 640,
            #     "target_fps": 30,
            #     "complexity": 6
            # },
            # {
            #     "name": "high",
            #     "path": "videos/person_bicycle_car_detection.mp4",
            #     "frame_size": 960,
            #     "target_fps": 30,
            #     "complexity": 10
            # }
        ]

        for workload in test_videos:

            robot = RobotEnv(robot_config, is_sim=False)
            robot.battery.set_initial_soh(soh)

            controller = ControllerClass(robot)

            yolo = YOLOManager(
                is_sim=False,
                robot=robot,
                model_path="./checkpoints/yolov8n.pt")
            yolo.set_enabled(True)

            global_time = 0.0
            distance = 0.0
            qos_count = 0
            total_energy = 0.0
            total_steps = 0
            last_speed = 1.0
            final_soh=0.0

            yolo_data = {
                "fps": 0,
                "target_fps": workload["target_fps"],
                "object_count": 0,
                "complexity_id": 1.0
            }
            print(f"\nRunning workload: {workload['name']}")

            player = VideoPlayer(workload["path"], reference_speed=1.0)
            yolo.set_scene_complexity(workload["complexity"])

            step_times = []

            while distance < MISSION_DISTANCE:
                
                frame = player.get_frame(speed=last_speed, control_period=control_period)
                
                # ==================================
                # Controller
                # ==================================
                start = time.perf_counter()
                action = controller.step(yolo_data)
                last_speed = action["speed"]
                step_times.append(time.perf_counter() - start)

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
                    input_data=frame,
                    frame_size=workload["frame_size"]
                )
                if yolo_data["fps"] >= workload["target_fps"]:
                    qos_count += 1
                print(
                    f"FPS={yolo_data['fps']:.2f} "
                    f"Latency={yolo_data['latency_ms']:.2f} ms "
                    f"Objects={yolo_data['object_count']}"
                )

                # ==================================
                # Derived metrics
                # ==================================
                p_comp = robot_state["p_comp"]
                p_mech = robot_state["p_mech"]
                print(f"Mech power: {p_mech}, Comp power: {p_comp}")
                p_total = p_comp + p_mech
                total_energy += p_total * control_period

                current = robot_state["battery"]["current"]
                distance += action["speed"] * control_period

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
                    f"SOC={robot_state['battery']['soc'] * 100:.2f}% | "
                    f"ExecTime={1000*np.mean(step_times):.3f} ms"
                )

                global_time += control_period
                time.sleep(control_period)
                total_steps += 1

            print(f"Average : {1000*np.mean(step_times):.3f} ms")
            print(f"Min     : {1000*np.min(step_times):.3f} ms")
            print(f"Max     : {1000*np.max(step_times):.3f} ms")

            mission_time = global_time
            eng_dist = total_energy / distance
            qos_percentage = 100.0 * qos_count / total_steps

            print("\n==============================")
            print(f"Controller : {controller_name}")
            print(f"SOH        : {soh: 8f}%")
            print(f"Workload   : {workload['name']}")
            print(f"Distance   : {distance:.1f} m")
            print(f"QoS        : {qos_percentage:.2f}%")
            print(f"Eng./Dist  : {eng_dist:.2f} J/m")
            print(f"Time       : {mission_time:.1f} s")
            print("==============================")

            summary_log.write(
                f"{controller_name:8s} | "
                f"SOH={soh:3d}% | "
                f"Workload={workload['name']:6s} | "
                f"QoS={qos_percentage:6.2f}% | "
                f"Eng./Dist={eng_dist:7.2f} J/m | "
                f"Time={mission_time:7.2f} s\n"
            )
            summary_log.flush()

        player.release()
        print(f"\nFinished {controller_name}, SOH: {soh}")
        

print("\nAll Experiments Finished")
summary_log.close()