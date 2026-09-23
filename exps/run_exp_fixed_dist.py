import argparse
import time

import cv2
import numpy as np
import yaml

from controller import (
    MaxPerformanceController,
    AdaptiveController,
    LongTermBatteryAwareController,
    PEOController,
    MultiScaleController,
)
from robot_env import RobotEnv
from video_player import VideoPlayer
from yolo_manager import YOLOManager


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run AsyncMEC experiments at a fixed mission distance.")
    parser.add_argument("--distance", type=float, default=500.0, help="Mission distance in meters.")
    parser.add_argument(
        "--complexity",
        choices=["low", "medium", "high"],
        default="low",
        help="Scene complexity used for the evaluation workload.",
    )
    parser.add_argument("--soh", type=float, default=100.0, help="Initial battery SOH in percent.")
    parser.add_argument(
        "--source-type",
        choices=["video", "camera"],
        default="video",
        help="Input source: a recorded video file or a live camera.",
    )
    parser.add_argument(
        "--video-file",
        default="videos/person_bicycle_car_detection.mp4",
        help="Path to the input video file when using --source-type video.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index when using --source-type camera.",
    )
    return parser.parse_args()


def read_source_frame(source_type, source, cap=None, speed=1.0, control_period=1.0):
    if source_type == "video":
        return source.get_frame(speed=speed, control_period=control_period)
    if source_type == "camera":
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read a frame from the camera source.")
        return frame
    raise ValueError(f"Unsupported source type: {source_type}")


def main():
    args = parse_args()

    complexity_map = {
        "low": 2,
        "medium": 6,
        "high": 10,
    }
    frame_size_map = {
        "low": 416,
        "medium": 640,
        "high": 960,
    }

    mission_distance = args.distance
    soh_list = [args.soh]

    summary_log = open("logs/results_fixed_distance.log", "a")
    summary_log.write("\n=============================================\n")
    summary_log.write(f"Experiment started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary_log.write(f"Distance={mission_distance} m | Complexity={args.complexity} | Source={args.source_type}\n")
    summary_log.write("=============================================\n")

    workload = {
        "name": args.complexity,
        "path": args.video_file if args.source_type == "video" else None,
        "frame_size": frame_size_map[args.complexity],
        "target_fps": 30,
        "complexity": complexity_map[args.complexity],
    }

    if args.source_type == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {args.camera_index}.")
    else:
        cap = None

    for controller_name, ControllerClass in controllers.items():
        print("\n================================================")
        print(f"Running {controller_name}")
        print("================================================")

        for soh in soh_list:
            print(f"\n========== SOH = {soh}% ==========")

            robot = RobotEnv(robot_config, is_sim=False)
            robot.battery.set_initial_soh(soh)

            controller = ControllerClass(robot)
            yolo = YOLOManager(
                is_sim=False,
                robot=robot,
                model_path="./checkpoints/yolov8n.pt",
            )
            yolo.set_enabled(True)

            global_time = 0.0
            distance = 0.0
            qos_count = 0
            total_energy = 0.0
            total_steps = 0
            last_speed = 1.0

            yolo_data = {
                "fps": 0,
                "target_fps": workload["target_fps"],
                "object_count": 0,
                "complexity_id": 1.0,
            }
            print(f"\nRunning workload: {workload['name']} ({args.source_type})")

            if args.source_type == "video":
                player = VideoPlayer(workload["path"], reference_speed=1.0)
                frame_source = player
            else:
                frame_source = None
            yolo.set_scene_complexity(workload["complexity"])
            step_times = []

            while distance < mission_distance:
                frame = read_source_frame(
                    args.source_type,
                    frame_source,
                    cap=cap,
                    speed=last_speed,
                    control_period=control_period,
                )

                start = time.perf_counter()
                action = controller.step(yolo_data)
                last_speed = action["speed"]
                step_times.append(time.perf_counter() - start)

                robot_state = robot.step(action, control_period)

                yolo_data = yolo.run(
                    action["cpu_freq"],
                    action["gpu_freq"],
                    input_data=frame,
                    frame_size=workload["frame_size"],
                )
                if yolo_data["fps"] >= workload["target_fps"]:
                    qos_count += 1

                print(
                    f"FPS={yolo_data['fps']:.2f} "
                    f"Latency={yolo_data['latency_ms']:.2f} ms "
                    f"Objects={yolo_data['object_count']}"
                )

                p_comp = robot_state["p_comp"]
                p_mech = robot_state["p_mech"]
                print(f"Mech power: {p_mech}, Comp power: {p_comp}")
                p_total = p_comp + p_mech
                total_energy += p_total * control_period
                distance += action["speed"] * control_period

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

            if args.source_type == "video":
                frame_source.release()
            else:
                cap.release()
            print(f"\nFinished {controller_name}, SOH: {soh}")

    print("\nAll Experiments Finished")
    summary_log.close()


if __name__ == "__main__":
    main()