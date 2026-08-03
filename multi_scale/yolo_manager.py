import time
import random
import cv2
import numpy as np
import torch
from robot_env import RobotEnv

class YOLOManager:
    def __init__(
            self, 
            is_sim=True,
            robot=None,
            model_path="./checkpoints/yolov8n.pt"):

        self.robot = robot
        self.is_sim = is_sim

        print("Torch:", torch.__version__)
        print("CUDA:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        print("Device count:", torch.cuda.device_count())

        if not is_sim:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.device = 0 if torch.cuda.is_available() else "cpu"
            print("YOLO device:", self.device)

        self.enabled = False

        self.max_cpu_freq = self.robot.board.cpu_freqs[-1]
        self.max_gpu_freq = self.robot.board.gpu_freqs[-1]

        # reference FPS at max frequency
        self.max_fps = 45
        self.target_fps = 30

        self.object_count = 2
        self.complexity = "low"

    def set_enabled(self, flag):
        self.enabled = flag

    def set_scene_complexity(self, complexity):
        self.complexity = complexity

    def run_sim(self, cpu_freq, gpu_freq, frame=None):
        if not self.enabled:
            return {
                "fps": 0,
                "object_count": self.object_count
            }

        fps = (
            3.6869
            + 1.92172226e-5 * cpu_freq
            + 2.82835822e-9 * gpu_freq
        )

        complexity_scale = 1.0
        if self.complexity == "low":
            complexity_scale = 1.0
        elif self.complexity == "medium":
            complexity_scale = 0.9
        else:
            complexity_scale = 0.8

        fps = fps * random.uniform(0.97 , 1.03)

        return {
            "fps": fps,
            "target_fps": self.target_fps,
            "object_count": self.object_count,
            "complexity_id": complexity_scale,
        }

    def run_real(self, input_data, frame_size, complexity):

        if input_data is None:
            raise ValueError("[YOLO] Input data cannot be None")

        start = time.perf_counter()
   
        results = self.model(
                input_data,
                imgsz=frame_size,
                device=self.device,
                verbose=False
            )

        latency = (
            time.perf_counter() - start) * 1000

        fps = 1000.0 / latency

        if len(results) > 0:
            object_count = len(results[0].boxes)
        else:
            object_count = 0
        
        complexity_scale = 1.0
        if complexity == "low":
            complexity_scale = 1.0
        elif complexity == "medium":
            complexity_scale = 0.9
        else:
            complexity_scale = 0.8

        return {
            "fps": fps * complexity_scale,
            "target_fps": self.target_fps,
            "latency_ms": latency,
            "object_count": object_count,
            "complexity_id": complexity_scale,
            "results": results
        }

    def run(self, cpu_freq=None, gpu_freq=None, input_data=None, frame_size=None, complexity=None):

        if not self.enabled:
            return {
                "fps": 0,
                "target_fps": self.target_fps,
                "latency_ms": 0,
                "object_count": 0,
                "results": None
            }

        if self.is_sim:
            return self.run_sim(cpu_freq, gpu_freq)
        
        if cpu_freq is not None:
            for cpu_idx in range(8):
                self.robot.board.set_cpu_freq(cpu_idx, cpu_freq)

        if gpu_freq is not None:
            self.robot.board.set_gpu_freq(gpu_freq)
        
        return self.run_real(input_data, frame_size, complexity)

# ---------------------------------------
# Demo
# ---------------------------------------
if __name__ == "__main__":
    # ---------------------------------------
    # Simulation Example
    # ---------------------------------------
    try:
        import yaml
        from pathlib import Path

        with open("configs/robot.yaml", "r") as f:
            robot_config = yaml.safe_load(f)

        robot = RobotEnv(robot_config, is_sim=False)
    except Exception:
        robot = None

    print("=" * 60)
    print("Test")
    print("=" * 60)

    detector = YOLOManager(
        is_sim=False,
        robot=robot,
        model_path="./checkpoints/yolov8n.pt")
    detector.set_enabled(True)

    # sim = YOLOManager(
    #     is_sim=False, computing_board=computing_board)

    # sim.set_scene_complexity(5)

    # result = sim.run(
    #     cpu_freq=1984000,
    #     gpu_freq=918000000)

    # print(result)

    # ---------------------------------------
    # Real YOLO Example
    # ---------------------------------------
    video = "person-bicycle-car-detection.mp4"

    if not Path(video).exists():
        raise FileNotFoundError(video)

    cap = cv2.VideoCapture(video)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detector.run(
            cpu_freq=robot.board.cpu_freqs[0],
            gpu_freq=robot.board.gpu_freqs[0],
            input_data=frame
        )

        # print(
        #     f"FPS={result['fps']:.2f} "
        #     f"Latency={result['latency_ms']:.2f} ms "
        #     f"Objects={result['object_count']}"
        # )

        # cv2.imshow(
        #     "YOLO",
        #     result["results"][0].plot()
        # )

        # if cv2.waitKey(1) == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
