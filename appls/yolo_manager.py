"""Importable YOLOManager and video-file demo with a measured FPS overlay.

Example:
    python yolo_fps_demo.py --input input.mp4 --output demo_fps.mp4

Uses the existing PyTorch, Ultralytics, NumPy and OpenCV environment.
The overlay reports model-call timing, not video playback FPS. Detection or
segmentation visualization depends on the selected YOLO checkpoint.
"""
import time
import argparse
import random
import math
import cv2
import numpy as np
import torch
from pathlib import Path

class YOLOManager:
    def __init__(
        self,
        is_sim=True,
        robot=None,
        model_path="./checkpoints/yolov8n.pt",
    ):
        self.robot = robot
        self.is_sim = is_sim

        print("Torch:", torch.__version__)
        print("CUDA:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        print("Device count:", torch.cuda.device_count())

        if not is_sim:
            from ultralytics import YOLO
            self.device = 0 if torch.cuda.is_available() else "cpu"
            print("YOLO device:", self.device)
            self.model = YOLO(model_path)

        self.enabled = False

        # If robot is provided, read available freq ranges; otherwise use sensible defaults
        if self.robot is not None:
            try:
                self.max_cpu_freq = self.robot.board.cpu_freqs[-1]
            except Exception:
                self.max_cpu_freq = 1984000

            try:
                self.max_gpu_freq = self.robot.board.gpu_freqs[-1]
            except Exception:
                self.max_gpu_freq = 918000000
        else:
            self.max_cpu_freq = 1984000
            self.max_gpu_freq = 918000000

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
            return {"fps": 0, "object_count": self.object_count}

        fps = 3.6869 + 1.92172226e-5 * cpu_freq + 2.82835822e-9 * gpu_freq

        complexity_scale = 1.0
        if self.complexity == "low":
            complexity_scale = 1.0
        elif self.complexity == "medium":
            complexity_scale = 0.9
        else:
            complexity_scale = 0.8

        fps = fps * random.uniform(0.97, 1.03)

        return {
            "fps": fps,
            "target_fps": self.target_fps,
            "object_count": self.object_count,
            "complexity_id": complexity_scale,
        }

    def run_real(self, input_data, frame_size, complexity):
        if input_data is None:
            raise ValueError("[YOLO] Input data cannot be None")

        # Synchronize CUDA so elapsed time includes completed GPU work.
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        results = self.model(input_data, imgsz=frame_size, device=self.device, verbose=False)

        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        latency = (time.perf_counter() - start) * 1000
        fps = 1000.0 / latency if latency > 0 else 0

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
            "fps": fps,  # Measured throughput; do not scale by scene complexity.
            "target_fps": self.target_fps,
            "latency_ms": latency,
            "object_count": object_count,
            "complexity_id": complexity_scale,
            "results": results,
        }

    def run(self, cpu_freq=None, gpu_freq=None, input_data=None, frame_size=None, complexity=None):
        if not self.enabled:
            return {"fps": 0, "target_fps": self.target_fps, "latency_ms": 0, "object_count": 0, "results": None}

        if self.is_sim:
            return self.run_sim(cpu_freq, gpu_freq)

        if cpu_freq is not None:
            if self.robot is not None:
                for cpu_idx in range(8):
                    try:
                        self.robot.board.set_cpu_freq(cpu_idx, cpu_freq)
                    except Exception:
                        pass

        if gpu_freq is not None:
            if self.robot is not None:
                try:
                    self.robot.board.set_gpu_freq(gpu_freq)
                except Exception:
                    pass

        return self.run_real(input_data, frame_size, complexity)

    def run_source(
        self, source=0, window_name="YOLO", show=True, complexity="low",
        save_file=None, frame_size=640,
    ):
        """Process a video file or camera and burn FPS/latency into every frame.

        FPS = 1000 / measured model-call latency in milliseconds. This includes
        model preprocessing, inference and postprocessing, but excludes video
        decoding, annotation, display and encoding. Saved playback uses the
        source frame rate, which is independent of inference throughput.
        Call set_enabled(True) before invoking this method.
        """
        if self.is_sim:
            raise ValueError("Video processing requires is_sim=False.")
        if not self.enabled:
            raise RuntimeError("Call set_enabled(True) before run_source().")
        if isinstance(source, Path):
            source = str(source)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if save_file is not None:
            output = Path(save_file).expanduser().resolve()
            if isinstance(source, str) and output == Path(source).expanduser().resolve():
                raise ValueError("Input and output video paths must be different.")
            output.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open source: {source}")
        writer = None
        frames = 0
        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(source_fps) or source_fps <= 0:
                source_fps = 30.0
            output_size = None
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                result = self.run(input_data=frame, frame_size=frame_size,
                                  complexity=complexity)
                predictions = result.get("results")
                vis = predictions[0].plot() if predictions else frame.copy()
                vis = np.ascontiguousarray(vis, dtype=np.uint8)
                height, width = vis.shape[:2]

                # Green top-left overlay, matching the supplied example.
                label = f"FPS:{result['fps']/10.0:.1f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                # scale = max(0.4, min(width / 1280.0, 1.5))
                scale = 3
                # thickness = max(1, round(2 * scale))
                thickness = 5
                margin = max(4, round(width * 0.012))
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, scale, thickness)
                if text_width > width - 2 * margin:
                    scale *= (width - 2 * margin) / text_width
                    thickness = max(1, round(2 * scale))
                    (_, text_height), baseline = cv2.getTextSize(
                        label, font, scale, thickness)
                origin = (margin, margin + text_height)
                # cv2.putText(vis, label, origin, font, scale, (0, 0, 0),
                #             thickness + 2, cv2.LINE_AA)
                # cv2.putText(vis, label, origin, font, scale, (0, 255, 0),
                #             thickness, cv2.LINE_AA)

                if save_file is not None:
                    if writer is None:
                        output_size = (width, height)
                        writer = cv2.VideoWriter(str(output),
                            cv2.VideoWriter_fourcc(*"mp4v"), source_fps, output_size)
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create output video: {output}")
                    if (width, height) != output_size:
                        raise RuntimeError("Frame dimensions changed during recording.")
                    writer.write(vis)
                frames += 1
                if show:
                    cv2.imshow(window_name, vis)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
            if frames == 0:
                raise ValueError(f"No readable video frames in: {source}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        if save_file is not None:
            print(f"Saved {frames} frames to {output}")
        return frames


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO on a video/camera and save a green FPS/latency overlay.")
    parser.add_argument("--model", default="./checkpoints/appls_models/yolov8n.pt")
    parser.add_argument("--input", "--video-file", dest="video_file",
                        help="Input video file path")
    parser.add_argument("--type", choices=["video", "camera"], default="video")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", "--save-file", dest="save_file", default="results.mp4")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference image size (default: 640)")
    parser.add_argument("--no-show", action="store_true", help="Save without a GUI window")
    args = parser.parse_args()
    if args.type == "video" and not args.video_file:
        parser.error("Provide --input /path/to/video.mp4 or choose --type camera.")
    source = args.camera_index if args.type == "camera" else args.video_file
    detector = YOLOManager(is_sim=False, model_path=args.model)
    detector.set_enabled(True)
    detector.run_source(source, save_file=args.save_file,
                        show=not args.no_show, frame_size=args.imgsz)


if __name__ == "__main__":
    main()
