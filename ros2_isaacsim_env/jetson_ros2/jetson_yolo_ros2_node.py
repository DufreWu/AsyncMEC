#!/usr/bin/env python3
"""YOLOv8 ROS 2 perception node for the AsyncMEC Isaac Sim--Jetson HIL demo."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32, String
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model or local TensorRT engine.")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--input-topic", default="/front_camera/rgb")
    parser.add_argument("--annotated-topic", default="/yolo/annotated")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-rate", type=float, default=0.0,
                        help="Maximum inference rate [Hz]; 0 processes every received frame.")
    parser.add_argument("--fps-alpha", type=float, default=0.2, help="EMA coefficient for reported FPS.")
    parser.add_argument("--low-max", type=int, default=5)
    parser.add_argument("--medium-max", type=int, default=9)
    args, ros_args = parser.parse_known_args()
    return args, ros_args


class YoloRosNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("asyncmec_yolo")
        self.args = args
        self.model = YOLO(args.model)
        self.last_inference_start = 0.0
        self.fps_ema = 0.0
        self.frame_count = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, args.input_topic, self.image_callback, sensor_qos)
        self.annotated_pub = self.create_publisher(Image, args.annotated_topic, sensor_qos)
        self.fps_pub = self.create_publisher(Float32, "/yolo/fps", 10)
        self.latency_pub = self.create_publisher(Float32, "/yolo/latency_ms", 10)
        self.count_pub = self.create_publisher(Int32, "/yolo/object_count", 10)
        self.workload_pub = self.create_publisher(String, "/yolo/workload", 10)
        self.detections_pub = self.create_publisher(String, "/yolo/detections", 10)

        self.get_logger().info(
            f"YOLO ready: model={args.model}, device={args.device}, input={args.input_topic}"
        )

    def workload_level(self, count: int) -> str:
        if count <= self.args.low_max:
            return "low"
        if count <= self.args.medium_max:
            return "medium"
        return "high"

    @staticmethod
    def image_message_to_bgr(message: Image) -> np.ndarray:
        """Convert common 8-bit ROS encodings without the binary cv_bridge module."""
        encoding = message.encoding.lower()
        channels_by_encoding = {
            "mono8": 1,
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
        }
        if encoding not in channels_by_encoding:
            raise ValueError(
                f"Unsupported image encoding '{message.encoding}'. "
                f"Supported: {sorted(channels_by_encoding)}"
            )

        channels = channels_by_encoding[encoding]
        packed_row_bytes = int(message.width) * channels
        step = int(message.step)
        if step < packed_row_bytes:
            raise ValueError(f"Invalid Image.step={step}; expected at least {packed_row_bytes}")

        raw = np.frombuffer(message.data, dtype=np.uint8)
        expected_bytes = int(message.height) * step
        if raw.size < expected_bytes:
            raise ValueError(f"Image data has {raw.size} bytes; expected at least {expected_bytes}")

        # Respect ROS row padding before reshaping into pixels.
        pixels = raw[:expected_bytes].reshape(int(message.height), step)
        pixels = pixels[:, :packed_row_bytes].reshape(int(message.height), int(message.width), channels)

        if encoding == "bgr8":
            return pixels.copy()
        if encoding == "rgb8":
            return pixels[:, :, ::-1].copy()
        if encoding == "bgra8":
            return pixels[:, :, :3].copy()
        if encoding == "rgba8":
            return pixels[:, :, :3][:, :, ::-1].copy()
        return np.repeat(pixels, 3, axis=2)

    @staticmethod
    def bgr_to_image_message(frame_bgr: np.ndarray, source: Image) -> Image:
        frame = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, received shape={frame.shape}")
        output = Image()
        output.header = source.header
        output.height = int(frame.shape[0])
        output.width = int(frame.shape[1])
        output.encoding = "bgr8"
        output.is_bigendian = 0
        output.step = int(frame.shape[1] * 3)
        output.data = frame.tobytes()
        return output

    def image_callback(self, message: Image) -> None:
        now = time.perf_counter()
        if self.args.max_rate > 0.0:
            minimum_period = 1.0 / self.args.max_rate
            if now - self.last_inference_start < minimum_period:
                return
        self.last_inference_start = now

        try:
            frame_bgr = self.image_message_to_bgr(message)
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        inference_start = time.perf_counter()
        results = self.model.predict(
            source=frame_bgr,
            device=self.args.device,
            conf=self.args.confidence,
            iou=self.args.iou,
            imgsz=self.args.imgsz,
            verbose=False,
        )
        latency_s = time.perf_counter() - inference_start
        instantaneous_fps = 1.0 / max(latency_s, 1.0e-9)
        alpha = min(1.0, max(0.0, self.args.fps_alpha))
        self.fps_ema = instantaneous_fps if self.frame_count == 0 else (
            alpha * instantaneous_fps + (1.0 - alpha) * self.fps_ema
        )
        self.frame_count += 1

        result = results[0]
        boxes = result.boxes
        object_count = 0 if boxes is None else len(boxes)
        workload = self.workload_level(object_count)

        detections = []
        if boxes is not None:
            names = result.names
            for xyxy, confidence, class_id in zip(
                boxes.xyxy.detach().cpu().tolist(),
                boxes.conf.detach().cpu().tolist(),
                boxes.cls.detach().cpu().tolist(),
            ):
                class_index = int(class_id)
                detections.append(
                    {
                        "class_id": class_index,
                        "class_name": str(names[class_index]),
                        "confidence": round(float(confidence), 4),
                        "xyxy": [round(float(value), 1) for value in xyxy],
                    }
                )

        annotated = result.plot()
        annotated_message = self.bgr_to_image_message(annotated, message)
        self.annotated_pub.publish(annotated_message)

        fps_message = Float32(data=float(self.fps_ema))
        latency_message = Float32(data=float(latency_s * 1000.0))
        count_message = Int32(data=int(object_count))
        workload_message = String(data=workload)
        detection_message = String(
            data=json.dumps(
                {
                    "stamp": {"sec": message.header.stamp.sec, "nanosec": message.header.stamp.nanosec},
                    "count": object_count,
                    "workload": workload,
                    "fps": round(self.fps_ema, 3),
                    "latency_ms": round(latency_s * 1000.0, 3),
                    "detections": detections,
                },
                separators=(",", ":"),
            )
        )
        self.fps_pub.publish(fps_message)
        self.latency_pub.publish(latency_message)
        self.count_pub.publish(count_message)
        self.workload_pub.publish(workload_message)
        self.detections_pub.publish(detection_message)


def main() -> None:
    args, ros_args = build_parser()
    rclpy.init(args=ros_args)
    node = YoloRosNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()