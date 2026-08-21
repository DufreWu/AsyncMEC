#!/usr/bin/env python3
"""Show a ROS 2 image topic in a top-left window and optionally record MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/leatherback/front_camera/rgb")
    parser.add_argument("--output", type=Path, help="MP4 output path; omit to preview only.")
    parser.add_argument("--fps", type=float, default=20.0, help="Recorded video frame rate.")
    parser.add_argument("--width", type=int, default=320, help="Preview window width.")
    parser.add_argument("--height", type=int, default=240, help="Preview window height.")
    parser.add_argument("--x", type=int, default=0, help="Preview window left coordinate.")
    parser.add_argument("--y", type=int, default=0, help="Preview window top coordinate.")
    return parser.parse_args()


class CameraViewRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("camera_view_recorder")
        self.args = args
        self.writer: cv2.VideoWriter | None = None
        self.window = "Leatherback front camera (Q/Esc to quit)"

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, args.width, args.height)
        cv2.moveWindow(self.window, args.x, args.y)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(Image, args.topic, self.on_image, qos)
        self.get_logger().info(f"Viewing {args.topic} at ({args.x}, {args.y})")

    def on_image(self, message: Image) -> None:
        frame = image_to_bgr(message)
        cv2.imshow(self.window, frame)

        if self.args.output is not None:
            if self.writer is None:
                self.args.output.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.args.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.args.fps,
                    (width, height),
                )
                if not self.writer.isOpened():
                    raise RuntimeError(f"Could not open video output: {self.args.output}")
                self.get_logger().info(f"Recording {width}x{height} MP4 to {self.args.output}")
            self.writer.write(frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            rclpy.shutdown()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()


def image_to_bgr(message: Image) -> np.ndarray:
    """Convert common 8-bit ROS Image encodings to an OpenCV BGR frame."""
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8UC1": 1,
        "8UC3": 3,
        "8UC4": 4,
    }
    encoding = message.encoding
    if encoding not in channels_by_encoding:
        raise ValueError(
            f"Unsupported image encoding {encoding!r}; expected an 8-bit RGB, BGR, RGBA, BGRA, or mono image."
        )

    channels = channels_by_encoding[encoding]
    row_bytes = int(message.width) * channels
    if int(message.step) < row_bytes:
        raise ValueError(f"Invalid image step {message.step}; expected at least {row_bytes} bytes.")

    # Respect row padding indicated by sensor_msgs/Image.step.
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected_bytes = int(message.step) * int(message.height)
    if raw.size < expected_bytes:
        raise ValueError(f"Incomplete image data: received {raw.size} bytes, expected {expected_bytes}.")
    rows = raw[:expected_bytes].reshape(int(message.height), int(message.step))
    pixels = rows[:, :row_bytes].reshape(int(message.height), int(message.width), channels)

    if encoding == "rgb8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if encoding in ("bgra8", "8UC4"):
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if encoding in ("mono8", "8UC1"):
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    return pixels


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = CameraViewRecorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
