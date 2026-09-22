#!/usr/bin/env python3
"""Receive Isaac Sim RGB frames as BGR NumPy arrays for OpenCV/YOLO.

    python -m multi_scale.appls.ros2_camera_node --show

Spin the node with rclpy.spin_once() or an executor, then call get_frame().
Only the latest frame is kept, so slow inference cannot build up a frame queue.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray  # BGR uint8, H x W x 3
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    received_at: float  # Local monotonic time; used for freshness checks.


class CameraSubscriber(Node):
    def __init__(self, topic='/front_camera/rgb', max_age=2.0):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError('max_age must be finite and positive')
        super().__init__('controller_camera_subscriber')
        self.max_age = max_age
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None
        self.received_count = 0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(Image, topic, self._on_image, qos)
        self.get_logger().info(f'Receiving camera frames from {topic} as BGR8')

    def _on_image(self, message):
        received_at = time.monotonic()
        try:
            # CvBridge handles RGB/BGR ordering and padded ROS image rows.
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
                raise ValueError('Expected a nonempty H x W x 3 image')
            frame = CameraFrame(
                image.copy(), message.header.stamp.sec, message.header.stamp.nanosec,
                message.header.frame_id, received_at,
            )
        except (CvBridgeError, ValueError, TypeError) as exc:
            self.get_logger().warning(f'Cannot decode camera frame: {exc}')
            return
        with self._lock:
            self._latest = frame
            self.received_count += 1

    def get_frame(self):
        """Consume the latest fresh frame, or return None; each frame is returned once.

        The caller owns the returned array and may annotate it. Spin this node
        regularly (or use a background executor) to receive new frames.
        """
        with self._lock:
            frame = self._latest
            self._latest = None
        if frame is None or time.monotonic() - frame.received_at > self.max_age:
            return None
        return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/front_camera/rgb')
    parser.add_argument('--max-age', type=float, default=2.0, help='Maximum local frame age [s].')
    parser.add_argument('--show', action='store_true', help='Display frames; press Q or Escape to exit.')
    args, ros_args = parser.parse_known_args(argv)
    if not math.isfinite(args.max_age) or args.max_age <= 0:
        parser.error('--max-age must be finite and positive')
    return args, ros_args


def main(argv=None):
    args, ros_args = parse_args(argv)
    cv2 = None
    if args.show:
        import cv2
    rclpy.init(args=ros_args)
    node = None
    try:
        node = CameraSubscriber(args.topic, args.max_age)
        last_report = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            frame = node.get_frame()
            if frame is not None:
                now = time.monotonic()
                if now - last_report >= 1.0:
                    height, width = frame.image.shape[:2]
                    node.get_logger().info(
                        f'Received {node.received_count} frames; {width}x{height}; '
                        f'stamp={frame.stamp_sec}.{frame.stamp_nanosec:09d}'
                    )
                    last_report = now
                if cv2 is not None:
                    cv2.imshow('Isaac Sim camera', frame.image)
            if cv2 is not None and cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
