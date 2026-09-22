#!/usr/bin/env python3
"""Publish controller speed commands to Isaac Sim's hybrid speed-limit topic.

Standalone test (source ROS 2 first):
    python -m multi_scale.methods.ros2_speed_node --speed 0.5

In a controller loop, initialize rclpy, create SpeedPublisher, and call
node.publish_action(controller.step(state)) once per control cycle. Publish at
least once within Isaac Sim's --speed-limit-timeout (default 2.5 seconds).
"""
from __future__ import annotations

import argparse
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32


def validated_speed(value, max_speed):
    """Reject invalid commands and cap forward speed to the simulator limit."""
    speed = float(value)
    if not math.isfinite(speed) or speed < 0:
        raise ValueError('Speed must be finite and nonnegative')
    return min(speed, max_speed)


class SpeedPublisher(Node):
    """Publish a methods/controller.py action without changing CPU/GPU settings."""

    def __init__(self, topic='/asyncmec/speed_limit', max_speed=1.0):
        max_speed = float(max_speed)
        if not math.isfinite(max_speed) or max_speed <= 0:
            raise ValueError('max_speed must be finite and positive')
        super().__init__('controller_speed_publisher')
        self.max_speed = max_speed
        self.speed_pub = self.create_publisher(Float32, topic, 1)
        self.get_logger().info(f'Publishing speed to {topic}; limit={max_speed:g} m/s')

    def publish_speed(self, speed):
        """Publish immediately; an invalid command sends a stop before raising."""
        try:
            value = validated_speed(speed, self.max_speed)
        except (TypeError, ValueError, OverflowError):
            self.stop()
            raise
        self.speed_pub.publish(Float32(data=value))
        return value

    def publish_action(self, action):
        """Accept the {'speed', 'cpu_freq', 'gpu_freq'} controller output."""
        try:
            speed = action['speed']
        except (KeyError, TypeError):
            self.stop()
            raise
        return self.publish_speed(speed)

    def stop(self):
        self.speed_pub.publish(Float32(data=0.0))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed', type=float, required=True, help='Fixed test speed [m/s].')
    parser.add_argument('--topic', default='/asyncmec/speed_limit')
    parser.add_argument('--rate', type=float, default=1.0, help='Publish frequency [Hz].')
    parser.add_argument('--max-speed', type=float, default=1.0, help='Match the simulator speed limit [m/s].')
    args, ros_args = parser.parse_known_args(argv)
    for key in ('rate', 'max_speed'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be finite and positive')
    try:
        validated_speed(args.speed, args.max_speed)
    except ValueError as exc:
        parser.error(str(exc))
    return args, ros_args


def main(argv=None):
    args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = SpeedPublisher(args.topic, args.max_speed)
        node.create_timer(1.0 / args.rate, lambda: node.publish_speed(args.speed))
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
