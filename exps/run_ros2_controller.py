#!/usr/bin/env python3
"""Run camera -> YOLO -> controller -> ROS speed commands for Isaac Sim hybrid mode."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = {
    'multiscale': 'MultiScaleController',
    'adaptive': 'AdaptiveController',
    'maxperf': 'MaxPerformanceController',
    'peo': 'PEOController',
    'ltba': 'LongTermBatteryAwareController',
    'rrl': 'RRLController',
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', choices=CONTROLLERS, default='maxperf')
    p.add_argument('--robot-config', type=Path, default=ROOT / 'configs/robot.yaml')
    p.add_argument('--model', type=Path, default=ROOT / 'checkpoints/yolov8n.pt')
    p.add_argument('--checkpoint', type=Path, help='Optional MultiScale controller checkpoint.')
    p.add_argument('--device', default='cpu', help='Controller inference device; YOLO selects CUDA if available.')
    p.add_argument('--image-size', type=int, default=640)
    p.add_argument('--target-fps', type=float, default=30)
    p.add_argument('--control-hz', type=float, default=1)
    p.add_argument('--input-timeout', type=float, default=2)
    p.add_argument('--max-speed', type=float, default=1)
    p.add_argument('--camera-topic', default='/front_camera/rgb')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--zone-topic', default='/asyncmec/scene_zone')
    p.add_argument('--speed-topic', default='/asyncmec/speed_limit')
    p.add_argument('--complexity', choices=['low', 'medium', 'high'],
                   help='Fixed override; otherwise use the simulator scene-zone topic.')
    p.add_argument('--apply-dvfs', action='store_true', help='Apply CPU/GPU settings to Jetson hardware.')
    args, ros_args = p.parse_known_args(argv)
    for name in ('target_fps', 'control_hz', 'input_timeout', 'max_speed'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f'--{name.replace("_", "-")} must be finite and positive')
    if args.image_size <= 0:
        p.error('--image-size must be positive')
    if args.checkpoint and args.controller != 'multiscale':
        p.error('--checkpoint is supported only for --controller multiscale')
    return args, ros_args


def control_cycle(args, robot, controller, yolo, publisher, frame, inputs, dt):
    """Evaluate one fresh snapshot; never apply actions from expired inputs."""
    def require_fresh():
        now = time.monotonic()
        if frame is None or now - frame.received_at > args.input_timeout:
            raise ValueError('Missing or stale camera frame')
        for key in ('odom',) + (() if args.complexity else ('zone',)):
            if key not in inputs or now - inputs[key][1] > args.input_timeout:
                raise ValueError(f'Missing or stale {key}')

    require_fresh()
    complexity = args.complexity or inputs['zone'][0]
    if complexity not in ('low', 'medium', 'high'):
        raise ValueError(f'Invalid scene complexity: {complexity}')
    speed = abs(float(inputs['odom'][0]))
    if not math.isfinite(speed):
        raise ValueError('Invalid odometry speed')
    robot.set_speed(speed)
    robot.update_battery(dt)
    state = yolo.run(input_data=frame.image, frame_size=args.image_size, complexity=complexity)
    action = dict(controller.step(state))
    speed = float(action['speed'])
    if not math.isfinite(speed) or speed < 0:
        raise ValueError('Invalid controller speed')
    action['speed'] = min(speed, args.max_speed)
    for key, allowed in [('cpu_freq', robot.board.cpu_freqs), ('gpu_freq', robot.board.gpu_freqs)]:
        if action[key] not in allowed:
            raise ValueError(f'Invalid controller {key}')
    require_fresh()
    # RobotEnv updates frequency state without hardware writes unless --apply-dvfs.
    for cpu in range(len(robot.fc_list)):
        robot.set_board_cpu_dvfs(cpu, action['cpu_freq'])
    robot.set_board_gpu_dvfs(action['gpu_freq'])
    require_fresh()
    publisher.publish_action(action)
    return state, action, complexity


def main(argv=None):
    args, ros_args = parse_args(argv)
    import yaml
    import rclpy
    from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from multi_scale.envs.robot_env import RobotEnv
    from multi_scale.appls.yolo_manager import YOLOManager
    from multi_scale.appls.ros2_camera_node import CameraSubscriber
    from multi_scale.methods.ros2_speed_node import SpeedPublisher
    from multi_scale.methods import controller as controller_module

    with args.robot_config.expanduser().open() as stream:
        cfg = yaml.safe_load(stream)
    for key, value in cfg['battery'].items():
        if isinstance(value, str) and (key.endswith('_path') or key.endswith('_scaler')):
            path = Path(value).expanduser()
            cfg['battery'][key] = str(path if path.is_absolute() else ROOT / path)
    cfg['motor']['v_min'] = 0.0
    robot = RobotEnv(cfg, is_sim=not args.apply_dvfs)
    yolo = YOLOManager(is_sim=False, robot=robot, model_path=str(args.model.expanduser()))
    yolo.target_fps = args.target_fps
    yolo.set_enabled(True)
    kwargs = {'device': args.device} if args.controller in ('multiscale', 'rrl') else {}
    if args.checkpoint:
        kwargs['checkpoint_path'] = str(args.checkpoint.expanduser())
    controller = getattr(controller_module, CONTROLLERS[args.controller])(robot, **kwargs)

    rclpy.init(args=ros_args)
    camera = publisher = executor = worker = None
    inputs, lock = {}, threading.Lock()

    def store(key, value):
        with lock:
            inputs[key] = (value, time.monotonic())

    try:
        camera = CameraSubscriber(args.camera_topic, args.input_timeout)
        publisher = SpeedPublisher(args.speed_topic, args.max_speed)
        publisher.create_subscription(Odometry, args.odom_topic,
                                      lambda msg: store('odom', msg.twist.twist.linear.x), qos_profile_sensor_data)
        publisher.create_subscription(String, args.zone_topic, lambda msg: store('zone', msg.data), 1)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(camera)
        executor.add_node(publisher)

        def spin():
            try:
                executor.spin()
            except ExternalShutdownException:
                pass

        worker = threading.Thread(target=spin, daemon=True)
        worker.start()
        publisher.get_logger().info(
            f'Controller={args.controller}; hardware DVFS={args.apply_dvfs}; battery=modeled'
        )
        previous = time.monotonic()
        while rclpy.ok():
            started = time.monotonic()
            dt = started - previous
            previous = started
            with lock:
                snapshot = dict(inputs)
            try:
                state, action, zone = control_cycle(
                    args, robot, controller, yolo, publisher, camera.get_frame(), snapshot, dt)
                publisher.get_logger().info(
                    f'{zone}: FPS={state["fps"]:.1f} objects={state["object_count"]} '
                    f'speed={action["speed"]:.2f} CPU={action["cpu_freq"]} GPU={action["gpu_freq"]}')
            except Exception as exc:
                if not rclpy.ok():
                    break
                publisher.stop()
                publisher.get_logger().warning(f'Stopped: {exc}')
            time.sleep(max(0, 1 / args.control_hz - (time.monotonic() - started)))
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if publisher is not None and rclpy.ok():
            publisher.stop()
        if executor is not None:
            executor.shutdown()
        if worker is not None:
            worker.join(timeout=2)
        for node in (camera, publisher):
            if node is not None:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
