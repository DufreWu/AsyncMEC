#!/usr/bin/env python3
"""Isaac Sim 6 PC-side HIL server for Leatherback and Jetson perception.

This is a standalone NVIDIA Isaac Sim 6 application. It does not depend on
Isaac Lab. Run it with Isaac Sim's ``python.sh`` (or with the Python environment
of a pip-installed Isaac Sim 6).

Features
--------
* NVIDIA Leatherback USD from the Isaac Sim asset server.
* Low/medium/high randomized cube complexity.
* Green navigation goal.
* Optional autonomous goal follower with reactive cube avoidance.
* Native Isaac Sim 6 Ackermann controller for Leatherback steering/wheels.
* Front RTX RGB + depth camera attached to Leatherback.
* ROS 2 RGB, depth (optional), and CameraInfo publishing.
* ROS 2 ``ackermann_msgs/AckermannDriveStamped`` command subscriber.
* ROS 2 clock, odometry, and TF publishing for Nav2/Jetson integration.
* Passive suspension settling, command timeout, acceleration limiting, and
  non-finite pose protection.

The autonomous avoidance intentionally uses simulator-known obstacle geometry.
YOLO runs independently on the Jetson for perception/FPS measurements. The
default ``ros`` mode accepts commands on ``/ackermann_cmd``.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without the interactive GUI.")
    parser.add_argument("--complexity", choices=("low", "medium", "high", "random"), default="medium")
    parser.add_argument("--num-obstacles", type=int, default=None, help="Exact cube count; overrides --complexity.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control-mode", choices=("auto", "ros"), default="ros")
    parser.add_argument("--ackermann-topic", default="/ackermann_cmd")
    parser.add_argument("--rgb-topic", default="/front_camera/rgb")
    parser.add_argument("--depth-topic", default="/front_camera/depth")
    parser.add_argument("--camera-info-topic", default="/front_camera/camera_info")
    parser.add_argument("--camera-frame", default="front_camera_link")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument("--camera-x", type=float, default=0.25, help="Camera local X offset [m].")
    parser.add_argument("--camera-z", type=float, default=0.25, help="Camera local Z offset [m].")
    parser.add_argument("--sim-hz", type=float, default=60.0)
    parser.add_argument("--device", default="cpu", help="Physics device, e.g. cpu or cuda:0.")
    parser.add_argument("--wheel-base", type=float, default=0.32)
    parser.add_argument("--track-width", type=float, default=0.24)
    parser.add_argument("--wheel-radius", type=float, default=0.052)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--max-steer", type=float, default=0.7854)
    parser.add_argument("--max-acceleration", type=float, default=1.0)
    parser.add_argument("--max-deceleration", type=float, default=2.0)
    parser.add_argument("--command-timeout", type=float, default=0.5)
    parser.add_argument("--settling-frames", type=int, default=60)
    parser.add_argument("--goal-tolerance", type=float, default=0.65)
    parser.add_argument("--lookahead", type=float, default=3.0)
    parser.add_argument("--corridor-half-width", type=float, default=1.15)
    parser.add_argument("--world-half-length", type=float, default=10.0)
    parser.add_argument("--world-half-width", type=float, default=5.5)
    parser.add_argument("--disable-depth", action="store_true", help="Do not render or publish depth.")
    parser.add_argument("--no-ros", action="store_true", help="Do not load the Isaac Sim ROS 2 bridge.")
    parser.add_argument("--test-steps", type=int, default=0, help="Exit after N simulation steps; 0 runs continuously.")
    args, _unknown = parser.parse_known_args()
    return args


ARGS = build_parser()

# Omniverse/Isaac modules must be imported after SimulationApp has started.
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": bool(ARGS.headless),
        "renderer": "RayTracedLighting",
    }
)

import carb
import numpy as np
import omni.usd
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
from omni.syntheticdata import SyntheticData

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.semantics as semantics_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.objects import Cube, Cylinder, DomeLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.wheeled_robots.controllers import AckermannController
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdPhysics

# Explicitly enable the new Isaac Sim 6 sensor extension and ROS bridge.
app_utils.enable_extension("isaacsim.sensors.experimental.rtx")
if not ARGS.no_ros:
    app_utils.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera


ROBOT_PRIM = "/World/Leatherback"
CHASSIS_PRIM = f"{ROBOT_PRIM}/Rigid_Bodies/Chassis"
CAMERA_PRIM = f"{CHASSIS_PRIM}/front_camera"
START_XY = (-8.0, 0.0)
GOAL_XY = (8.0, 0.0)

# The same physical joints used in the user's previous Leatherback environment.
# The order here follows NVIDIA's native Isaac Sim 6 Leatherback controller example.
STEERING_JOINTS = [
    "Knuckle__Upright__Front_Left",
    "Knuckle__Upright__Front_Right",
]
WHEEL_JOINTS = [
    "Wheel__Upright__Rear_Left",
    "Wheel__Upright__Rear_Right",
    "Wheel__Knuckle__Front_Left",
    "Wheel__Knuckle__Front_Right",
]


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    size: float


def choose_obstacle_count(args: argparse.Namespace, rng: random.Random) -> int:
    if args.num_obstacles is not None:
        return max(0, min(20, int(args.num_obstacles)))
    complexity = args.complexity
    if complexity == "random":
        complexity = rng.choice(("low", "medium", "high"))
    # Non-overlapping interpretation of the requested complexity bins.
    ranges = {"low": (0, 5), "medium": (6, 9), "high": (10, 15)}
    lo, hi = ranges[complexity]
    return rng.randint(lo, hi)


def sample_obstacles(args: argparse.Namespace, count: int, rng: random.Random) -> list[Obstacle]:
    obstacles: list[Obstacle] = []
    attempts = 0
    while len(obstacles) < count and attempts < 5000:
        attempts += 1
        size = rng.uniform(0.45, 0.85)
        x = rng.uniform(-6.3, 6.3)
        y = rng.uniform(-args.world_half_width + 0.8, args.world_half_width - 0.8)

        if math.hypot(x - START_XY[0], y - START_XY[1]) < 2.2:
            continue
        if math.hypot(x - GOAL_XY[0], y - GOAL_XY[1]) < 2.2:
            continue
        if any(math.hypot(x - o.x, y - o.y) < 0.5 * (size + o.size) + 0.65 for o in obstacles):
            continue
        obstacles.append(Obstacle(x=x, y=y, size=size))

    if len(obstacles) != count:
        raise RuntimeError(f"Could only place {len(obstacles)} of {count} obstacles.")
    return obstacles


def spawn_world(args: argparse.Namespace, obstacles: list[Obstacle]) -> None:
    """Build the whole scene with native Isaac Sim 6 Core Experimental objects."""
    stage = omni.usd.get_context().get_stage()
    ground_size = max(2.0 * args.world_half_length + 8.0, 2.0 * args.world_half_width + 8.0)
    ground = GroundPlane(
        "/World/Ground",
        sizes=[ground_size],
        colors=np.array([[0.18, 0.20, 0.22]], dtype=np.float32),
        templates=None,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    )
    ground_material = RigidBodyMaterial(
        "/World/Materials/GroundPhysics",
        static_frictions=[1.0],
        dynamic_frictions=[0.9],
        restitutions=[0.0],
    )
    ground.apply_physics_materials(ground_material)

    light = DomeLight("/World/DomeLight")
    light.set_intensities(2200.0)

    color_choices = np.array(
        [
            [0.75, 0.16, 0.12],
            [0.10, 0.42, 0.78],
            [0.88, 0.58, 0.08],
            [0.42, 0.16, 0.66],
        ],
        dtype=np.float32,
    )
    for index, obstacle in enumerate(obstacles):
        path = f"/World/Obstacles/Cube_{index:02d}"
        cube = Cube(
            path,
            sizes=[obstacle.size],
            colors=color_choices[index % len(color_choices)].reshape(1, 3),
            positions=np.array([[obstacle.x, obstacle.y, 0.5 * obstacle.size]], dtype=np.float32),
        )
        cube_prim = stage.GetPrimAtPath(path)
        if not cube_prim.IsValid():
            raise RuntimeError(f"Failed to create obstacle prim: {path}")
        if not cube_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(cube_prim)
        semantics_utils.add_labels(path, labels=["cube"], taxonomy="class")

    Cylinder(
        "/World/Goal",
        radii=[0.45],
        heights=[0.08],
        axes=["Z"],
        colors=np.array([[0.08, 0.85, 0.15]], dtype=np.float32),
        positions=np.array([[GOAL_XY[0], GOAL_XY[1], 0.04]], dtype=np.float32),
    )
    semantics_utils.add_labels("/World/Goal", labels=["goal"], taxonomy="class")


def load_leatherback() -> tuple[Articulation, float]:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError(
            "Isaac Sim assets root could not be resolved. Check your Isaac Sim asset configuration/network access."
        )
    leatherback_usd = assets_root + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
    stage_utils.add_reference_to_stage(usd_path=leatherback_usd, path=ROBOT_PRIM)
    robot = Articulation(ROBOT_PRIM)

    # The official Leatherback asset is authored to sit on a z=0 ground plane.
    # Place it once, before physics initialization; never teleport the
    # constrained suspension after playback starts.
    start_z = 0.0
    start_position = np.array([[START_XY[0], START_XY[1], start_z]], dtype=np.float32)
    start_orientation = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    robot.set_world_poses(
        positions=start_position,
        orientations=start_orientation,
    )
    robot.set_default_state(
        positions=start_position,
        orientations=start_orientation,
    )
    return robot, start_z


def create_camera(args: argparse.Namespace) -> CameraSensor:
    """Create a Leatherback-mounted Isaac Sim 6 RTX camera."""
    rtx_camera = RtxCamera(
        CAMERA_PRIM,
        tick_rate=max(float(args.camera_fps), 1.0),
        translations=np.array([[args.camera_x, 0.0, args.camera_z]], dtype=np.float32),
        # USD camera -Z forward rotated to Leatherback +X forward; quaternion WXYZ.
        orientations=np.array([[0.5, 0.5, -0.5, -0.5]], dtype=np.float32),
    )
    rtx_camera.camera.set_focal_lengths([24.0])
    rtx_camera.camera.set_focus_distances([400.0])
    rtx_camera.camera.set_apertures(horizontal_apertures=[20.955])
    rtx_camera.camera.set_clipping_ranges(near_distances=[0.05], far_distances=[30.0])
    simulation_app.update()

    annotators = ["rgb"]
    if not args.disable_depth:
        annotators.append("distance_to_image_plane")

    sensor = CameraSensor(
        rtx_camera,
        resolution=(int(args.camera_width), int(args.camera_height)),
        annotators=annotators,
    )
    simulation_app.update()
    return sensor


def camera_render_product_path(camera: CameraSensor) -> str:
    return str(camera.render_product.GetPath())


def configure_ros_camera_publishers(camera: CameraSensor, args: argparse.Namespace) -> list[object]:
    """Publish RGB/depth/CameraInfo using Isaac Sim 6 Replicator ROS writers."""
    from isaacsim.ros2.core import read_camera_info

    render_product = camera_render_product_path(camera)
    writers: list[object] = []

    rgb_rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    rgb_writer = rep.writers.get(rgb_rv + "ROS2PublishImage")
    rgb_writer.initialize(
        frameId=args.camera_frame,
        nodeNamespace="",
        queueSize=1,
        topicName=args.rgb_topic,
    )
    rgb_writer.attach([render_product])
    writers.append(rgb_writer)

    if not args.disable_depth:
        depth_rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.DistanceToImagePlane.name
        )
        depth_writer = rep.writers.get(depth_rv + "ROS2PublishImage")
        depth_writer.initialize(
            frameId=args.camera_frame,
            nodeNamespace="",
            queueSize=1,
            topicName=args.depth_topic,
        )
        depth_writer.attach([render_product])
        writers.append(depth_writer)

    camera_info, _ = read_camera_info(render_product_path=render_product)
    info_writer = rep.writers.get("ROS2PublishCameraInfo")
    info_writer.initialize(
        frameId=args.camera_frame,
        nodeNamespace="",
        queueSize=1,
        topicName=args.camera_info_topic,
        width=camera_info.width,
        height=camera_info.height,
        projectionType=camera_info.distortion_model,
        k=camera_info.k.reshape([1, 9]),
        r=camera_info.r.reshape([1, 9]),
        p=camera_info.p.reshape([1, 12]),
        physicalDistortionModel=camera_info.distortion_model,
        physicalDistortionCoefficients=camera_info.d,
    )
    info_writer.attach([render_product])
    writers.append(info_writer)
    return writers


class PcRosInterface:
    """Small rclpy interface for commands, clock, odometry, and TF.

    High-bandwidth camera data remains on Isaac Sim's native Replicator ROS
    writers. This node handles only low-bandwidth control and state messages.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from ackermann_msgs.msg import AckermannDriveStamped
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)

        self.node = rclpy.create_node("isaac_pc_hil")
        self._Odometry = Odometry
        self._Clock = Clock
        self._last_command_time = float("-inf")
        self.speed_command = 0.0
        self.steering_command = 0.0

        self.node.create_subscription(
            AckermannDriveStamped,
            args.ackermann_topic,
            self._command_callback,
            1,
        )
        self.odom_pub = self.node.create_publisher(Odometry, args.odom_topic, 10)
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self.tf_pub = TransformBroadcaster(self.node)
        self.static_tf_pub = StaticTransformBroadcaster(self.node)
        self._publish_camera_static_tf(args)

    def _command_callback(self, message) -> None:
        self.speed_command = float(message.drive.speed)
        self.steering_command = float(message.drive.steering_angle)
        self._last_command_time = time.monotonic()

    @staticmethod
    def _stamp(sim_time: float):
        from builtin_interfaces.msg import Time

        seconds = max(0.0, float(sim_time))
        sec = int(seconds)
        nanosec = int(round((seconds - sec) * 1.0e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return Time(sec=sec, nanosec=nanosec)

    def _publish_camera_static_tf(self, args: argparse.Namespace) -> None:
        from geometry_msgs.msg import TransformStamped

        transform = TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = args.base_frame
        transform.child_frame_id = args.camera_frame
        transform.transform.translation.x = float(args.camera_x)
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = float(args.camera_z)
        # Camera quaternion is WXYZ [0.5, 0.5, -0.5, -0.5]. ROS uses XYZW.
        transform.transform.rotation.x = 0.5
        transform.transform.rotation.y = -0.5
        transform.transform.rotation.z = -0.5
        transform.transform.rotation.w = 0.5
        self.static_tf_pub.sendTransform(transform)

    def spin_once(self) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def get_command(self, timeout: float) -> tuple[float, float]:
        if time.monotonic() - self._last_command_time > float(timeout):
            return 0.0, 0.0
        return self.speed_command, self.steering_command

    def publish_state(
        self,
        sim_time: float,
        args: argparse.Namespace,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        from geometry_msgs.msg import TransformStamped

        stamp = self._stamp(sim_time)
        w, x, y, z = [float(value) for value in quaternion_wxyz]

        odom = self._Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = args.odom_frame
        odom.child_frame_id = args.base_frame
        odom.pose.pose.position.x = float(position[0])
        odom.pose.pose.position.y = float(position[1])
        odom.pose.pose.position.z = float(position[2])
        odom.pose.pose.orientation.x = x
        odom.pose.pose.orientation.y = y
        odom.pose.pose.orientation.z = z
        odom.pose.pose.orientation.w = w
        odom.twist.twist.linear.x = float(linear_velocity[0])
        odom.twist.twist.linear.y = float(linear_velocity[1])
        odom.twist.twist.linear.z = float(linear_velocity[2])
        odom.twist.twist.angular.x = float(angular_velocity[0])
        odom.twist.twist.angular.y = float(angular_velocity[1])
        odom.twist.twist.angular.z = float(angular_velocity[2])
        self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = args.odom_frame
        transform.child_frame_id = args.base_frame
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])
        transform.transform.rotation.x = x
        transform.transform.rotation.y = y
        transform.transform.rotation.z = z
        transform.transform.rotation.w = w
        self.tf_pub.sendTransform(transform)

        clock = self._Clock()
        clock.clock = stamp
        self.clock_pub.publish(clock)

    def close(self) -> None:
        self.node.destroy_node()
        if self._owns_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_wxyz_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def autonomous_command(
    robot_x: float,
    robot_y: float,
    heading: float,
    goal_xy: tuple[float, float],
    obstacles: list[Obstacle],
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    """Return desired forward speed, steering angle, and nearest cube distance."""
    goal_dx = goal_xy[0] - robot_x
    goal_dy = goal_xy[1] - robot_y
    goal_heading = math.atan2(goal_dy, goal_dx)
    heading_error = wrap_to_pi(goal_heading - heading)

    c = math.cos(heading)
    s = math.sin(heading)
    avoidance = 0.0
    nearest = float("inf")

    for obstacle in obstacles:
        dx = obstacle.x - robot_x
        dy = obstacle.y - robot_y
        x_b = c * dx + s * dy
        y_b = -s * dx + c * dy
        radius = 0.5 * obstacle.size + 0.45
        if x_b <= 0.0 or x_b > args.lookahead:
            continue
        lateral_clearance = abs(y_b) - radius
        if lateral_clearance > args.corridor_half_width:
            continue

        distance = max(0.05, math.hypot(x_b, y_b) - radius)
        nearest = min(nearest, distance)
        side = -1.0 if y_b >= 0.0 else 1.0
        front_weight = math.exp(-distance / 1.15)
        center_weight = 1.0 - min(1.0, abs(y_b) / max(args.corridor_half_width, 1e-3))
        avoidance += side * front_weight * (0.55 + 0.75 * center_weight)

    steer = 0.95 * heading_error + 1.35 * avoidance
    steer = max(-args.max_steer, min(args.max_steer, steer))

    angle_scale = max(0.18, 1.0 - abs(heading_error) / math.pi)
    obstacle_scale = 1.0
    if math.isfinite(nearest):
        obstacle_scale = max(0.10, min(1.0, (nearest - 0.15) / 1.6))
    speed = args.max_speed * angle_scale * obstacle_scale
    return speed, steer, nearest


def get_valid_robot_pose(robot: Articulation) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a finite, normalized root pose or ``None`` on PhysX failure."""
    positions, orientations = robot.get_world_poses()
    position = positions.numpy()[0]
    quaternion = orientations.numpy()[0]

    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        carb.log_error(f"Non-finite Leatherback pose: position={position}, quaternion={quaternion}")
        return None

    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        carb.log_error(f"Invalid Leatherback quaternion norm: {norm}")
        return None
    return position, quaternion / norm


def main() -> None:
    rng = random.Random(ARGS.seed)
    obstacle_count = choose_obstacle_count(ARGS, rng)
    obstacles = sample_obstacles(ARGS, obstacle_count, rng)

    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)

    spawn_world(ARGS, obstacles)
    robot, _start_z = load_leatherback()
    camera = create_camera(ARGS)

    # Native Isaac Sim 6 simulation management; no World/SimulationContext from Isaac Lab.
    SimulationManager.setup_simulation(dt=1.0 / float(ARGS.sim_hz), device=ARGS.device)
    RenderingManager.set_dt(1.0 / float(ARGS.sim_hz))

    if not ARGS.headless:
        try:
            viewport_camera = ViewportManager.get_camera()
            ViewportManager.set_camera_view(
                viewport_camera,
                eye=[-13.0, -12.0, 10.0],
                target=[0.0, 0.0, 0.0],
            )
        except Exception as exc:
            carb.log_warn(f"Could not set the GUI viewport camera: {exc}")

    # Resolve joint IDs before simulation. These are the names used by the native
    # NVIDIA Leatherback USD/controller example.
    steering_dof_indices = robot.get_dof_indices(STEERING_JOINTS)
    wheel_dof_indices = robot.get_dof_indices(WHEEL_JOINTS)

    controller = AckermannController(
        wheel_base=float(ARGS.wheel_base),
        track_width=float(ARGS.track_width),
        front_wheel_radius=float(ARGS.wheel_radius),
        back_wheel_radius=float(ARGS.wheel_radius),
    )

    ros_writers: list[object] = []
    ros_interface: PcRosInterface | None = None
    app_utils.play()

    # Let the constrained suspension settle without applying any joint target.
    settling_frames = max(1, int(ARGS.settling_frames))
    print(f"[Initialization] Passive settling for {settling_frames} frames...")
    for settling_frame in range(settling_frames):
        simulation_app.update()
        if get_valid_robot_pose(robot) is None:
            raise RuntimeError(
                f"Leatherback pose became invalid during settling frame {settling_frame}."
            )

    if not ARGS.no_ros:
        # Start recording/publishing only after the suspension is stable.
        ros_writers = configure_ros_camera_publishers(camera, ARGS)
        ros_interface = PcRosInterface(ARGS)
        print(f"[ROS2] RGB        -> {ARGS.rgb_topic}")
        if not ARGS.disable_depth:
            print(f"[ROS2] Depth      -> {ARGS.depth_topic}")
        print(f"[ROS2] CameraInfo -> {ARGS.camera_info_topic}")
        print(f"[ROS2] Clock      -> /clock")
        print(f"[ROS2] Odometry   -> {ARGS.odom_topic}")
        print(f"[ROS2] TF         -> {ARGS.odom_frame} -> {ARGS.base_frame} -> {ARGS.camera_frame}")
        print(f"[ROS2] Ackermann  <- {ARGS.ackermann_topic}")

    print(
        f"[Scene] complexity={ARGS.complexity} cubes={obstacle_count} "
        f"control={ARGS.control_mode} camera={ARGS.camera_width}x{ARGS.camera_height}@{ARGS.camera_fps:g}Hz"
    )

    frame = 0
    last_print_second = -1
    commanded_speed = 0.0
    previous_position: np.ndarray | None = None
    previous_heading: float | None = None
    while simulation_app.is_running():
        simulation_app.update()
        if not app_utils.is_playing():
            continue

        pose = get_valid_robot_pose(robot)
        if pose is None:
            carb.log_error("Stopping because the Leatherback pose is non-finite.")
            break
        pos, quat = pose
        robot_x = float(pos[0])
        robot_y = float(pos[1])
        heading = quaternion_wxyz_to_yaw(quat)
        goal_distance = math.hypot(GOAL_XY[0] - robot_x, GOAL_XY[1] - robot_y)

        dt = 1.0 / float(ARGS.sim_hz)
        if previous_position is None or previous_heading is None:
            linear_velocity = np.zeros(3, dtype=np.float64)
            angular_velocity = np.zeros(3, dtype=np.float64)
        else:
            linear_velocity = (pos.astype(np.float64) - previous_position) / dt
            angular_velocity = np.array(
                [0.0, 0.0, wrap_to_pi(heading - previous_heading) / dt],
                dtype=np.float64,
            )
        previous_position = pos.astype(np.float64).copy()
        previous_heading = heading

        if ros_interface is not None:
            ros_interface.spin_once()
            ros_interface.publish_state(
                frame * dt,
                ARGS,
                pos,
                quat,
                linear_velocity,
                angular_velocity,
            )

        if ARGS.control_mode == "ros":
            if ros_interface is None:
                raise RuntimeError("--control-mode ros requires ROS 2; remove --no-ros.")
            desired_speed, desired_steer = ros_interface.get_command(ARGS.command_timeout)
            nearest = float("nan")
        else:
            desired_speed, desired_steer, nearest = autonomous_command(
                robot_x, robot_y, heading, GOAL_XY, obstacles, ARGS
            )

        desired_speed = max(-ARGS.max_speed, min(ARGS.max_speed, desired_speed))
        desired_steer = max(-ARGS.max_steer, min(ARGS.max_steer, desired_steer))

        # Slew-rate limit wheel commands; a stale ROS command naturally ramps
        # down to zero through the command-timeout behavior above.
        rate = ARGS.max_acceleration if abs(desired_speed) > abs(commanded_speed) else ARGS.max_deceleration
        max_delta = max(0.0, float(rate)) * dt
        commanded_speed += float(np.clip(desired_speed - commanded_speed, -max_delta, max_delta))

        # Native Isaac Sim 6 Ackermann conversion gives individual front steering
        # angles and wheel velocities for the Leatherback geometry.
        joint_positions, joint_velocities = controller.forward(
            [desired_steer, 0.0, commanded_speed, 0.0, dt]
        )
        if joint_positions is not None:
            robot.set_dof_position_targets(joint_positions, dof_indices=steering_dof_indices)
        if joint_velocities is not None:
            robot.set_dof_velocity_targets(joint_velocities, dof_indices=wheel_dof_indices)

        if ARGS.control_mode == "auto" and goal_distance < ARGS.goal_tolerance:
            carb.log_info("Goal reached. Stopping cleanly; no in-play articulation teleport.")
            break

        current_second = int(frame / max(ARGS.sim_hz, 1.0))
        if current_second != last_print_second:
            last_print_second = current_second
            nearest_text = "n/a" if not math.isfinite(nearest) else f"{nearest:.2f} m"
            print(
                f"[Leatherback] mode={ARGS.control_mode:<4} cubes={obstacle_count:02d} "
                f"goal={goal_distance:5.2f}m target={desired_speed:4.2f}m/s "
                f"applied={commanded_speed:4.2f}m/s "
                f"steer={desired_steer:+.3f}rad nearest={nearest_text}",
                flush=True,
            )

        frame += 1
        if ARGS.test_steps > 0 and frame >= ARGS.test_steps:
            break

    for writer in ros_writers:
        try:
            writer.detach()
        except Exception:
            pass
    if ros_interface is not None:
        ros_interface.close()
    app_utils.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        carb.log_error(f"Leatherback native Isaac Sim 6 demo failed: {exc}")
        raise
    finally:
        simulation_app.close()