#!/usr/bin/env python3
"""AsyncMEC Flat Grid demo: Isaac Sim 6 PC-side HIL server.

This is a standalone NVIDIA Isaac Sim 6 application. It does not depend on
Isaac Lab. Run it with Isaac Sim's ``python.sh`` (or with the Python environment
of a pip-installed Isaac Sim 6).

Features
--------
* NVIDIA Leatherback USD from the Isaac Sim asset server.
* Lightweight Flat Grid scene with repeatable low/medium/high complexity zones.
* Simple colored obstacles and a green route goal with no environment download.
* ``hybrid`` mode: simulator route steering plus an AsyncMEC speed limit from Jetson.
* Optional autonomous goal follower with reactive carton avoidance.
* Native Isaac Sim 6 Ackermann controller for Leatherback steering/wheels.
* Front RTX RGB + depth camera attached to Leatherback.
* ROS 2 RGB, depth (optional), and CameraInfo publishing.
* ROS 2 ``ackermann_msgs/AckermannDriveStamped`` command subscriber.
* ROS 2 clock, odometry, and TF publishing for Nav2/Jetson integration.
* Passive suspension settling, command timeout, acceleration limiting, and
  non-finite pose protection.

The route follower intentionally uses simulator-known geometry; it is not a
claim of perception-based autonomous navigation. YOLO runs independently on
the Jetson for perception/FPS measurements. In the recommended ``hybrid`` mode,
Jetson publishes AsyncMEC's selected speed to ``/asyncmec/speed_limit`` while
Isaac Sim supplies repeatable steering along the grid route.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without the interactive GUI.")
    parser.add_argument("--scene", choices=("grid", "random"), default="grid")
    parser.add_argument("--start-x", type=float, default=-8.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float, default=8.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--complexity", choices=("low", "medium", "high", "random"), default="medium",
                        help="Used by random mode; grid mode uses fixed low/medium/high zones.")
    parser.add_argument("--num-obstacles", type=int, default=None, help="Exact object count; overrides --complexity.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control-mode", choices=("auto", "ros", "hybrid"), default="hybrid")
    parser.add_argument("--ackermann-topic", default="/ackermann_cmd")
    parser.add_argument("--speed-limit-topic", default="/asyncmec/speed_limit")
    parser.add_argument("--scene-zone-topic", default="/asyncmec/scene_zone")
    parser.add_argument("--scene-count-topic", default="/asyncmec/scene_object_count")
    parser.add_argument("--speed-limit-timeout", type=float, default=2.5,
                        help="Stop in hybrid mode if the 1 Hz AsyncMEC command becomes stale [s].")
    parser.add_argument("--rgb-topic", default="/front_camera/rgb")
    parser.add_argument("--depth-topic", default="/front_camera/depth")
    parser.add_argument("--camera-info-topic", default="/front_camera/camera_info")
    parser.add_argument("--camera-frame", default="camera_right")
    parser.add_argument(
        "--camera-prim",
        default="/World/Leatherback/Rigid_Bodies/Chassis/Camera_Right",
        help="Existing Leatherback USD Camera prim to publish.",
    )
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument("--camera-x", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--camera-z", type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument("--sim-hz", type=float, default=60.0)
    parser.add_argument("--device", default="cpu", help="Physics device, e.g. cpu or cuda:0.")
    parser.add_argument("--wheel-base", type=float, default=0.32)
    parser.add_argument("--track-width", type=float, default=0.24)
    parser.add_argument("--wheel-radius", type=float, default=0.052)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--max-steer", type=float, default=0.7854)
    parser.add_argument("--max-acceleration", type=float, default=0.5)
    parser.add_argument("--max-deceleration", type=float, default=1.0)
    parser.add_argument("--max-steer-rate", type=float, default=0.8, help="Maximum steering change [rad/s].")
    parser.add_argument("--steer-time-constant", type=float, default=0.20,
                        help="First-order steering smoothing time constant [s].")
    parser.add_argument("--enable-avoidance", action="store_true",
                        help="Enable simulator-geometry obstacle avoidance; off by default for a smooth demo route.")
    parser.add_argument("--command-timeout", type=float, default=0.5)
    parser.add_argument("--settling-frames", type=int, default=60)
    parser.add_argument("--goal-tolerance", type=float, default=0.90, help="Closed-loop waypoint switch radius [m].")
    parser.add_argument("--lookahead", type=float, default=3.0)
    parser.add_argument("--corridor-half-width", type=float, default=1.15)
    parser.add_argument("--world-half-length", type=float, default=10.0)
    parser.add_argument("--world-half-width", type=float, default=5.5)
    parser.add_argument("--disable-depth", action="store_true", help="Do not render or publish depth.")
    parser.add_argument("--no-ros", action="store_true", help="Do not load the Isaac Sim ROS 2 bridge.")
    parser.add_argument("--test-steps", type=int, default=0, help="Exit after N simulation steps; 0 runs continuously.")
    parser.add_argument("--record-video", action="store_true", help="Record the active Isaac Sim viewport to MP4.")
    parser.add_argument("--record-seconds", type=float, default=30.0)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-output", default="~/Videos/AsyncMEC")
    parser.add_argument("--record-name", default="asyncmec_isaac_view")
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
from pxr import Usd, UsdGeom, UsdPhysics

# Explicitly enable the new Isaac Sim 6 sensor extension and ROS bridge.
app_utils.enable_extension("isaacsim.sensors.experimental.rtx")
if not ARGS.no_ros:
    app_utils.enable_extension("isaacsim.ros2.bridge")
if ARGS.record_video:
    app_utils.enable_extension("omni.kit.capture.viewport")
    app_utils.enable_extension("omni.videoencoding")
simulation_app.update()

from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera


ROBOT_PRIM = "/World/Leatherback"
CHASSIS_PRIM = f"{ROBOT_PRIM}/Rigid_Bodies/Chassis"
CAMERA_PRIM = f"{CHASSIS_PRIM}/Camera_Right"
START_XY = (float(ARGS.start_x), float(ARGS.start_y))
GOAL_XY = (float(ARGS.goal_x), float(ARGS.goal_y))
# Clockwise rounded-rectangle route. The final waypoint connects back to the
# first one, so autonomous/hybrid operation can repeat indefinitely.
ROUTE_WAYPOINTS = (
    START_XY,
    (-6.0, -3.6),
    (0.0, -4.0),
    (6.0, -3.6),
    (8.0, 0.0),
    (6.0, 3.6),
    (0.0, 4.0),
    (-6.0, 3.6),
)

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


@dataclass(frozen=True)
class DemoZone:
    name: str
    progress_min: float
    progress_max: float
    object_count: int
    color: tuple[float, float, float]


DEMO_ZONES = (
    DemoZone("low", 0.00, 0.33, 4, (0.12, 0.68, 0.26)),
    DemoZone("medium", 0.33, 0.66, 8, (0.95, 0.62, 0.08)),
    DemoZone("high", 0.66, 1.01, 14, (0.82, 0.16, 0.12)),
)


def point_to_segment_distance(
    x: float,
    y: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    """Return distance and clamped projection fraction for a line segment."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator < 1.0e-9:
        return math.hypot(x - start[0], y - start[1]), 0.0
    fraction = float(np.clip(((x - start[0]) * dx + (y - start[1]) * dy) / denominator, 0.0, 1.0))
    nearest_x = start[0] + fraction * dx
    nearest_y = start[1] + fraction * dy
    return math.hypot(x - nearest_x, y - nearest_y), fraction


def route_progress(x: float, y: float) -> float:
    """Project a position onto the closed route and return lap progress [0, 1)."""
    segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total_length = 0.0
    for index, start in enumerate(ROUTE_WAYPOINTS):
        end = ROUTE_WAYPOINTS[(index + 1) % len(ROUTE_WAYPOINTS)]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        segments.append((start, end, length))
        total_length += length

    best_distance = float("inf")
    best_progress = 0.0
    travelled = 0.0
    for start, end, length in segments:
        distance, fraction = point_to_segment_distance(x, y, start, end)
        if distance < best_distance:
            best_distance = distance
            best_progress = (travelled + fraction * length) / max(total_length, 1.0e-9)
        travelled += length
    return best_progress % 1.0


def distance_to_route(x: float, y: float) -> float:
    return min(
        point_to_segment_distance(
            x,
            y,
            ROUTE_WAYPOINTS[index],
            ROUTE_WAYPOINTS[(index + 1) % len(ROUTE_WAYPOINTS)],
        )[0]
        for index in range(len(ROUTE_WAYPOINTS))
    )


def demo_zone_at(x: float, y: float) -> DemoZone:
    progress = route_progress(x, y)
    for zone in DEMO_ZONES:
        if zone.progress_min <= progress < zone.progress_max:
            return zone
    return DEMO_ZONES[-1]


def zoned_grid_objects(args: argparse.Namespace, seed: int) -> list[Obstacle]:
    """Scatter deterministic cartons across the world outside the loop lane.

    Counts are environment labels for the repeatable demo. Jetson should still
    use its measured application workload (for example YOLO detections) as the
    controller input; these counts are only ground truth for plots/debugging.
    """
    rng = random.Random(seed)
    objects: list[Obstacle] = []
    for zone in DEMO_ZONES:
        placed = 0
        attempts = 0
        while placed < zone.object_count and attempts < 20000:
            attempts += 1
            size = rng.uniform(0.35, 0.72)
            x = rng.uniform(-args.world_half_length + 0.7, args.world_half_length - 0.7)
            y = rng.uniform(-args.world_half_width + 0.7, args.world_half_width - 0.7)
            if demo_zone_at(x, y).name != zone.name:
                continue
            # Keep a safe 2.4 m-wide lane around the entire closed route.
            if distance_to_route(x, y) < 1.20 + 0.5 * size:
                continue
            if math.hypot(x - START_XY[0], y - START_XY[1]) < 1.8:
                continue
            if any(math.hypot(x - item.x, y - item.y) < 0.5 * (size + item.size) + 0.35 for item in objects):
                continue
            objects.append(Obstacle(x=x, y=y, size=size))
            placed += 1
        if placed != zone.object_count:
            raise RuntimeError(f"Could only place {placed}/{zone.object_count} boxes in {zone.name} zone.")
    return objects


def choose_obstacle_count(args: argparse.Namespace, rng: random.Random) -> int:
    if args.num_obstacles is not None:
        return max(0, min(30, int(args.num_obstacles)))
    complexity = args.complexity
    if complexity == "random":
        complexity = rng.choice(("low", "medium", "high"))
    # Non-overlapping interpretation of the requested complexity bins.
    box_ranges = {"low": (0, 5), "medium": (6, 9), "high": (10, 15)}
    box_lo, box_hi = box_ranges[complexity]
    return rng.randint(box_lo, box_hi)


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
        if distance_to_route(x, y) < 1.20 + 0.5 * size:
            continue
        if any(math.hypot(x - o.x, y - o.y) < 0.5 * (size + o.size) + 0.65 for o in obstacles):
            continue
        obstacles.append(Obstacle(x=x, y=y, size=size))

    if len(obstacles) != count:
        raise RuntimeError(f"Could only place {len(obstacles)} of {count} obstacles.")
    return obstacles


def spawn_grid_world(args: argparse.Namespace, obstacles: list[Obstacle]) -> None:
    """Build a lightweight Flat Grid scene without external environment assets."""
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

    box_colors = np.array(
        [
            [0.58, 0.34, 0.14],
            [0.72, 0.48, 0.24],
        ],
        dtype=np.float32,
    )
    # People are created separately as static character USDs; every object
    # passed to this function is therefore a cardboard box.
    for index, obstacle in enumerate(obstacles):
        path = f"/World/GridObstacles/Box_{index:02d}"
        Cube(
            path,
            sizes=[obstacle.size],
            colors=box_colors[(index // 2) % len(box_colors)].reshape(1, 3),
            positions=np.array(
                [[obstacle.x, obstacle.y, 0.5 * obstacle.size]], dtype=np.float32
            ),
        )
        obstacle_prim = stage.GetPrimAtPath(path)
        if not obstacle_prim.IsValid():
            raise RuntimeError(f"Failed to create obstacle prim: {path}")
        if not obstacle_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(obstacle_prim)
        semantics_utils.add_labels(path, labels=["box"], taxonomy="class")

    # Thin route markers are visual-only and show the repeating loop.
    for index, waypoint in enumerate(ROUTE_WAYPOINTS):
        marker_path = f"/World/RouteMarkers/Waypoint_{index:02d}"
        Cylinder(
            marker_path,
            radii=[0.16],
            heights=[0.025],
            axes=["Z"],
            colors=np.array([[0.08, 0.85, 0.15]], dtype=np.float32),
            positions=np.array([[waypoint[0], waypoint[1], 0.013]], dtype=np.float32),
        )


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
    start_yaw = math.radians(float(ARGS.start_yaw_deg))
    start_orientation = np.array(
        [[math.cos(0.5 * start_yaw), 0.0, 0.0, math.sin(0.5 * start_yaw)]],
        dtype=np.float32,
    )
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
    """Wrap Leatherback's authored right camera without changing its transform."""
    stage = omni.usd.get_context().get_stage()
    camera_path = str(args.camera_prim)
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Leatherback camera prim not found or not a USD Camera: {camera_path}")

    # Cameras authored in the Leatherback asset are plain UsdGeom.Camera prims.
    # Isaac Sim 6 multi-tick RTX cameras require this API schema.
    if not camera_prim.ApplyAPI("OmniSensorAPI"):
        raise RuntimeError(f"Could not apply OmniSensorAPI to camera prim: {camera_path}")

    rtx_camera = RtxCamera(
        camera_path,
        tick_rate=max(float(args.camera_fps), 1.0),
    )
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

def start_viewport_recording(args: argparse.Namespace):
    """Start an MP4 capture of the active overview viewport."""
    if not args.record_video:
        return None
    if args.headless:
        raise RuntimeError("--record-video requires a visible viewport; remove --headless.")

    import omni.kit.viewport.utility as viewport_utils
    import omni.timeline
    from omni.kit.capture.viewport import (
        CaptureExtension,
        CaptureOptions,
        CaptureRangeType,
        CaptureRenderPreset,
    )

    viewport = viewport_utils.get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport is available for video capture.")

    output_folder = Path(args.record_output).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    timeline = omni.timeline.get_timeline_interface()
    start_time = float(timeline.get_current_time())
    duration = max(0.1, float(args.record_seconds))

    options = CaptureOptions()
    options.camera = viewport.camera_path.pathString
    options.range_type = CaptureRangeType.SECONDS
    options.start_time = start_time
    options.end_time = start_time + duration
    options.capture_every_Nth_frames = 1
    options.fps = max(1, int(args.record_fps))
    options.res_width = max(64, int(args.record_width))
    options.res_height = max(64, int(args.record_height))
    options.render_preset = CaptureRenderPreset.RAY_TRACE
    options.real_time_settle_latency_frames = 2
    options.output_folder = str(output_folder)
    options.file_name = str(args.record_name)
    options.file_type = ".mp4"
    options.overwrite_existing_frames = True
    options.mp4_encoding_bitrate = 25_000_000
    options.app_level_capture = False

    capture = CaptureExtension.get_instance()
    capture.options = options
    if not capture.start():
        raise RuntimeError("Isaac Sim viewport video capture failed to start.")

    print(
        f"[Recording] {duration:.1f}s at {options.fps:g} FPS, "
        f"{options.res_width}x{options.res_height} -> {output_folder / (args.record_name + '.mp4')}",
        flush=True,
    )
    return capture


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
        from std_msgs.msg import Float32, Int32, String
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)

        self.node = rclpy.create_node("isaac_pc_hil")
        self._Odometry = Odometry
        self._Clock = Clock
        self._last_command_time = float("-inf")
        self._last_speed_limit_time = float("-inf")
        self.speed_command = 0.0
        self.steering_command = 0.0
        self.speed_limit = 0.0

        self.node.create_subscription(
            AckermannDriveStamped,
            args.ackermann_topic,
            self._command_callback,
            1,
        )
        self.node.create_subscription(Float32, args.speed_limit_topic, self._speed_limit_callback, 1)
        self.odom_pub = self.node.create_publisher(Odometry, args.odom_topic, 10)
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self.zone_pub = self.node.create_publisher(String, args.scene_zone_topic, 10)
        self.scene_count_pub = self.node.create_publisher(Int32, args.scene_count_topic, 10)
        self.tf_pub = TransformBroadcaster(self.node)
        self.static_tf_pub = StaticTransformBroadcaster(self.node)
        self._publish_camera_static_tf(args)

    def _command_callback(self, message) -> None:
        self.speed_command = float(message.drive.speed)
        self.steering_command = float(message.drive.steering_angle)
        self._last_command_time = time.monotonic()

    def _speed_limit_callback(self, message) -> None:
        value = float(message.data)
        if math.isfinite(value):
            self.speed_limit = max(0.0, value)
            self._last_speed_limit_time = time.monotonic()

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

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(ROBOT_PRIM)
        camera_prim = stage.GetPrimAtPath(str(args.camera_prim))
        robot_world = UsdGeom.Xformable(robot_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_in_robot = camera_world * robot_world.GetInverse()
        translation = camera_in_robot.ExtractTranslation()
        rotation = camera_in_robot.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()

        transform = TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = args.base_frame
        transform.child_frame_id = args.camera_frame
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        transform.transform.rotation.x = float(imaginary[0])
        transform.transform.rotation.y = float(imaginary[1])
        transform.transform.rotation.z = float(imaginary[2])
        transform.transform.rotation.w = float(rotation.GetReal())
        self.static_tf_pub.sendTransform(transform)

    def spin_once(self) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def get_command(self, timeout: float) -> tuple[float, float]:
        if time.monotonic() - self._last_command_time > float(timeout):
            return 0.0, 0.0
        return self.speed_command, self.steering_command

    def get_speed_limit(self, timeout: float) -> tuple[float, bool]:
        fresh = time.monotonic() - self._last_speed_limit_time <= float(timeout)
        return (self.speed_limit if fresh else 0.0), fresh

    def publish_scene_state(self, zone: DemoZone) -> None:
        from std_msgs.msg import Int32, String

        zone_message = String()
        zone_message.data = zone.name
        self.zone_pub.publish(zone_message)
        count_message = Int32()
        count_message.data = int(zone.object_count)
        self.scene_count_pub.publish(count_message)

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
    if ARGS.scene == "grid":
        boxes = zoned_grid_objects(ARGS, ARGS.seed)
        box_cnt = len(boxes)
    else:
        box_cnt = choose_obstacle_count(ARGS, rng)
        boxes = sample_obstacles(ARGS, box_cnt, rng)
    obstacles = boxes

    stage_utils.set_stage_up_axis("Z")
    stage_utils.set_stage_units(meters_per_unit=1.0)
    spawn_grid_world(ARGS, boxes)
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
                eye=[START_XY[0] - 8.0, START_XY[1] - 8.0, 9.0],
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
        print(f"[ROS2] CameraPrim -> {ARGS.camera_prim}")
        print(f"[ROS2] Clock      -> /clock")
        print(f"[ROS2] Odometry   -> {ARGS.odom_topic}")
        print(f"[ROS2] TF         -> {ARGS.odom_frame} -> {ARGS.base_frame} -> {ARGS.camera_frame}")
        print(f"[ROS2] Ackermann  <- {ARGS.ackermann_topic}")
        print(f"[ROS2] SpeedLimit <- {ARGS.speed_limit_topic}")
        print(f"[ROS2] SceneZone  -> {ARGS.scene_zone_topic}")
        print(f"[ROS2] SceneCount -> {ARGS.scene_count_topic} (ground truth/debug only)")

    print(
        f"[Scene] scene={ARGS.scene} boxes={box_cnt} route=closed-loop "
        f"control={ARGS.control_mode} camera={ARGS.camera_width}x{ARGS.camera_height}@{ARGS.camera_fps:g}Hz"
    )
    viewport_capture = start_viewport_recording(ARGS)

    frame = 0
    last_print_second = -1
    commanded_speed = 0.0
    commanded_steer = 0.0
    previous_position: np.ndarray | None = None
    previous_heading: float | None = None
    previous_zone_name: str | None = None
    waypoint_index = 1
    lap_count = 0

    while simulation_app.is_running():
        simulation_app.update()
        if viewport_capture is not None and viewport_capture.done:
            outputs = viewport_capture.get_outputs()
            print(f"[Recording] Completed: {outputs}", flush=True)
            break
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
        target_waypoint = ROUTE_WAYPOINTS[waypoint_index]
        goal_distance = math.hypot(target_waypoint[0] - robot_x, target_waypoint[1] - robot_y)
        if ARGS.control_mode in ("auto", "hybrid") and goal_distance < ARGS.goal_tolerance:
            reached_index = waypoint_index
            waypoint_index = (waypoint_index + 1) % len(ROUTE_WAYPOINTS)
            if reached_index == 0:
                lap_count += 1
                carb.log_info(f"Closed-loop lap {lap_count} completed.")
            target_waypoint = ROUTE_WAYPOINTS[waypoint_index]
            goal_distance = math.hypot(target_waypoint[0] - robot_x, target_waypoint[1] - robot_y)
        current_zone = demo_zone_at(robot_x, robot_y)

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

        speed_limit_fresh = True
        speed_limit = float(ARGS.max_speed)
        if ARGS.control_mode == "ros":
            if ros_interface is None:
                raise RuntimeError("--control-mode ros requires ROS 2; remove --no-ros.")
            desired_speed, desired_steer = ros_interface.get_command(ARGS.command_timeout)
            nearest = float("nan")
        else:
            navigation_obstacles = obstacles if ARGS.enable_avoidance else []
            desired_speed, desired_steer, nearest = autonomous_command(
                robot_x, robot_y, heading, target_waypoint, navigation_obstacles, ARGS
            )
            if ARGS.control_mode == "hybrid":
                if ros_interface is None:
                    raise RuntimeError("--control-mode hybrid requires ROS 2; remove --no-ros.")
                speed_limit, speed_limit_fresh = ros_interface.get_speed_limit(ARGS.speed_limit_timeout)
                desired_speed = min(desired_speed, speed_limit)

        desired_speed = max(-ARGS.max_speed, min(ARGS.max_speed, desired_speed))
        desired_steer = max(-ARGS.max_steer, min(ARGS.max_steer, desired_steer))

        # Slew-rate limit wheel commands; a stale ROS command naturally ramps
        # down to zero through the command-timeout behavior above.
        rate = ARGS.max_acceleration if abs(desired_speed) > abs(commanded_speed) else ARGS.max_deceleration
        max_delta = max(0.0, float(rate)) * dt
        commanded_speed += float(np.clip(desired_speed - commanded_speed, -max_delta, max_delta))

        # Smooth steering independently of the 1 Hz AsyncMEC speed decisions.
        # The first-order filter suppresses frame-to-frame heading noise, while
        # the slew-rate limit prevents abrupt front-wheel target changes.
        steer_tau = max(0.0, float(ARGS.steer_time_constant))
        steer_alpha = 1.0 if steer_tau == 0.0 else dt / (steer_tau + dt)
        filtered_steer = commanded_steer + steer_alpha * (desired_steer - commanded_steer)
        max_steer_delta = max(0.0, float(ARGS.max_steer_rate)) * dt
        commanded_steer += float(
            np.clip(filtered_steer - commanded_steer, -max_steer_delta, max_steer_delta)
        )

        # Native Isaac Sim 6 Ackermann conversion gives individual front steering
        # angles and wheel velocities for the Leatherback geometry.
        joint_positions, joint_velocities = controller.forward(
            [commanded_steer, 0.0, commanded_speed, 0.0, dt]
        )
        if joint_positions is not None:
            robot.set_dof_position_targets(joint_positions, dof_indices=steering_dof_indices)
        if joint_velocities is not None:
            robot.set_dof_velocity_targets(joint_velocities, dof_indices=wheel_dof_indices)

        current_second = int(frame / max(ARGS.sim_hz, 1.0))
        if current_second != last_print_second:
            last_print_second = current_second
            if ros_interface is not None and ARGS.scene == "grid":
                ros_interface.publish_scene_state(current_zone)
            if current_zone.name != previous_zone_name:
                print(
                    f"[Grid] ENTER {current_zone.name.upper()} zone: "
                    f"ground_truth_objects={current_zone.object_count}",
                    flush=True,
                )
                previous_zone_name = current_zone.name
            nearest_text = "n/a" if not math.isfinite(nearest) else f"{nearest:.2f} m"
            limiter_text = "n/a"
            if ARGS.control_mode == "hybrid":
                limiter_text = f"{speed_limit:.2f}m/s" if speed_limit_fresh else "STALE->STOP"
            print(
                f"[Leatherback] mode={ARGS.control_mode:<6} zone={current_zone.name:<6} "
                f"lap={lap_count:02d} wp={waypoint_index}/{len(ROUTE_WAYPOINTS)-1} "
                f"goal={goal_distance:5.2f}m target={desired_speed:4.2f}m/s "
                f"applied={commanded_speed:4.2f}m/s "
                f"limit={limiter_text} steer={commanded_steer:+.3f}rad nearest={nearest_text}",
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