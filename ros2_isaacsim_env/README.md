# Native Isaac Sim 6 — Leatherback + ROS 2 + YOLO

This package runs directly in **NVIDIA Isaac Sim 6**. It does **not** require Isaac Lab.

## Included features

- Native NVIDIA Leatherback USD.
- Random static-tree scenes:
  - Low: 0–5 trees
  - Medium: 6–9 trees
  - High: 10–15 trees
  - configurable static people with `--num-people N`
  - exact count with `--num-obstacles N`
- Green goal marker.
- Autonomous goal following.
- Reactive tree and person avoidance.
- Native Isaac Sim 6 Ackermann steering/wheel controller.
- Leatherback-mounted RTX camera.
- RGB and optional depth rendering.
- ROS 2 publishing:
  - `/leatherback/front_camera/rgb`
  - `/leatherback/front_camera/depth`
  - `/leatherback/front_camera/camera_info`
- ROS 2 control subscription:
  - `/leatherback/cmd_vel`
- Jetson Orin NX YOLO/YOLO-World ROS 2 node.
- YOLO detections, annotated image, and inference FPS topics.
- Fast DDS configuration for laptop ↔ Jetson communication.

## 1. Fastest first test — no ROS

Run from your **Isaac Sim 6 installation root**, where `python.sh` exists:

```bash
cd /path/to/isaac-sim

./python.sh /path/to/leatherback_yolo_ros2_isaacsim6_native/simulator/leatherback_yolo_ros2_demo.py \
  --complexity medium \
  --control-mode auto \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 15 \
  --no-ros
```

You should see Isaac Sim open, the Leatherback spawn, randomized static trees
and people appear, and a green goal at x=+8 m.

## 2. Short launcher

Set your Isaac Sim root once:

```bash
export ISAAC_SIM_ROOT=/path/to/isaac-sim
```

Then from this package:

```bash
./run_sim.sh --complexity medium --control-mode auto --no-ros
```

## 3. Run with ROS 2

Source your ROS 2 environment **before starting Isaac Sim**. Example with Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Then:

```bash
cd /path/to/isaac-sim

./python.sh /path/to/leatherback_yolo_ros2_isaacsim6_native/simulator/leatherback_yolo_ros2_demo.py \
  --complexity medium \
  --control-mode auto \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 15
```

Check topics from another ROS terminal:

```bash
ros2 topic list | grep leatherback
ros2 topic hz /leatherback/front_camera/rgb
```

## 4. Run Jetson YOLO

On the Jetson:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
export ROS_DOMAIN_ID=42

python3 -m pip install -r jetson_ros2/requirements.txt

python3 jetson_ros2/jetson_yolo_world_node.py \
  --weights yolov8s-worldv2.pt \
  --prompts cube,box,block \
  --device 0 \
  --half
```

Outputs:

```text
/leatherback/yolo/detections
/leatherback/yolo/annotated
/leatherback/yolo/fps
```

## 5. ROS control mode

Start Isaac Sim with:

```bash
./run_sim.sh --complexity medium --control-mode ros
```

Then publish a command:

```bash
ros2 topic pub -r 10 /leatherback/cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.6}, angular: {z: 0.12}}"
```

For this demo:

- `linear.x` = desired forward speed in m/s.
- `angular.z` = desired steering angle in radians.

## Efficient settings

For initial HIL / YOLO measurements use:

```bash
./run_sim.sh \
  --complexity medium \
  --control-mode auto \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 15 \
  --disable-depth
```

This keeps the RGB workload needed by YOLO but avoids extra depth rendering/ROS traffic. Re-enable depth later when you want to estimate obstacle distance and close the perception-control loop.

For maximum simulator throughput, add:

```bash
--headless
```

See `TUTORIAL.md` for the full workflow and troubleshooting.
