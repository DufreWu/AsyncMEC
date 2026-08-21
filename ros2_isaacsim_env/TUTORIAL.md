# Tutorial: Native Isaac Sim 6 Leatherback HIL Platform

## 1. Architecture

```text
                LAPTOP — NATIVE ISAAC SIM 6
┌────────────────────────────────────────────────────┐
│ NVIDIA Leatherback                                 │
│   │                                                │
│   ├── RTX front camera                             │
│   │      ├── RGB ──────────────── ROS 2 ───────┐  │
│   │      └── Depth (optional) ─── ROS 2 ────┐   │  │
│   │                                         │   │  │
│   ├── randomized cubes                      │   │  │
│   ├── green navigation goal                 │   │  │
│   └── native Ackermann controller           │   │  │
│          ▲                                  │   │  │
│          ├── autonomous controller          │   │  │
│          └── /leatherback/cmd_vel ◄─────┐   │   │  │
└──────────────────────────────────────────┼───┼───┼──┘
                                           │   │   │
                                           │   │   ▼
                                JETSON ORIN NX      RGB
                              ┌─────────────────────────┐
                              │ YOLO / YOLO-World       │
                              │ GPU inference           │
                              │ FPS measurement         │
                              │ detections              │
                              │ annotated image         │
                              │ future controller       │
                              └─────────────────────────┘
```

The baseline deliberately separates **control** from **perception**:

1. Isaac Sim knows where the cubes are and can run a simple reactive controller by itself.
2. The camera publishes what the robot actually sees.
3. Jetson receives only the camera stream and runs YOLO.
4. You measure YOLO FPS / hardware DVFS / power without risking a perception failure immediately crashing the rover.
5. Later, change the simulator to `--control-mode ros` and let Jetson send commands back.

This staged workflow is much easier to debug and gives clean experimental baselines.

---

## 2. Package layout

```text
leatherback_yolo_ros2_isaacsim6_native/
├── README.md
├── TUTORIAL.md
├── run_sim.sh
├── simulator/
│   └── leatherback_yolo_ros2_demo.py
├── jetson_ros2/
│   ├── jetson_yolo_world_node.py
│   └── requirements.txt
└── ros2/
    └── fastdds.xml
```

There is no Isaac Lab task, environment registry, RL runner, or Isaac Lab launcher in this package.

---

## 3. Prerequisite: verify Isaac Sim 6 itself

Find your Isaac Sim 6 root. It should contain `python.sh` on Linux.

Example:

```bash
cd ~/isaacsim
./python.sh standalone_examples/api/isaacsim.simulation_app/hello_world.py
```

If your install lives elsewhere, use that directory instead.

If you installed Isaac Sim from pip, use the Python environment in which Isaac Sim 6 is installed instead of `python.sh`:

```bash
python /path/to/leatherback_yolo_ros2_demo.py ...
```

Do not run this package through `isaaclab.sh`.

---

## 4. First simulator-only run

Start without ROS. This isolates the simulator, USD, physics, camera, and controller from network issues.

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

Expected console output resembles:

```text
[Scene] complexity=medium cubes=08 control=auto camera=640x480@15Hz
[Leatherback] mode=auto cubes=08 goal=15.7m speed=1.20m/s steer=+0.04rad nearest=n/a
```

Expected visual behavior:

- Leatherback starts on the left side.
- Green goal is on the right side.
- Cubes are distributed between start and goal.
- Rover steers toward the goal.
- When a cube enters the controller's forward corridor, the rover slows and steers away.
- When the goal is reached, Leatherback resets and starts again.

---

## 5. Complexity experiments

### Low

```bash
./run_sim.sh --complexity low --control-mode auto --no-ros
```

0–5 cubes.

### Medium

```bash
./run_sim.sh --complexity medium --control-mode auto --no-ros
```

6–9 cubes.

### High

```bash
./run_sim.sh --complexity high --control-mode auto --no-ros
```

10–15 cubes.

### Exact object count

```bash
./run_sim.sh --num-obstacles 12 --control-mode auto --no-ros
```

This is useful for a paper because you can benchmark YOLO at fixed scene loads such as 0, 3, 6, 9, 12, and 15 objects.

Use a fixed seed for reproducibility:

```bash
./run_sim.sh --num-obstacles 12 --seed 123 --control-mode auto --no-ros
```

---

## 6. What the autonomous controller does

At every simulation step it computes:

1. Heading from Leatherback to the goal.
2. Heading error.
3. Cube positions in the Leatherback body frame.
4. Cubes that are ahead and inside a configurable corridor.
5. A steering repulsion away from nearby cubes.
6. A reduced forward speed near obstacles or during large turns.
7. Desired speed and steering angle.
8. Isaac Sim's native Ackermann controller converts these into front steering joint positions and four wheel velocities.

Important: this controller uses **ground-truth simulator geometry**, not YOLO detections. That is intentional for the first HIL phase.

Controller parameters:

```text
--max-speed
--max-steer
--lookahead
--corridor-half-width
--goal-tolerance
```

Example with slower conservative driving:

```bash
./run_sim.sh \
  --complexity high \
  --max-speed 0.8 \
  --lookahead 3.5 \
  --corridor-half-width 1.3 \
  --no-ros
```

---

## 7. Enable ROS 2 on the laptop

Source ROS before launching Isaac Sim.

For ROS 2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

If you use Humble, source Humble instead.

Run:

```bash
cd /path/to/isaac-sim

./python.sh /path/to/leatherback_yolo_ros2_isaacsim6_native/simulator/leatherback_yolo_ros2_demo.py \
  --complexity medium \
  --control-mode auto \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 15
```

Do not use `--no-ros` here.

---

## 8. Verify camera topics

In a second laptop terminal:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42

ros2 topic list | grep leatherback
```

Expected topics include:

```text
/leatherback/front_camera/rgb
/leatherback/front_camera/depth
/leatherback/front_camera/camera_info
/leatherback/cmd_vel
```

Check RGB rate:

```bash
ros2 topic hz /leatherback/front_camera/rgb
```

Check bandwidth:

```bash
ros2 topic bw /leatherback/front_camera/rgb
```

Visualize RGB:

```bash
rqt_image_view /leatherback/front_camera/rgb
```

If you only need YOLO RGB, disable depth:

```bash
./run_sim.sh --complexity medium --camera-fps 15 --disable-depth
```

---

## 9. Laptop ↔ Jetson ROS 2 networking

Use the same ROS domain on both machines:

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

The package includes:

```text
ros2/fastdds.xml
```

You can place it on both systems and set:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/absolute/path/to/fastdds.xml
```

First verify basic communication before involving Isaac Sim:

Laptop:

```bash
ros2 topic pub -r 1 /network_test std_msgs/msg/String "{data: hello}"
```

Jetson:

```bash
ros2 topic echo /network_test
```

Then start Isaac Sim and test:

```bash
ros2 topic hz /leatherback/front_camera/rgb
```

from the Jetson.

---

## 10. Jetson YOLO setup

The included node receives `sensor_msgs/Image`, runs YOLO, and publishes detections, an annotated image, and FPS.

Install its Python packages in the Jetson ROS/Python environment:

```bash
cd /path/to/leatherback_yolo_ros2_isaacsim6_native
python3 -m pip install -r jetson_ros2/requirements.txt
```

You also need ROS packages such as `cv_bridge` and `vision_msgs` appropriate for your ROS distribution.

Run YOLO-World:

```bash
python3 jetson_ros2/jetson_yolo_world_node.py \
  --weights yolov8s-worldv2.pt \
  --prompts cube,box,block \
  --confidence 0.25 \
  --iou 0.45 \
  --imgsz 640 \
  --device 0 \
  --half
```

The node subscribes to:

```text
/leatherback/front_camera/rgb
```

It publishes:

```text
/leatherback/yolo/detections
/leatherback/yolo/annotated
/leatherback/yolo/fps
```

Monitor FPS:

```bash
ros2 topic echo /leatherback/yolo/fps
```

View detections:

```bash
rqt_image_view /leatherback/yolo/annotated
```

---

## 11. Efficient benchmarking strategy

For perception/DVFS experiments, avoid mixing too many variables at once.

Recommended baseline:

```bash
./run_sim.sh \
  --num-obstacles 6 \
  --seed 10 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 15 \
  --control-mode auto \
  --disable-depth
```

Then sweep one dimension.

### Object-count sweep

```text
0, 3, 6, 9, 12, 15 cubes
```

Record at each point:

```text
actual camera ROS FPS
actual camera bandwidth
YOLO inference FPS
end-to-end image latency
number of detections
precision/recall if ground truth is evaluated
Jetson CPU frequency
Jetson GPU frequency
Jetson power
Jetson temperature
```

### Camera-rate sweep

```text
10, 15, 20, 30 FPS
```

### Resolution sweep

```text
320x240
640x480
1280x720
```

For a fair experiment, keep the random seed fixed while changing only the variable being studied.

---

## 12. Headless mode

Once visual verification is complete, benchmark with:

```bash
./run_sim.sh \
  --headless \
  --num-obstacles 12 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 15 \
  --disable-depth
```

The RTX camera still renders because the standalone app is running with rendering enabled; only the interactive GUI is removed.

---

## 13. External ROS control

Start simulator:

```bash
./run_sim.sh --complexity medium --control-mode ros
```

Send test command:

```bash
ros2 topic pub -r 10 \
  /leatherback/cmd_vel \
  geometry_msgs/msg/Twist \
  "{linear: {x: 0.6}, angular: {z: 0.12}}"
```

In this demo's interface:

```text
linear.x  = desired forward speed [m/s]
angular.z = desired steering angle [rad]
```

The simulator converts those commands to proper Leatherback Ackermann joint targets.

---

## 14. Closing the YOLO control loop later

A 2D YOLO bounding box alone does not provide reliable metric distance. The intended next stage is:

```text
RGB image
   │
   ▼
YOLO bounding box ───────────────┐
                                │
Depth image ── median depth ────┤
                                ▼
                    bearing + obstacle distance
                                │
                                ▼
                       Jetson controller
                                │
                                ▼
                   /leatherback/cmd_vel
                                │
                                ▼
                         Isaac Sim 6
```

The simulator already publishes depth unless `--disable-depth` is used, so the interface for this second stage is present.

---

## 15. Common problems

### `No module named isaacsim`

You ran the script with ordinary system Python. Use:

```bash
/path/to/isaac-sim/python.sh simulator/leatherback_yolo_ros2_demo.py ...
```

or activate the Python environment in which pip-installed Isaac Sim 6 exists.

### Leatherback asset does not load

The script obtains Isaac Sim's assets root and loads:

```text
/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd
```

Check Isaac Sim asset access/network/cache configuration.

### No ROS topics

Make sure you did **not** pass `--no-ros`, source ROS before launching Isaac Sim, and use matching `ROS_DOMAIN_ID` on all terminals/machines.

### RGB topic exists but Jetson receives nothing

Check:

```bash
ros2 topic hz /leatherback/front_camera/rgb
ros2 topic info -v /leatherback/front_camera/rgb
```

Then validate laptop↔Jetson DDS discovery with a trivial topic.

### YOLO does not recognize synthetic cubes well

YOLO-World is convenient for the first test, but synthetic cube appearance can differ from the model's learned concepts. For research-quality results, collect Isaac Sim images and train/fine-tune a detector for your exact obstacle classes.

### Rover gets trapped between cubes

The autonomous controller is reactive, not a global planner. Increase spacing, lower complexity, adjust `--lookahead`/`--corridor-half-width`, or later add A*/Nav2/global planning. The reactive controller is intentionally simple so perception/HIL experiments remain interpretable.

---

## 16. Recommended first three commands

### Test 1: simulator only

```bash
./run_sim.sh --complexity medium --control-mode auto --no-ros --camera-fps 15
```

### Test 2: ROS RGB only

```bash
./run_sim.sh --complexity medium --control-mode auto --camera-fps 15 --disable-depth
```

### Test 3: full RGB + depth + Jetson

```bash
./run_sim.sh --complexity medium --control-mode auto --camera-fps 15
```

Then start `jetson_yolo_world_node.py` on the Jetson.
