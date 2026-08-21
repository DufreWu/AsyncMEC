# YOLO node for the AsyncMEC HIL demo

Run this node on the Jetson Orin NX. Isaac Sim remains on the host PC and publishes RGB images over ROS 2.

## Dependencies

```bash
sudo apt install ros-$ROS_DISTRO-cv-bridge
python3 -m pip install ultralytics
```

Use the CUDA-enabled PyTorch installation already configured for the Jetson. Do not replace it with a generic CPU wheel.

## Run

Start the Isaac Sim side first:

```bash
./python.sh asyncmec_grid_demo.py --scene grid --control-mode hybrid --disable-depth
```

On the Jetson:

```bash
python3 yolo_ros2_node.py \
  --model yolov8n.pt \
  --device 0 \
  --input-topic /front_camera/rgb \
  --imgsz 640
```

For a local TensorRT model:

```bash
python3 yolo_ros2_node.py --model checkpoints/yolov8n.engine --device 0
```

## Published topics

| Topic | Type | Purpose |
| --- | --- | --- |
| `/yolo/annotated` | `sensor_msgs/Image` | Image with YOLO bounding boxes |
| `/yolo/fps` | `std_msgs/Float32` | Smoothed inference throughput |
| `/yolo/latency_ms` | `std_msgs/Float32` | Per-frame inference latency |
| `/yolo/object_count` | `std_msgs/Int32` | Measured number of detections |
| `/yolo/workload` | `std_msgs/String` | `low`, `medium`, or `high` |
| `/yolo/detections` | `std_msgs/String` | Compact JSON detection details |

The default workload mapping is low for 0--5 detections, medium for 6--9, and high for 10 or more. Override it using `--low-max` and `--medium-max`.

## Verify

```bash
ros2 topic hz /front_camera/rgb
ros2 topic echo /yolo/fps
ros2 topic echo /yolo/object_count
ros2 topic echo /yolo/workload
```

To view the annotated output, open RViz2 and add an Image display for `/yolo/annotated`, or run:

```bash
ros2 run rqt_image_view rqt_image_view /yolo/annotated
```
