# multi_scale/appls

applications for running YOLO object detection and SegFormer semantic segmentation on video files or live camera input.

## Prerequisites

Activate your project virtual environment before running the demos. Ensure the required dependencies are available:

- `torch`
- `transformers` for SegFormer
- `ultralytics` for YOLO
- `opencv-python` or a compatible system OpenCV installation

For full `RobotEnv` support, install its dependencies, including `onnxruntime` where required. On Jetson, use a PyTorch installation compatible with your JetPack version and CUDA environment.

## Command-line options

| Option | Description |
| --- | --- |
| `--type` | Input source: `video` or `camera`. Default: `video`. |
| `--video-file` | Input video path; required for video mode. |
| `--camera-index` | Camera index for camera mode; use `0` for `/dev/video0`. |
| `--model` | Model path or identifier accepted by the manager. |
| `--save-file` | Path for the annotated output video. |

The supplied YOLO script also supports `--input` as an alias for `--video-file`, `--output` as an alias for `--save-file`, `--imgsz` for inference image size, and `--no-show` to disable the display window. It saves to `results.mp4` when no output path is specified.

Check SegFormer's `--help` before using these additional YOLO options with it.

## YOLO examples

### Video input

```bash
python -m appls.yolo_manager \
  --type video \
  --video-file /path/to/input.mp4 \
  --model ./checkpoints/appls_models/yolov8n.pt \
  --save-file /path/to/yolo_video.mp4
```

### USB camera input over SSH

```bash
python -m appls.yolo_manager \
  --type camera \
  --camera-index 0 \
  --model /path/to/yolo_model \
  --imgsz 640 \
  --no-show \
  --save-file ./camera_yolo_test.mp4
```

Use `--no-show` when running over SSH without a graphical display. To preview on a connected desktop display, omit `--no-show`.

Press `Ctrl+C` to stop a headless camera test. The supplied YOLO script releases the camera and video writer during cleanup. When a preview window is open, press `q` or `Esc` to stop.

## SegFormer examples

### Video input

```bash
python -m appls.segformer_manager \
  --type video \
  --video-file /path/to/input.mp4 \
  --model /path/to/segformer_model \
  --save-file /path/to/segformer_video.mp4
```

### Camera input

```bash
python -m appls.segformer_manager \
  --type camera \
  --camera-index 0 \
  --model /path/to/segformer_model \
  --save-file /path/to/segformer_camera.mp4
```

For SSH sessions without a graphical display, add `--no-show` only if it is listed in SegFormer's `--help`. Otherwise, the manager needs a headless option before it can run without a display.

## RobotEnv integration

For managers that implement the optional `RobotEnv` fallback, inference can run with `robot=None` when `envs/robot_env.py` or its optional dependencies are unavailable. Board-specific functionality, such as CPU/GPU frequency control, is then disabled.

### Camera cannot be opened

Check the available device nodes:

```bash
ls /dev/video*
```

Confirm the camera index, device permissions, and that another process is not using the camera. Multiple device nodes can belong to one USB camera; not every node necessarily provides capture frames.

### Display errors over SSH

Use `--no-show` for YOLO. For SegFormer, check whether the manager provides an equivalent option.

### Output video is empty or only a few hundred bytes

Confirm that the source produces readable frames, the output directory is writable, and no encoder error was reported. Stop the program cleanly so the video writer can finalize the file; avoid force-killing it or powering off during recording.

### YOLO uses the CPU

Check the startup output for `CUDA available: True` and `YOLO device: 0`. The supplied YOLO script falls back to the CPU when CUDA is unavailable.

### FPS overlay is lower than expected

The supplied YOLO script divides the measured FPS by 10 when drawing its label. To display the unscaled model-call FPS, use:

```python
label = f"FPS:{result['fps']:.1f}"
```

This value includes model preprocessing, inference, and postprocessing. It excludes camera capture, annotation, display, and video encoding, so it is not the end-to-end pipeline FPS.
