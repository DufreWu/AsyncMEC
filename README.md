# Repository Structure

```text
AsyncMEC/
├── appls/                 # Robotic application workloads
├── assets/                # Demo overview
├── battery_feature/       # Battery condition feature extraction and related utilities
├── checkpoints/           # Pretrained checkpoints for battery, controller, and appls models
├── configs/               # Jetson Orin NX configurations
├── envs/                  # Jetson Orin NX platform interfaces and robot simulation environments
├── exps/                  # Experiment scripts and analysis utilities
├── methods/               # AsyncMEC controller and baseline methods
├── ros2_isaacsim_env/     # ROS 2 + Isaac Sim integration for HIL and simulation experiments
├── train/                 # Data generation and training scripts
├── README.md              # Project overview and usage notes
```

# Supported Workloads and Platforms

## Supported application workloads
The current supported workloads include:
- YOLOv8
- SegFormer

Additional workloads, including applications based on the ZED SDK, are planned for future extensions.

## Training and runtime control
The framework supports CPU and GPU dynamic voltage and frequency scaling (DVFS) and power monitoring on the Jetson Orin NX. It can also estimate power consumption and application performance using linear models fitted to hardware measurements for training and evaluation without requiring physical hardware at all times.

# Simulation and Evaluation

<p align="center">
  <img src="assets/demo.gif" alt="AsyncMEC demo" width="30%">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/overview.png" alt="AsyncMEC overview" width="42%">
</p>

We use Isaac Sim 6 together with the Jetson Orin NX to build a hardware-in-the-loop (HIL) evaluation platform. Isaac Sim simulates rover motion and sensor observations, while the controller and application workloads run on the Jetson Orin NX. The two sides communicate through ROS 2, enabling runtime control and evaluation under realistic compute and dynamics constraints.

The ROS 2 + Isaac Sim environment is organized under [ros2_isaacsim_env/README.md](ros2_isaacsim_env/README.md). The included launcher scripts support both fixed-complexity and varying-complexity simulation modes.

# Datasets and Training Data

##  Battery-Condition Feature Pretraining
Battery models are trained using NASA battery datasets.
Dataset link: [NASA Li-ion Battery Aging Datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)

See [battery_feature/README.md]

### Prepare the Data

Run these commands from the AsyncMEC repository root:

```bash
python battery_feature/scripts/nasa/data_prep.py \
  --nasa-data-dir "./battery_feature/data/nasa/randomized_battery_usage_dataset \
  --data-dir ./battery_feature/data/nasa \
  --normalized-time-step 1 \
  --window-length 60 \
  --window-overlap 0 \
  --nj 8
```

### Pretrain the Feature Encoder

```bash
python battery_feature/scripts/nasa/train_ae.py \
  --out-dir ./battery_feature/exps/test/ae \
  --data-dir ./battery_feature/data/nasa \
  --train-cells RW13,RW14,RW15,RW16 \
  --valid-cells RW17 \
  --window-length 60 \
  --window-step 50 \
  --latent-size 32 \
  --epochs 200 \
  --lr 0.0001 \
  --nj 8
```

## Controller Training Pipeline

Controller training data are generated through the following steps:

1. **Hardware profiling:** Measure application performance and power consumption across CPU and GPU frequency settings on the Jetson Orin NX.
2. **Model construction:** Build models to estimate application performance and computational and mechanical power consumption under different operating conditions.
3. **Label generation:** Evaluate candidate combinations of CPU frequency, GPU frequency, and motion speed using these models to generate control-action labels.

Train the controller from the repository root:

```bash
python train/train_msc.py \
  --dataset train/dataset/expert_dataset.npz \
  --output-dir checkpoints/msc_control \
  --epochs 200 \
  --batch-size 64
```

# Pretrained Checkpoints

Pretrained model checkpoints are not included in this repository during the anonymous review process. They will be released after paper acceptance.

Included checkpoints target:

- Application models ([YOLOv8](https://yolov8.com/) and [SegFormer](https://github.com/NVlabs/SegFormer)).
- Battery feature.
- Controller.

# Acknowledgments

We thank the developers of the following open-source projects for their contributions and inspiration:

- [MAVBench](https://github.com/harvard-edge/MAVBench), which inspired the development of the AsyncMEC hardware-in-the-loop simulator.
- [battery-soh-models](https://github.com/sajadshah/battery-soh-models), which inspired the design of the battery models used in this work.
