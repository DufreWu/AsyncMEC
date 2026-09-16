This repository contains the source code for AsyncMEC, an asynchronous multi-scale runtime control framework that jointly optimizes runtime energy efficiency and long-term battery health for battery-powered mobile robots.

# Repository Structure
```text
AsyncMEC/
├── appls/              # Robotic application workloads
├── battery_model/      # Battery modeling and battery condition feature extraction
├── checkpoints/        # Pretrained battery models and controller checkpoints
|── configs/            # Jetson Orin NX DVFS and power monitoring configurations
|── envs/               # Training environments and Jetson Orin NX platform interfaces
|── exps/               # Scripts for running experiments and analyzing results
|── methods/            # AsyncMEC controller and baseline methods
├── ros2_isaacsim_env/  # ROS 2–Isaac Sim integration for HIL (Hardware-in-the-Loop) and fully simulated experiments
|── train/              # Controller training scripts
└── README.md
```

# Checkpoints
The pretrained model checkpoints are not included in this repository during the anonymous review process.
The checkpoints will be released after the paper has been accepted.

# Datasets and Training Data

## Battery Training Data
Battery models are trained using NASA battery datasets.
Download link: [NASA Li-on Battery Datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)

## Controller Training Data
Controller training data are generated through the following steps:

1. **Profile computational workloads.** Measure power consumption and application performance on the Jetson Orin NX across CPU and GPU frequency combinations under different workloads.
2. **Build energy models.** Model computational and mechanical power consumption to estimate the robot’s energy consumption under different operating conditions.
3. **Generate training labels.** Evaluate candidate control configurations using the resulting models and generate labels for controller training.controller training.

# Acknowledge
We thank the developers of the following open-source projects:

- [MAVBench](https://github.com/harvard-edge/MAVBench), which inspired the development of the AsyncMEC hardware-in-the-loop simulator.
- [battery-soh-models](https://github.com/sajadshah/battery-soh-models), which inspired the development of our battery models.
