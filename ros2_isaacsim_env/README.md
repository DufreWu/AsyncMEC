# ROS 2 + Isaac Sim Environment

This directory contains the host-side Isaac Sim setup used for the AsyncMEC hardware-in-the-loop (HIL) and simulation workflows. It provides the ROS 2 bridge between the simulated robot environment and the controller/runtime stack used in the project.

## Overview

The simulation launcher scripts in this folder start Isaac Sim with the appropriate scene complexity and ROS topics for AsyncMEC experiments:

- `launch_fixed_complexity.sh`: runs a single fixed scene complexity across the route.
- `launch_vary_complexity.sh`: launches the variable-complexity demo flow using the planned path and cruise speed commands.

These scripts are designed to work with ROS 2 and Isaac Sim 6, and they source the ROS 2 environment automatically from `/opt/ros/${ROS_DISTRO}/setup.bash` (defaulting to `jazzy`).

## Prerequisites

Before running the simulation, make sure the following are already set up:

- Ubuntu with ROS 2 installed
- Isaac Sim 6 or a compatible Isaac Sim release
- A Python environment that can launch the simulator scripts
- ROS 2 workspace sourced in the terminal session

## Quick Start

```bash
cd ros2_isaacsim_env

# Fixed scene complexity: low, medium, or high
./simulator/launch_fixed_complexity.sh [low|medium|high]

# Variable-complexity demo
./simulator/launch_vary_complexity.sh
```

## Common Usage Examples

```bash
# Pass simulator options directly
./simulator/launch_fixed_complexity.sh --complexity [low|medium|high] --headless
```

## Notes

- The default ROS domain is `20` and can be overridden with `ROS_DOMAIN_ID`.
- The default ROS middleware is `rmw_fastrtps_cpp` and can be overridden with `RMW_IMPLEMENTATION`.
- The simulator scripts expect the Isaac Sim Python entrypoint to be available through the default `python` interpreter, or through the `ISAAC_PYTHON` environment variable.

For the full project setup and controller workflow, refer to the repository root README in the parent directory.