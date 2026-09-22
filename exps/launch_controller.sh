#!/usr/bin/env bash
# Usage: bash multi_scale/exps/launch_controller.sh --controller multiscale
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# Help works without ROS or inference dependencies installed.
if [[ "${1:-}" != "--help" && "${1:-}" != "-h" ]]; then
    ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
    if [[ ! -f "$ROS_SETUP" ]]; then
        echo "ROS setup not found: $ROS_SETUP" >&2
        exit 1
    fi
    source "$ROS_SETUP"
fi
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN:-python3}" -u -m multi_scale.exps.run_ros2_controller "$@"
