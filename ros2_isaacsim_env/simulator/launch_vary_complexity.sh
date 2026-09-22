#!/usr/bin/env bash
# ROS setup scripts may probe variables that are initially unset, so enable
# nounset only after sourcing ROS 2.
set -eo pipefail

# PC-side launcher: Nav2 supplies /plan, AsyncMEC supplies cruise speed on
# /ackermann_cmd, and Isaac Sim retains local steering/obstacle-stop authority.

ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "ROS 2 setup not found: ${ROS_SETUP}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export LD_LIBRARY_PATH="/opt/ros/${ROS_DISTRO_NAME}/lib:/opt/ros/${ROS_DISTRO_NAME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_SCRIPT="${SCRIPT_DIR}/asyncmec_demo.py"
ISAAC_PYTHON="${ISAAC_PYTHON:-python}"

if [[ ! -f "${SIM_SCRIPT}" ]]; then
    echo "Isaac Sim script not found: ${SIM_SCRIPT}" >&2
    exit 1
fi

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "Waiting for Nav2 path on /plan and cruise speed on /ackermann_cmd"

exec "${ISAAC_PYTHON}" "${SIM_SCRIPT}" \
    --complexity medium \
    --control-mode auto \
    --record-video \
    --path-topic /plan \
    --ackermann-topic /ackermann_cmd \
    --path-lookahead "${PATH_LOOKAHEAD:-0.9}" \
    --path-timeout "${PATH_TIMEOUT:-5.0}" \
    --max-speed "${MAX_SPEED:-1.0}" \
    --avoid-stop-distance "${AVOID_STOP_DISTANCE:-0.55}" \
    --avoid-slow-distance "${AVOID_SLOW_DISTANCE:-2.0}" \
    "$@"