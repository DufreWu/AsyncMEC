#!/usr/bin/env bash
# ROS setup scripts may probe variables that are initially unset, so enable
# nounset only after sourcing ROS 2.
set -eo pipefail

# Run one complexity over the entire route:
#   bash launch_fixed_complexity.sh high
#   bash launch_fixed_complexity.sh --complexity low --headless
# Auto mode follows the simulator route; hybrid uses /asyncmec/speed_limit.
SCENE_COMPLEXITY="medium"
SIM_ARGS=()
if [[ $# -gt 0 && "$1" != -* ]]; then
    SCENE_COMPLEXITY="$1"
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --complexity)
            if [[ $# -lt 2 ]]; then
                echo "--complexity requires low, medium, or high" >&2
                exit 2
            fi
            SCENE_COMPLEXITY="$2"
            shift 2
            ;;
        --complexity=*)
            SCENE_COMPLEXITY="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: bash launch_fixed_complexity.sh [low|medium|high] [simulator options]"
            echo "Alternatively: --complexity low|medium|high (default: medium)"
            exit 0
            ;;
        *)
            SIM_ARGS+=("$1")
            shift
            ;;
    esac
done
case "${SCENE_COMPLEXITY}" in
    low|medium|high) ;;
    *)
        echo "Invalid complexity: ${SCENE_COMPLEXITY}; choose low, medium, or high" >&2
        exit 2
        ;;
esac

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
SIM_SCRIPT="${SCRIPT_DIR}/fixed_complexity.py"
ISAAC_PYTHON="${ISAAC_PYTHON:-python}"

if [[ ! -f "${SIM_SCRIPT}" ]]; then
    echo "Isaac Sim script not found: ${SIM_SCRIPT}" >&2
    exit 1
fi

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "Launching fixed-complexity simulation: ${SCENE_COMPLEXITY}"

exec "${ISAAC_PYTHON}" "${SIM_SCRIPT}" \
    --complexity "${SCENE_COMPLEXITY}" \
    --control-mode auto \
    --record-video \
    --ackermann-topic /ackermann_cmd \
    --max-speed "${MAX_SPEED:-1.0}" \
    "${SIM_ARGS[@]}"
