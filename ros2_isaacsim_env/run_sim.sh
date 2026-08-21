#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$HOME/isaacsim}"
if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
  echo "ERROR: Isaac Sim python.sh not found at: $ISAAC_SIM_ROOT/python.sh" >&2
  echo "Set ISAAC_SIM_ROOT to your Isaac Sim 6 installation directory." >&2
  echo "Example: export ISAAC_SIM_ROOT=/path/to/isaac-sim" >&2
  exit 2
fi
exec "$ISAAC_SIM_ROOT/python.sh" "$HERE/simulator/leatherback_yolo_ros2_demo.py" "$@"
