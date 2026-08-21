#!/usr/bin/env bash
set -Eeo pipefail

# One-command launcher for AsyncMEC YOLO, image viewing, and aligned MCAP recording.
# When Isaac Sim stops publishing /clock, the watchdog closes the recorder cleanly.

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-jazzy}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-env_isaacsim}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/Documents/Github/llm_controller/ros2_isaacsim_env}"
YOLO_SCRIPT="${YOLO_SCRIPT:-$PROJECT_DIR/jetson_ros2/jetson_yolo_ros2_node.py}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/../multi_scale/checkpoints/yolov8n.pt}"
INPUT_TOPIC="${INPUT_TOPIC:-/front_camera/rgb}"
OUTPUT_TOPIC="${OUTPUT_TOPIC:-/yolo/annotated}"
DEVICE="${DEVICE:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
TOPIC_TIMEOUT="${TOPIC_TIMEOUT:-30}"

RECORD_BAG="${RECORD_BAG:-1}"
BAG_STORAGE="${BAG_STORAGE:-mcap}"
BAG_ROOT="${BAG_ROOT:-$HOME/Videos/AsyncMEC/recordings}"
RUN_NAME="${RUN_NAME:-asyncmec_$(date +%Y%m%d_%H%M%S)}"
BAG_PATH="$BAG_ROOT/$RUN_NAME"

# The watchdog arms only after the first /clock message. A later clock silence
# means that the Isaac Sim timeline has stopped.
AUTO_STOP_ON_CLOCK="${AUTO_STOP_ON_CLOCK:-1}"
CLOCK_START_TIMEOUT="${CLOCK_START_TIMEOUT:-60}"
CLOCK_STOP_TIMEOUT="${CLOCK_STOP_TIMEOUT:-3}"

LAUNCHER_PID="$$"
YOLO_PID=""
BAG_PID=""
VIEWER_PID=""
CLOCK_WATCHDOG_PID=""
CLEANUP_STARTED=0

process_is_running() {
    local process_pid
    process_pid="$1"
    [[ -n "$process_pid" ]] && kill -0 "$process_pid" 2>/dev/null
}

stop_process() {
    local process_name
    local process_pid
    local process_signal
    local timeout_seconds
    local deadline

    process_name="$1"
    process_pid="$2"
    process_signal="${3:-TERM}"
    timeout_seconds="${4:-10}"

    if ! process_is_running "$process_pid"; then
        return
    fi

    echo "[Launcher] Stopping $process_name (PID $process_pid, SIG$process_signal)..."
    kill "-$process_signal" "$process_pid" 2>/dev/null || true

    deadline=$((SECONDS + timeout_seconds))
    while process_is_running "$process_pid" && (( SECONDS < deadline )); do
        sleep 0.2
    done

    if process_is_running "$process_pid"; then
        echo "[Warning] $process_name did not stop within ${timeout_seconds}s; sending SIGTERM." >&2
        kill -TERM "$process_pid" 2>/dev/null || true
        deadline=$((SECONDS + 10))
        while process_is_running "$process_pid" && (( SECONDS < deadline )); do
            sleep 0.2
        done
    fi

    if process_is_running "$process_pid"; then
        echo "[Warning] $process_name is still running (PID $process_pid)." >&2
        echo "          Not using SIGKILL automatically, to avoid corrupting the MCAP file." >&2
        return
    fi

    wait "$process_pid" 2>/dev/null || true
}

cleanup() {
    if (( CLEANUP_STARTED == 1 )); then
        return
    fi
    CLEANUP_STARTED=1
    trap - EXIT INT TERM

    stop_process "clock watchdog" "$CLOCK_WATCHDOG_PID" TERM 5
    stop_process "image viewer" "$VIEWER_PID" TERM 5
    # SIGTERM allows rosbag2 to flush its cache and write the MCAP footer.
    stop_process "rosbag recorder" "$BAG_PID" TERM 30
    stop_process "YOLO node" "$YOLO_PID" TERM 10

    if [[ "$RECORD_BAG" == "1" && -n "$BAG_PID" ]]; then
        echo "[Launcher] ROS bag saved to: $BAG_PATH"
    fi
}

request_shutdown() {
    echo ""
    echo "[Launcher] Shutdown requested."
    exit 0
}

trap cleanup EXIT
trap request_shutdown INT TERM

if [[ ! -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ]]; then
    echo "[Error] ROS 2 setup not found: /opt/ros/$ROS_DISTRO_NAME/setup.bash" >&2
    exit 1
fi

if [[ ! -f "$YOLO_SCRIPT" ]]; then
    echo "[Error] YOLO script not found: $YOLO_SCRIPT" >&2
    echo "        Set YOLO_SCRIPT=/absolute/path/to/your_node.py" >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[Error] YOLO model not found: $MODEL_PATH" >&2
    echo "        Set MODEL_PATH=/absolute/path/to/yolov8n.pt" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[Error] conda is not available in PATH." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "[Error] conda.sh not found below: $CONDA_BASE" >&2
    exit 1
fi

echo "[Launcher] Starting YOLO..."
echo "           ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "           input        : $INPUT_TOPIC"
echo "           output       : $OUTPUT_TOPIC"

(
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    # shellcheck disable=SC1090
    source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
    cd "$(dirname "$YOLO_SCRIPT")"
    exec python -u "$YOLO_SCRIPT" \
        --model "$MODEL_PATH" \
        --device "$DEVICE" \
        --input-topic "$INPUT_TOPIC" \
        --imgsz "$IMAGE_SIZE"
) &
YOLO_PID=$!

echo "[Launcher] Waiting up to ${TOPIC_TIMEOUT}s for $OUTPUT_TOPIC ..."
topic_ready=0
for ((second = 1; second <= TOPIC_TIMEOUT; second++)); do
    if ! process_is_running "$YOLO_PID"; then
        echo "[Error] YOLO node exited before publishing its output." >&2
        wait "$YOLO_PID" || true
        exit 1
    fi

    if env ROS_DOMAIN_ID="$ROS_DOMAIN_ID" RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION" \
        bash --noprofile --norc -c \
        "source '/opt/ros/$ROS_DISTRO_NAME/setup.bash' && ros2 topic list 2>/dev/null" \
        | grep -Fxq "$OUTPUT_TOPIC"; then
        topic_ready=1
        break
    fi
    sleep 1
done

if (( topic_ready == 0 )); then
    echo "[Error] $OUTPUT_TOPIC did not appear within ${TOPIC_TIMEOUT}s." >&2
    echo "        Confirm Isaac Sim publishes $INPUT_TOPIC on ROS_DOMAIN_ID=$ROS_DOMAIN_ID." >&2
    exit 1
fi

if [[ "$RECORD_BAG" == "1" ]]; then
    mkdir -p "$BAG_ROOT"
    echo "[Launcher] Starting timestamp-aligned ROS bag..."
    echo "           sync : image header.stamp from Isaac camera"
    echo "           bag  : $BAG_PATH"

    env \
        -u CONDA_PREFIX \
        -u CONDA_DEFAULT_ENV \
        -u CONDA_PROMPT_MODIFIER \
        -u PYTHONPATH \
        ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
        RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION" \
        bash --noprofile --norc -c '
            source "/opt/ros/'"$ROS_DISTRO_NAME"'/setup.bash"
            exec ros2 bag record \
                --storage "'"$BAG_STORAGE"'" \
                --output "'"$BAG_PATH"'" \
                /clock \
                "'"$INPUT_TOPIC"'" \
                /front_camera/camera_info \
                "'"$OUTPUT_TOPIC"'" \
                /yolo/detections \
                /yolo/fps \
                /yolo/latency_ms \
                /yolo/object_count \
                /yolo/workload \
                /odom \
                /tf \
                /tf_static \
                /asyncmec/speed_limit \
                /asyncmec/scene_zone \
                /asyncmec/scene_object_count
        ' &
    BAG_PID=$!

    sleep 1
    if ! process_is_running "$BAG_PID"; then
        echo "[Error] rosbag recorder failed to start." >&2
        echo "        If MCAP is missing: sudo apt install ros-$ROS_DISTRO_NAME-rosbag2-storage-mcap" >&2
        wait "$BAG_PID" || true
        exit 1
    fi
fi

if [[ "$AUTO_STOP_ON_CLOCK" == "1" ]]; then
    echo "[Launcher] Starting Isaac Sim /clock watchdog..."
    echo "           first-clock timeout : ${CLOCK_START_TIMEOUT}s"
    echo "           stopped-clock delay : ${CLOCK_STOP_TIMEOUT}s"

    (
        unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONPATH
        export ROS_DOMAIN_ID RMW_IMPLEMENTATION
        # shellcheck disable=SC1090
        source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
        exec /usr/bin/python3 -u - \
            "$LAUNCHER_PID" \
            "$CLOCK_START_TIMEOUT" \
            "$CLOCK_STOP_TIMEOUT" <<'PYTHON'
import os
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

launcher_pid = int(sys.argv[1])
start_timeout = float(sys.argv[2])
stop_timeout = float(sys.argv[3])


class ClockWatchdog(Node):
    def __init__(self):
        super().__init__("asyncmec_clock_watchdog")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.started_at = time.monotonic()
        self.last_clock_at = None
        self.create_subscription(Clock, "/clock", self.on_clock, qos)

    def on_clock(self, _message):
        if self.last_clock_at is None:
            print("[Clock Watchdog] Isaac Sim /clock detected; watchdog armed.", flush=True)
        self.last_clock_at = time.monotonic()


rclpy.init()
node = ClockWatchdog()

try:
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.monotonic()

        if node.last_clock_at is None:
            if now - node.started_at > start_timeout:
                print(
                    f"[Clock Watchdog] No /clock received within {start_timeout:.1f}s; "
                    "stopping the launcher.",
                    flush=True,
                )
                os.kill(launcher_pid, signal.SIGTERM)
                break
        elif now - node.last_clock_at > stop_timeout:
            print(
                f"[Clock Watchdog] /clock stopped for {stop_timeout:.1f}s; "
                "stopping the AsyncMEC run.",
                flush=True,
            )
            os.kill(launcher_pid, signal.SIGTERM)
            break
finally:
    node.destroy_node()
    rclpy.shutdown()
PYTHON
    ) &
    CLOCK_WATCHDOG_PID=$!
fi

env \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    -u PYTHONPATH \
    ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
    RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION" \
    bash --noprofile --norc -c \
    "source '/opt/ros/$ROS_DISTRO_NAME/setup.bash'; exec ros2 run image_view image_view --ros-args -r image:='$OUTPUT_TOPIC'" &
VIEWER_PID=$!

echo ""
echo "[Launcher] Ready. Start Isaac Movie Capture now."
echo "           Stop the Isaac Sim timeline when capture finishes."
echo "           The launcher will stop automatically after ${CLOCK_STOP_TIMEOUT}s without /clock."
echo "           You can also press Ctrl+C to stop everything manually."

while true; do
    if ! process_is_running "$YOLO_PID"; then
        echo "[Error] YOLO node exited unexpectedly." >&2
        exit 1
    fi
    if [[ "$RECORD_BAG" == "1" ]] && ! process_is_running "$BAG_PID"; then
        echo "[Error] rosbag recorder exited unexpectedly." >&2
        exit 1
    fi
    if ! process_is_running "$VIEWER_PID"; then
        echo "[Launcher] Image viewer closed. Ending the run."
        break
    fi
    if [[ "$AUTO_STOP_ON_CLOCK" == "1" ]] && ! process_is_running "$CLOCK_WATCHDOG_PID"; then
        echo "[Error] Clock watchdog exited unexpectedly." >&2
        exit 1
    fi
    sleep 1
done