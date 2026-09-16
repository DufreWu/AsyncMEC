import os
import yaml
import numpy as np
import random
import cv2

from robot_env import RobotEnv
from yolo_manager import YOLOManager
from rrl_control import RRLController


# =====================================================
# Config
# =====================================================

CONFIG = "configs/robot.yaml"

TARGET_FPS = 30
NUM_EPISODES = 1000
MAX_STEPS = 200
MODEL_DIR = "checkpoints"

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
# Environment
# =====================================================

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

robot = RobotEnv(
    cfg,
    is_sim=False
)

yolo = YOLOManager(
    is_sim=False,
    robot=robot
)
yolo.set_enabled(True)


# =====================================================
# RL
# =====================================================

controller = RRLController(robot)

# =====================================================
# Random scene complexity
# =====================================================

MISSION = [
    {
        "name": "low",
        "video": "videos/people_detection.mp4",
        "frame_size": 416,
        "target_fps": 30,
        "duration": 20
    },
    {
        "name": "medium",
        "video": "videos/bottle_detection.mp4",
        "frame_size": 640,
        "target_fps": 30,
        "duration": 20
    },
    {
        "name": "high",
        "video": "videos/fruit_and_vegetable_detection.mp4",
        "frame_size": 960,
        "target_fps": 30,
        "duration": 20
    },
]

def sample_random_frame():
    """Randomly choose a mission and return one random frame."""

    mission = random.choice(MISSION)
    cap = cv2.VideoCapture(mission["video"])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = random.randint(0, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Cannot read frame {frame_id} from {mission['video']}"
        )

    return {
        "frame": frame,
        "frame_size": mission["frame_size"],
        "complexity": mission["name"]
    }


# =====================================================
# Reset
# =====================================================

def reset():
    robot.battery.set_initial_soh(1.0)
    robot.set_speed(
        robot.motor.max_speed
    )

    for i in range(8):
        robot.set_board_cpu_dvfs(
            i,
            robot.board.cpu_freqs[0]
        )

    robot.set_board_gpu_dvfs(
        robot.board.gpu_freqs[0]
    )

    state = np.array([
        0,
        0,
        robot.board.cpu_freqs[0] / robot.board.cpu_freqs[-1],
        robot.board.gpu_freqs[0] / robot.board.gpu_freqs[-1],
        0.5 / robot.get_speed(),
        1.0,
    ], dtype=np.float32)

    return state


# =====================================================
# Training
# =====================================================

best_reward = -1e9

for episode in range(NUM_EPISODES):

    state = reset()
    total_reward = 0

    sample = random.choice(MISSION)
    cap = cv2.VideoCapture(sample["video"])

    for step in range(MAX_STEPS):

        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        yolo_data = yolo.run(
            cpu_freq=robot.board.read_cpu_freq(0),
            gpu_freq=robot.board.read_gpu_freq(),
            input_data=frame,
            frame_size=sample["frame_size"],
            complexity=sample["name"]
        )

        action, state, control = controller.step(yolo_data)

        next_state, reward, done, info = controller.execute_action(control, yolo_data)
        print(f"control: {control}, next_state: {next_state}")

        controller.agent.store(state, action, reward, next_state, done)
        controller.agent.learn()

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"episode: {episode}, reward: {total_reward/MAX_STEPS:.2f}, fps: {info['fps']:.2f}, epsilon: {controller.agent.epsilon:.3f}")

    if total_reward > best_reward:
        best_reward = total_reward
        controller.agent.save(
            os.path.join(
                MODEL_DIR,
                "rrl_model.pt"
            )
        )

cap.release()
print("Training Finished")
print("Best Reward =", best_reward/MAX_STEPS)