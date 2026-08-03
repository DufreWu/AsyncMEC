from robot_env import RobotEnv
from yolo_manager import YOLOManager
from video_player import VideoPlayer
import itertools
import time
import yaml
import pandas as pd
import numpy as np

# cpu_freqs: [422400, 576000, 729600, 883200, 1036800, 1190400, 1344000, 1497600, 1651200, 1804800, 1958400, 1984000]
# gpu_freqs: [306000000, 408000000, 510000000, 612000000, 714000000, 816000000, 918000000]

CPU_FREQS = [
    422400,
    729600,
    1036800,
    1344000,
    1651200,
    1984000
]

GPU_FREQS = [
    408000000,
    510000000,
    612000000,
    714000000,
    816000000,
    918000000
]

TEST_TIME = 10      # seconds

# ==========================================
# Load configs
# ==========================================
with open("configs/mission.yaml", "r") as f:
    mission_config = yaml.safe_load(f)

with open("configs/robot.yaml", "r") as f:
    robot_config = yaml.safe_load(f)

env = RobotEnv(robot_config, is_sim=False)
detector = YOLOManager(is_sim=False, robot=env, model_path="./checkpoints/yolov8n.pt")
detector.set_enabled(True)
player = VideoPlayer("videos/person_bicycle_car_detection.mp4")

results = []

for cpu, gpu in itertools.product(CPU_FREQS, GPU_FREQS):

    print("=" * 60)
    print(f"CPU={cpu} GPU={gpu}")

    time.sleep(2)

    total_energy = 0.0
    total_power = 0.0
    total_current = 0.0
    total_temp = 0.0

    fps_list = []
    latency_list = []
    power_list = []
    t0 = time.time()

    while time.time() - t0 < TEST_TIME:

        frame = player.get_frame(speed=1.0, control_period=1.0)
        result = detector.run(
            cpu_freq=cpu,
            gpu_freq=gpu,
            input_data=frame
        )

        fps_list.append(result["fps"])
        latency_list.append(result["latency_ms"])

        power = env.board.read_power()
        power_list.append(power)

    results.append({
        "cpu": cpu,
        "gpu": gpu,
        "fps": np.mean(fps_list),
        "latency": np.mean(latency_list),
        "power": np.mean(power_list),
    })
    print(f"cpu:{cpu} | gpu:{gpu} | fps:{np.mean(fps_list)} | power:{power}")

df = pd.DataFrame(results)

df.to_csv("logs/motivation_dvfs_tradeoff.csv", index=False)

print(df)