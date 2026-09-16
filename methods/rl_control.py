import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================
# Action Table
# 6 CPU × 6 GPU × 6 Speed = 216 actions
# ==========================================================

class DQNController:
    # =========================================================
    # Replay Buffer
    # =========================================================
    class ReplayBuffer:

        def __init__(self, capacity=50000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):

            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = zip(*batch)

            return (
                np.array(state, dtype=np.float32),
                np.array(action),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32)
            )

        def __len__(self):
            return len(self.buffer)


    # =========================================================
    # Q Network
    # =========================================================
    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

        def forward(self, x):
            return self.net(x)


    # =========================================================
    # DQN Agent
    # =========================================================
    class DQNAgent:
        def __init__(
            self,
            state_dim,
            action_dim,
            lr=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
            batch_size=64,
            target_update=200
        ):

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.action_dim = action_dim
            self.gamma = gamma
            self.batch_size = batch_size
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.target_update = target_update
            self.tau = 0.005
            self.learn_step = 0
            self.policy_net = DQNController.QNetwork(state_dim, action_dim).to(self.device)
            self.target_net = DQNController.QNetwork(state_dim, action_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.memory = DQNController.ReplayBuffer()

        # =====================================================
        # Epsilon Greedy
        # =====================================================
        def select_action(self, state):
            if random.random() < self.epsilon:
                return random.randrange(self.action_dim)

            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q = self.policy_net(state)

            return q.argmax().item()

        # =====================================================
        # Store
        # =====================================================
        def store(self, state, action, reward, next_state, done):
            self.memory.push(state, action, reward, next_state, done)

        # =====================================================
        # Learn
        # =====================================================
        def learn(self):
            if len(self.memory) < self.batch_size:
                return

            
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

            states = torch.tensor(states, device=self.device)
            actions = torch.tensor(actions, device=self.device).unsqueeze(1)
            rewards = torch.tensor(rewards, device=self.device).unsqueeze(1)
            next_states = torch.tensor(next_states, device=self.device)
            dones = torch.tensor(dones, device=self.device).unsqueeze(1)

            current_q = self.policy_net(states).gather(1, actions)

            # Target Q-values calculated using target network
            with torch.no_grad():
                max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
                target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

            loss = nn.MSELoss()(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Soft update target network parameters
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

            # Decay Epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        def save(self, path):
            torch.save(self.policy_net.state_dict(), path)
        
        def load(self, path):
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def __init__(self, robot):
        STATE_DIM = 6
        ACTION_DIM = 216

        self.robot = robot
        self.agent = self.DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

        self.cpu_freqs = robot.board.cpu_freqs
        self.gpu_freqs = robot.board.gpu_freqs

        self.speed_levels = np.linspace(robot.motor.min_speed, robot.motor.max_speed, 6)

        # Generate 216 Discrete Action Combinations
        self.action_table = [
            (c, g, s)
            for c in range(len(self.cpu_freqs))
            for g in range(len(self.gpu_freqs))
            for s in range(len(self.speed_levels))
        ]

    # ======================================================
    # Build RL state
    # ======================================================
    def build_state(self, yolo_data):
        cpu_norm = self.robot.fc_list[0] / self.cpu_freqs[-1]
        gpu_norm = self.robot.fg / self.gpu_freqs[-1]
        speed_norm = self.robot.get_speed() / self.robot.motor.max_speed

        state = np.array([
            yolo_data["fps"] / yolo_data["target_fps"],
            yolo_data["target_fps"] / yolo_data["target_fps"],
            cpu_norm,
            gpu_norm,
            speed_norm,
            yolo_data["complexity_id"],
        ], dtype=np.float32)

        return state

    # ======================================================
    # Choose action
    # ======================================================

    def step(self, yolo_data):
        state = self.build_state(yolo_data)
        action = self.agent.select_action(state)
        cpu_idx, gpu_idx, speed_idx = self.action_table[action]

        control = {
            "cpu_freq": self.cpu_freqs[cpu_idx],
            "gpu_freq": self.gpu_freqs[gpu_idx],
            "speed": float(self.speed_levels[speed_idx])
        }

        return control

    # ======================================================
    # Environment transition
    # ======================================================

    def execute_action(self, control, yolo_data):
        robot_state = self.robot.step(control)
        next_state = self.build_state(yolo_data)

        p_comp = robot_state["p_comp"]
        p_mech = robot_state["p_mech"]
        speed = max(self.robot.get_speed(), 0.1)

        energy_per_dist = (p_comp + p_mech) / speed

        fps_reward = 10.0
        if yolo_data["fps"] < yolo_data["target_fps"]:
            fps_reward = yolo_data["fps"] - yolo_data["target_fps"]

        reward = fps_reward - 0.1 * energy_per_dist

        done = 0.0

        info = {
            "fps": yolo_data["fps"],
            "energy_per_dist": energy_per_dist,
            "reward": reward
        }

        return next_state, reward, done, info