import os
import numpy as np
import torch
import torch.nn as nn
from tensorflow import keras
import joblib

# ============================================================
# Device
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)

    def forward(self, q, kv):
        """
        q: [batch_size, seq_len_q, embed_dim]
        kv: [batch_size, seq_len_kv, embed_dim]
        """

        out, attn = self.attn(
            query=q, 
            key=kv,
            value=kv
        )

        return out, attn

# ============================================================
# Robot State Encoder
# ============================================================
class RobotStateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state):
        return self.net(state)

# ============================================================
# Feature Adapter
# ============================================================
class FeatureAdapter(nn.Module):
    def __init__(self, adapter_dim, hidden_dim):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(adapter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.adapter(x)

# ============================================================
# Adaptive Gate
# ============================================================
class AdaptiveGatingBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, self_out, cross_out):

        alpha = self.gate(cross_out)

        fused = (
            alpha * cross_out
            + (1 - alpha) * self_out
        )

        return fused

# ============================================================
# Feed Forward Network
# ============================================================
class FeedForwardNet(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# Decoder Block
# ============================================================
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim=128, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True
        )

        self.cross_attn = CrossAttention(
            embed_dim,
            num_heads,
            dropout
        )

        self.gate = AdaptiveGatingBlock(embed_dim)

        self.ffn = FeedForwardNet(
            embed_dim,
            ffn_dim,
            dropout
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, s, h_hat):

        # ------------------------------------
        # Self attention
        # ------------------------------------
        self_out, _ = self.self_attn(s, s, s)

        s1 = self.norm1(s + self.dropout(self_out))

        # ------------------------------------
        # Cross attention
        # ------------------------------------
        cross_out, _ = self.cross_attn(q=s1, kv=h_hat)

        # ------------------------------------
        # Adaptive gating
        # ------------------------------------
        gated = self.gate(self_out, cross_out)

        s2 = self.norm2(s1 + self.dropout(gated))

        # ------------------------------------
        # FFN
        # ------------------------------------
        ff = self.ffn(s2)

        s3 = self.norm3(s2 + self.dropout(ff))

        return s3

# ============================================================
# Multi-scale Controller
# ============================================================
class EnergyEfficientMultiScaleController(nn.Module):

    def __init__(
        self,
        state_dim,
        adapter_dim=256,
        hidden_dim=128,
        n_heads=4,
        num_decoder_layers=2,
        action_dim=3
    ):
        super().__init__()

        # ------------------------------------
        # Robot state encoder
        # ------------------------------------
        self.state_enc = RobotStateEncoder(state_dim, hidden_dim)

        # ------------------------------------
        # Battery health adapter
        # ------------------------------------
        self.feature_adapter = FeatureAdapter(adapter_dim, hidden_dim)

        # ------------------------------------
        # Decoder stack
        # ------------------------------------
        self.decoders = nn.ModuleList([
            DecoderBlock(hidden_dim, n_heads)
            for _ in range(num_decoder_layers)
        ])

        # ------------------------------------
        # Policy head
        # ------------------------------------
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, health_feature):
        # ------------------------------------
        # Encode robot state
        # ------------------------------------
        s_local = self.state_enc(state)

        q = s_local.unsqueeze(1)

        # ------------------------------------
        # Adapt health feature
        # ------------------------------------
        h_hat = self.feature_adapter(health_feature)

        h_hat = h_hat.unsqueeze(1)

        # ------------------------------------
        # Multi-scale decoding
        # ------------------------------------
        for dec in self.decoders:
            q = dec(q, h_hat)

        h = q.squeeze(1)

        # ------------------------------------
        # Policy
        # ------------------------------------
        action = self.policy(h)

        return action


# =====================================================
# Example
# =====================================================
import yaml
import torch
import numpy as np

from robot_env import RobotEnv

# =====================================================
# Load configuration
# =====================================================
with open("./configs/robot.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Create environment
# =====================================================
env = RobotEnv(cfg, is_sim=True)

# or
# env.battery.set_initial_soh(0.8)

# =====================================================
# Build model
# =====================================================
model = EnergyEfficientMultiScaleController(
    state_dim=cfg["controller"]["state_dim"],
    adapter_dim=32,
    action_dim=cfg["controller"]["action_dim"],
).to(device)

model.eval()

# --------------------------------------
# Robot state
# --------------------------------------
robot_state = torch.tensor(
    [[
        12.3,      # mechanical power
        7.8,       # computation power
        0.8,       # speed
        30.0,      # FPS
        0.9,       # QoS
        1.0        # complexity 
    ]],
    dtype=torch.float32,
    device=device,
)

health_feature = torch.randn (
    1,
    cfg["battery"]["encoder_dim"],
    device=device,
)

with torch.no_grad():
    action = model(
        state = robot_state,
        health_feature = health_feature,
    )

print("=" * 50)
print("Action shape :", action.shape)
print("Action value :")
print(action.cpu().numpy())
print("=" * 50)

assert action.shape == (1, 3)

print("✓ Forward pass successful.")