import torch
import torch.nn as nn
import torch.nn.functional as F
from cross_attention import CrossAttention

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
        # s: [B, state_dim]
        return self.net(state)  # [B, hidden_dim]

# ============================================================
# Feature Adapter
# battery health feature -> controller hidden dimension
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
        # x: [B, adapter_dim]
        return self.adapter(x)  # [B, hidden_dim]

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
        super(FeedForwardNet, self).__init__()

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

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.gate = AdaptiveGatingBlock(embed_dim)
        self.ffn = FeedForwardNet(embed_dim, ffn_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, s, h_hat):
        """
        s:              [batch, 1, embed_dim], 
        h_hat:          [batch, 32, embed_dim]
        """

        self_out, _ = self.self_attn(s, s, s)
        s1 = self.norm1(s + self.dropout(self_out))

        cross_out, _ = self.cross_attn(q=s1, kv=h_hat)
        gated = self.gate(self_out, cross_out)

        s2 = self.norm2(s1 + self.dropout(gated))
        ff = self.ffn(s2)
        s3 = self.norm3(s2 + self.dropout(ff))

        return s3


class EnergyEfficientMultiScaleController(nn.Module):
    def __init__(
        self,
        state_dim,
        adapter_dim,
        hidden_dim=128,
        n_heads=4,
        num_decoder_layers=2,
        action_dim=3
    ):
        super().__init__()

        self.state_enc = RobotStateEncoder(state_dim, hidden_dim)

        self.decoders = nn.ModuleList([
            DecoderBlock(hidden_dim, n_heads)
            for _ in range(num_decoder_layers)
        ])

        self.feature_adapter = FeatureAdapter(adapter_dim, hidden_dim)

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, health_feature):
        """
        state:          [B, state_dim]      (high-frequency)
        health_feature: [B, adapter_dim]    (cached, low-frequency)
        """

        # ---- encode state ----
        s_local = self.state_enc(state)      # [B, D]
        q = s_local.unsqueeze(1)             # [B, 1, D]

        # ---- feature adapter ----
        h_hat = self.feature_adapter(health_feature)
        h_hat = h_hat.unsqueeze(1)

        # ---- transformer decoding ----
        for dec in self.decoders:
            q = dec(q, h_hat)

        h = q.squeeze(1)

        # ---- action ----
        action = self.policy(h)

        return action


# -------------------------
# Instantiate controller
# -------------------------
state_dim = 6        # example: battery_soc, soh, speed, cpu, gpu, temp
adapter_dim = 128    # must match health energy adapter
action_dim = 3       # e.g., linear_vel, angular_vel, dvfs

controller = EnergyEfficientMultiScaleController(
    state_dim=state_dim,        
    adapter_dim=adapter_dim,
    action_dim=action_dim        
).cuda().half()

controller.eval()

# cached health feature (1–5 Hz)
cached_energy_adapter = torch.tensor(
    [[
        0.12, -0.08, 0.31, -0.22, 0.05, 0.18, -0.11, 0.09,
        0.27, -0.14, 0.06, 0.21, -0.19, 0.02, 0.15, -0.07,
        # ... (repeat until 128 values total)
    ]],
    device="cuda",
    dtype=torch.float16
)

# If you want a quick synthetic placeholder instead:
cached_energy_adapter = torch.randn(
    1, 128,
    device="cuda",
    dtype=torch.float16
)

# current robot state at time t (100 Hz)
current_robot_state = torch.tensor(
    [[
        0.23,   # battery_soc   (23%)
        0.93,   # battery_soh   (healthy)
        0.80,   # speed         (fast)
        0.71,   # cpu_util
        0.65,   # gpu_util
        0.75    # temperature   (normalized, e.g. 75°C / 100)
    ]],
    device="cuda",
    dtype=torch.float16
)  # shape: [1, 6]


# high-frequency loop (e.g. 100 Hz)
with torch.no_grad():
    action = controller(
        state=current_robot_state,   # [1, state_dim]
        health_feature=cached_energy_adapter,
    )

print("Action:", action)
# send_to_robot(action)