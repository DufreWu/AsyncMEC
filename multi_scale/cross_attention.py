import torch
import torch.nn as nn

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
