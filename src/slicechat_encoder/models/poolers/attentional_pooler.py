import torch
import torch.nn as nn
from .base import BasePooler
from einops import rearrange
from torch.nn import functional as F


class AttentionalPoolerMLP(nn.Module):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AttentionalPooler(BasePooler):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072, num_heads: int = 12):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = AttentionalPoolerMLP(hidden_size, intermediate_size)


    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        q = self.q_proj(probe)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        
        # Get attention mask for scaled_dot_product_attention
        attention_mask = (padding_mask == 0)

        # The mask needs to be broadcastable to the attention weights.
        # Attention weights shape: (batch_size, num_heads, query_seq_len, key_seq_len)
        # Mask shape: (batch_size, key_seq_len) -> (batch_size, 1, 1, key_seq_len)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        hidden_states = rearrange(hidden_states, 'b h l d -> b l (h d)')

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states[:, 0]
