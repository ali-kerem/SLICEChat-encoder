import torch
from typing import Optional
from .base import BasePooler

class AveragePooler(BasePooler):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if padding_mask is None:
            return hidden_states.mean(dim=1)
        else:
            # padding_mask: (B, L) where 1 = padded, 0 = valid
            valid = (padding_mask == 0).to(dtype=hidden_states.dtype)  # (B, L)
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
            return (hidden_states * valid.unsqueeze(-1)).sum(dim=1) / denom
