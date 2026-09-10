import torch
from .base import BasePooler

class CLSPooler(BasePooler):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states[:, 0]
