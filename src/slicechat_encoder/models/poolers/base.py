import torch
from abc import ABC
import torch.nn as nn

class BasePooler(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
