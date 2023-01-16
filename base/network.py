from typing import Tuple

import torch
from torch import nn


class Curvature:
    step_length: float

    def __init__(self, step_length: float):
        self.step_length = step_length

    def __call__(self, u0, u1, u2, u3):
        return (
            self._calculate_curvature(u0, u1, u2)
            + self._calculate_curvature(u1, u2, u3)
        ) / 2

    def _calculate_curvature(self, u0, u1, u2):
        return (
            abs(u0 - 2 * u1 + u2)
            * self.step_length
            / (self.step_length**2 + 0.25 * (u0 - u2) ** 2) ** (3 / 2)
        )


class Normalization(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean=None, std=None):
        nn.Module.__init__(self)
        self.mean = mean if mean is not None else torch.empty(0)
        self.std = std if std is not None else torch.empty(0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def get_extra_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mean, self.std

    def set_extra_state(self, state):
        self.mean, self.std = state

    def extra_repr(self) -> str:
        return f"mean={self.mean}, std={self.std}"


class NeuralNetwork(nn.Module):
    def __init__(self, mean=None, std=None):
        nn.Module.__init__(self)

        self.normalize = Normalization(mean, std)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        x = self.normalize(x)

        return self.linear_relu_stack(x)
