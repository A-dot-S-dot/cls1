from typing import Any, OrderedDict, Sequence, Tuple

import torch
from torch import nn


class Normalize(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class NeuralNetwork(nn.Sequential):
    normalize = Normalize()
    neurons: Sequence[int]

    def build_network(self, neurons: Sequence[int]):
        self._add_normalize()

        for layer in range(len(neurons) - 1):
            if layer > 0:
                self._add_activation_function(layer)

            self._add_cross(layer, neurons)

    def _add_normalize(self):
        self.add_module("Normalize", self.normalize)

    def _add_cross(self, layer: int, neurons: Sequence[int]):
        cross_name = f"Layer{layer}->Layer{layer+1}"
        self.add_module(cross_name, nn.Linear(neurons[layer], neurons[layer + 1]))

    def _add_dropout(self, layer: int, p: float):
        self.add_module(f"Layer{layer} Dropout", nn.Dropout(p))

    def _add_activation_function(self, layer: int):
        activ_fn_name = f"Layer{layer} Activation Function"
        self.add_module(activ_fn_name, nn.LeakyReLU())

    def setup_normalization(self, mean: Any, std: Any):
        self.normalize.mean = torch.tensor(mean)
        self.normalize.std = torch.tensor(std)

    def get_extra_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.normalize.mean, self.normalize.std

    def set_extra_state(self, state):
        self.normalize.mean = state[0]
        self.normalize.std = state[1]

    def load_state_dict(self, state: OrderedDict):
        neurons = self._get_neurons(state)
        self.build_network(neurons)

        super().load_state_dict(state)

    def _get_neurons(self, state: OrderedDict) -> Sequence[int]:
        neurons = []

        for key, values in state.items():
            if "weight" in key:
                neurons += self._get_weight_dependent_neurons(values, "0" in key)

        return neurons

    def _get_weight_dependent_neurons(
        self, weights: torch.Tensor, first_layer: bool
    ) -> Sequence[int]:
        right_neurons, left_neurons = weights.shape

        if first_layer:
            return [left_neurons, right_neurons]
        else:
            return [right_neurons]
