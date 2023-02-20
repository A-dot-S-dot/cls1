from typing import Tuple, Dict

import defaults

import core
import defaults
import lib
import numpy as np
import shallow_water as swe
import torch
from torch import nn

from . import low_order
from .solver import ShallowWaterSolver


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
    input_dimension = 10

    def __init__(self, mean=None, std=None):
        nn.Module.__init__(self)

        self.normalize = Normalization(mean, std)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_dimension, 32),
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


class NetworkSubgridFlux(lib.NumericalFlux):
    """Calculates subgrid flux for shallow water equations with flat bottom. To
    calculate a subgrid flux of a certain edge the Network needs a certain
    number of values of neighboured cells. Requires periodic boundaries."""

    _network: nn.Module
    _curvature: Curvature
    _input_radius: int

    def __init__(
        self,
        network: NeuralNetwork,
        network_path: str,
        curvature: Curvature,
    ):
        self.input_dimension = network.input_dimension - 2
        self._input_radius = self.input_dimension // 2

        self._curvature = curvature
        self._network = network

        self._network.load_state_dict(torch.load(network_path))
        self._network.eval()

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input = self._get_input(*values)
        subgrid_flux = self._network(input).detach().numpy()

        return lib.transform_node_to_cell_flux(subgrid_flux)

    def _get_input(self, *values: np.ndarray) -> torch.Tensor:
        curvature = self._curvature(*values)
        network_input = np.concatenate(
            [*values, curvature],
            axis=1,
        )[: -2 * self._input_radius + 1]

        return torch.Tensor(network_input)


class SubgridNetworkSolver(ShallowWaterSolver):
    _network_path: str

    def _build_args(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        network_path=None,
        mesh_size=None,
        cfl_number=None,
        **kwargs,
    ) -> Dict:
        mesh_size = (
            mesh_size or defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
        )
        cfl_number = (
            cfl_number or defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE
        )
        self._network_path = network_path or defaults.NETWORK_PATH
        return super()._build_args(benchmark, kwargs)

    def _get_flux(
        self, benchmark: swe.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        numerical_flux = low_order.get_low_order_flux(benchmark, mesh)

        subgrid_flux = NetworkSubgridFlux(
            NeuralNetwork(),
            self._network_path,
            Curvature(mesh.step_length),
        )
        return lib.CorrectedNumericalFlux(numerical_flux, subgrid_flux)
