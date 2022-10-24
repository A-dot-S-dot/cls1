from typing import Tuple

import numpy as np
import torch
from pde_solver.solver_space import FiniteVolumeSpace
from torch import nn
from network import NeuralNetwork

from .system_flux import SystemFlux


class NetworkApproximatedFlatBottomSubgridFlux(SystemFlux):
    """Calculates subgrid flux for shallow water equations with flat bottom. To
    calculate a subgrid flux of a certain edge the Network needs a certain
    number of values of neighboured cells."""

    volume_space: FiniteVolumeSpace

    _model: nn.Module

    def __init__(self, network_path: str):
        self._model = NeuralNetwork()
        self._model.load_state_dict(torch.load(network_path))
        self._model.eval()

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input = self._get_input(dof_vector)
        subgrid_flux = self._model(input).detach().numpy()

        return subgrid_flux, np.roll(subgrid_flux, -1, axis=0)

    def _get_input(self, dof_vector: np.ndarray) -> torch.Tensor:
        return torch.Tensor(
            np.concatenate((np.roll(dof_vector, 1, axis=0), dof_vector), axis=1)
        )
