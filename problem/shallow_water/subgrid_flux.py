from typing import Tuple

import numpy as np
import torch
from base import network
from base.discretization import DiscreteSolution
from base.numerical_flux import (
    NumericalFlux,
    NumericalFluxContainer,
    ObservedNumericalFlux,
)
from torch import nn


class ExactSubgridFlux(NumericalFlux):
    """Calculates exat subgrid flux for shallow water equations with flat
    bottom.

    """

    _fine_numerical_fluxes: NumericalFluxContainer
    _coarse_solution: DiscreteSolution
    _coarse_numerical_flux: ObservedNumericalFlux
    _coarsening_degree: int
    _time_index: int

    def __init__(
        self,
        fine_numerical_fluxes: NumericalFluxContainer,
        coarse_solution: DiscreteSolution,
        coarse_numerical_flux: ObservedNumericalFlux,
        coarsening_degree: int,
    ):
        self._fine_numerical_fluxes = fine_numerical_fluxes
        self._coarse_solution = coarse_solution
        self._coarse_numerical_flux = coarse_numerical_flux
        self._coarsening_degree = coarsening_degree
        self._time_index = 0

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_subgrid_flux = self._get_left_subgrid_flux()
        self._time_index += 1

        return left_subgrid_flux, np.roll(left_subgrid_flux, -1, axis=0)

    def _get_left_subgrid_flux(
        self,
    ) -> np.ndarray:
        left_coarse_flux = self._coarse_numerical_flux.left_numerical_flux
        left_fine_flux = self._fine_numerical_fluxes.left_numerical_fluxes[
            self._time_index
        ]

        return left_fine_flux[:: self._coarsening_degree] + -left_coarse_flux


class NetworkSubgridFlux(NumericalFlux):
    """Calculates subgrid flux for shallow water equations with flat bottom. To
    calculate a subgrid flux of a certain edge the Network needs a certain
    number of values of neighboured cells."""

    _network: nn.Module
    _local_degree: int
    _curvature: network.Curvature

    def __init__(
        self, network: nn.Module, curvature: network.Curvature, local_degree: int
    ):
        self._local_degree = local_degree
        self._curvature = curvature
        self._network = network

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input = self._get_input(dof_vector)
        left_subgrid_flux = self._network(input).detach().numpy()

        return left_subgrid_flux, np.roll(left_subgrid_flux, -1, axis=0)

    def _get_input(self, dof_vector: np.ndarray) -> torch.Tensor:
        stencil_values = np.array(
            [
                np.roll(dof_vector, i, axis=0)
                for i in range(self._local_degree, -self._local_degree, -1)
            ]
        )
        curvature = self._curvature(*stencil_values)

        return torch.Tensor(
            np.concatenate(
                [*stencil_values, curvature],
                axis=1,
            )
        )


class CorrectedNumericalFlux(NumericalFlux):
    """Adds to a given flux a subgrid flux."""

    _numerical_flux: NumericalFlux
    _subgrid_flux: NumericalFlux

    def __init__(self, numerical_flux: NumericalFlux, subgrid_flux: NumericalFlux):
        self._numerical_flux = numerical_flux
        self._subgrid_flux = subgrid_flux

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flux = self._numerical_flux(dof_vector)
        subgrid_flux = self._subgrid_flux(dof_vector)
        return flux[0] + subgrid_flux[0], flux[1] + subgrid_flux[1]
