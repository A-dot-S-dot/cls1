from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
from core import finite_volume
import core


class NumericalFlux(ABC):
    input_dimension: int

    @abstractmethod
    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...


class NumericalFluxBuilder:
    @staticmethod
    def build_flux(benchmark: core.Benchmark, mesh: core.Mesh) -> NumericalFlux:
        raise NotImplementedError


class NumericalFluxWithHistory(NumericalFlux):
    """Saves numerical flux calculated once in a history."""

    flux_left_history: np.ndarray
    flux_right_history: np.ndarray

    _numerical_flux: NumericalFlux
    _update_history: Callable[[np.ndarray, np.ndarray], None]

    def __init__(self, numerical_flux: NumericalFlux):
        self._numerical_flux = numerical_flux
        self.input_dimension = numerical_flux.input_dimension
        self.flux_left_history = np.empty(0)
        self.flux_right_history = np.empty(0)

        self._update_history = self._initialize_history

    def _initialize_history(self, flux_left: np.ndarray, flux_right: np.ndarray):
        self.flux_left_history = np.array([flux_left])
        self.flux_right_history = np.array([flux_right])

        self._update_history = self._append_to_history

    def _append_to_history(self, flux_left: np.ndarray, flux_right: np.ndarray):
        self.flux_left_history = np.append(
            self.flux_left_history, np.array([flux_left.copy()]), axis=0
        )
        self.flux_right_history = np.append(
            self.flux_right_history, np.array([flux_right.copy()]), axis=0
        )

    def __call__(self, *values) -> Tuple[np.ndarray, np.ndarray]:
        flux_left, flux_right = self._numerical_flux(*values)
        self._update_history(flux_left, flux_right)

        return flux_left, flux_right


class NumericalFluxWithArbitraryInput(NumericalFlux):
    _numerical_flux: NumericalFlux

    def __init__(self, numerical_flux: NumericalFlux):
        self.input_dimension = numerical_flux.input_dimension
        self._numerical_flux = numerical_flux

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert (
            len(values) >= self.input_dimension
        ), f"{self.input_dimension} inputs are needed."
        assert len(values) % 2 == 0, "Even number of inputs needed."

        delta = (len(values) - self.input_dimension) // 2

        return self._numerical_flux(*values[delta : len(values) - delta])


class CorrectedNumericalFlux(NumericalFlux):
    """Adds to a given flux a subgrid flux."""

    _numerical_flux: NumericalFlux
    _flux_correction: NumericalFlux

    def __init__(self, numerical_flux: NumericalFlux, flux_correction: NumericalFlux):
        self._numerical_flux = NumericalFluxWithArbitraryInput(numerical_flux)
        self._flux_correction = NumericalFluxWithArbitraryInput(flux_correction)

        self.input_dimension = max(
            numerical_flux.input_dimension, flux_correction.input_dimension
        )

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flux_left, flux_right = self._numerical_flux(*values)
        flux_correction_left, flux_correction_right = self._flux_correction(*values)
        return flux_left + flux_correction_left, flux_right + flux_correction_right


class SubgridFlux(NumericalFlux):
    _fine_flux: NumericalFlux
    _coarse_flux: NumericalFlux
    _coarsener: core.VectorCoarsener

    def __init__(
        self,
        fine_flux: NumericalFlux,
        coarse_flux: NumericalFlux,
        coarsening_degree: int,
    ):
        assert (
            fine_flux.input_dimension == coarse_flux.input_dimension
        ), "Input dimension of FINE_FLUX and COARSE_FLUX must be identical."

        self.input_dimension = fine_flux.input_dimension
        self._fine_flux = fine_flux
        self._coarse_flux = coarse_flux
        self._coarsener = core.VectorCoarsener(coarsening_degree)

    def __call__(
        self, fine_values: Tuple[np.ndarray, ...], coarse_values: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = self._coarsener.coarsening_degree

        fine_flux_left, fine_flux_right = self._fine_flux(*fine_values)
        coarse_flux_left, coarse_flux_right = self._coarse_flux(*coarse_values)

        subgrid_flux_left = fine_flux_left[::N] + -coarse_flux_left
        subgrid_flux_right = fine_flux_right[N - 1 :: N] + -coarse_flux_right

        return subgrid_flux_left, subgrid_flux_right


class LinearAntidiffusiveFlux(NumericalFlux):
    input_dimension = 2

    _gamma: float
    _step_length: float

    def __init__(self, gamma: float, step_length: float):
        self._gamma = gamma
        self._step_length = step_length

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        flux = self._gamma * (value_right + -value_left) / self._step_length

        return -flux, flux


class LaxFriedrichsFlux(NumericalFlux):
    input_dimension = 2
    riemann_solver: core.RiemannSolver

    def __init__(self, riemann_solver: core.RiemannSolver):
        self.riemann_solver = riemann_solver

    def __call__(self, value_left, value_right) -> Tuple[np.ndarray, np.ndarray]:
        _, flux = self.riemann_solver.solve(value_left, value_right)

        return -flux, flux


class CentralFlux(NumericalFlux):
    input_dimension = 2
    _flux: Callable

    def __init__(self, flux: Callable):
        self._flux = flux

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        node_flux = (self._flux(value_left) + self._flux(value_right)) / 2

        return -node_flux, node_flux


class NumericalFluxDependentRightHandSide:
    _numerical_flux: NumericalFlux
    _step_length: float
    _neighbours: finite_volume.NodeNeighbours

    def __init__(
        self,
        numerical_flux: NumericalFlux,
        step_length: float,
        neighbours: finite_volume.NodeNeighbours,
    ):
        self._numerical_flux = numerical_flux
        self._step_length = step_length
        self._neighbours = neighbours

        self._adjust_neighbours()

    def _adjust_neighbours(self):
        if not self._neighbours.periodic:
            raise NotImplementedError(
                "Must be shorten such that fluxes are calculated only for the inner nodes. "
            )

    def __call__(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        left_flux, right_flux = self._transform_node_to_cell_fluxes(
            time, *self._numerical_flux(*self._neighbours(dof_vector))
        )

        return (left_flux + right_flux) / self._step_length

    def _transform_node_to_cell_fluxes(
        self, time: float, node_flux_left: np.ndarray, node_flux_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # print(node_flux_left, node_flux_right)
        if self._neighbours.periodic:
            return node_flux_right[:-1], node_flux_left[1:]
        else:
            raise NotImplementedError(
                "Also Weak boundary conditions must be implemented with a Riemann Solver."
            )
