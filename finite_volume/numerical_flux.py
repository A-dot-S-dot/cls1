from abc import ABC, abstractmethod
from typing import Callable, Tuple, TypeVar

import core
import numpy as np

BENCHMARK = TypeVar("BENCHMARK", bound=core.Benchmark)


def get_required_values(
    input_dimension: int, *values: np.ndarray
) -> Tuple[np.ndarray, ...]:
    assert len(values) >= input_dimension, f"{input_dimension} inputs is required."
    assert len(values) % 2 == 0, "Even number of inputs is required."

    delta = (len(values) - input_dimension) // 2

    return values[delta : len(values) - delta]


def get_dof_vector(*values: np.ndarray) -> np.ndarray:
    return np.array([*values[0], *[v[-1] for v in values[1:]]])


class NumericalFlux(ABC):
    input_dimension: int

    @abstractmethod
    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


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
        return self._numerical_flux(*get_required_values(self.input_dimension, *values))


class LaxFriedrichsFlux(NumericalFlux):
    input_dimension = 2
    _riemann_solver: core.RiemannSolver

    def __init__(self, riemann_solver: core.RiemannSolver):
        self._riemann_solver = riemann_solver

    def __call__(self, value_left, value_right) -> Tuple[np.ndarray, np.ndarray]:
        flux, _ = self._riemann_solver.solve(value_left, value_right)

        return -flux, flux


class CentralFlux(NumericalFlux):
    input_dimension = 2
    _flux: Callable

    def __init__(self, flux: Callable):
        self._flux = flux

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        flux = (self._flux(value_left) + self._flux(value_right)) / 2

        return -flux, flux


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


class NumericalFluxDependentRightHandSide:
    _numerical_flux: NumericalFlux
    _step_length: float
    _boundary_conditions: core.BoundaryConditions
    _cell_left: core.GhostCell
    _cell_right: core.GhostCell

    def __init__(
        self,
        numerical_flux: NumericalFlux,
        step_length: float,
        boundary_conditions: core.BoundaryConditions,
    ):
        self._numerical_flux = numerical_flux
        self._step_length = step_length
        self._boundary_conditions = boundary_conditions

    def __call__(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        left_flux, right_flux = self._transform_node_to_cell_fluxes(
            *self._numerical_flux(
                *self._boundary_conditions.get_node_neighbours(
                    dof_vector,
                    radius=self._numerical_flux.input_dimension // 2,
                    time=time,
                )
            ),
        )

        return (left_flux + right_flux) / self._step_length

    def _transform_node_to_cell_fluxes(
        self,
        node_flux_left: np.ndarray,
        node_flux_right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return node_flux_right[:-1], node_flux_left[1:]

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(numerical_flux={self._numerical_flux})"
