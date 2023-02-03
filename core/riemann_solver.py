from typing import Callable, Tuple

import numpy as np

from .finite_volume import BoundaryConditionsApplier, PeriodicBoundaryConditionsApplier

FLUX = Callable[[np.ndarray], np.ndarray]
WAVE_SPEED = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class RiemannSolver:
    _flux: FLUX
    _wave_speed: WAVE_SPEED
    _conditions_applier: BoundaryConditionsApplier

    _dof_vector_with_applied_boundary_conditions: np.ndarray
    _value_left: np.ndarray
    _value_right: np.ndarray
    _flux_left: np.ndarray
    _flux_right: np.ndarray
    _wave_speed_left: np.ndarray
    _wave_speed_right: np.ndarray

    def __init__(
        self,
        flux: FLUX,
        wave_speed: WAVE_SPEED,
        conditions_applier=None,
    ):
        self._flux = flux
        self._wave_speed = wave_speed

        self._conditions_applier = (
            conditions_applier or PeriodicBoundaryConditionsApplier()
        )
        assert self._conditions_applier.cells_to_add_numbers == (1, 1)

    def solve(self, time: float, dof_vector: np.ndarray):
        self._dof_vector_with_applied_boundary_conditions = (
            self._conditions_applier.add_conditions(time, dof_vector)
        )
        self._value_left, self._value_right = (
            self._dof_vector_with_applied_boundary_conditions[:-1],
            self._dof_vector_with_applied_boundary_conditions[1:],
        )
        self._flux_left, self._flux_right = self._flux(self._value_left), self._flux(
            self._value_right
        )
        self._wave_speed_left, self._wave_speed_right = self._wave_speed(
            self._value_left, self._value_right
        )

        factor = 1 / (self._wave_speed_right + -self._wave_speed_left)
        self._intermediate_state = factor * (
            self._wave_speed_right * self._value_right
            + -self._wave_speed_left * self._value_left
            - (self._flux_right + -self._flux_left)
        )

        self._flux_HLL = self._flux_right - self._wave_speed_right * (
            self._value_right - self._intermediate_state
        )

    @property
    def periodic_boundary_condition(self) -> bool:
        return isinstance(self._conditions_applier, PeriodicBoundaryConditionsApplier)

    @property
    def dof_vector_with_applied_boundary_conditions(self) -> np.ndarray:
        return self._dof_vector_with_applied_boundary_conditions

    @property
    def flux(self) -> FLUX:
        return self._flux

    @property
    def wave_speed(self) -> WAVE_SPEED:
        return self._wave_speed

    @property
    def value_left(self) -> np.ndarray:
        return self._value_left

    @property
    def value_right(self) -> np.ndarray:
        return self._value_right

    @property
    def flux_left(self) -> np.ndarray:
        return self._flux_left

    @property
    def flux_right(self) -> np.ndarray:
        return self._flux_right

    @property
    def wave_speed_left(self) -> np.ndarray:
        return self._wave_speed_left

    @property
    def wave_speed_right(self) -> np.ndarray:
        return self._wave_speed_right

    @property
    def intermediate_flux(self) -> np.ndarray:
        return self._flux_HLL

    @property
    def intermediate_state(self) -> np.ndarray:
        return self._intermediate_state
