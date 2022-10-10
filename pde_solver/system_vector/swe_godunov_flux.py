from typing import Tuple

import numpy as np
from pde_solver.solver_space import FiniteVolumeSpace
from pde_solver.system_vector import SystemVector


class SWEGodunovNumericalFlux(SystemVector):
    """Calculates the shallow-water Godunov numerical fluxes,
    i.e.

        FR_{i-1/2}, FL_{i+1/2}

    where i is the cell index.

    """

    volume_space: FiniteVolumeSpace
    gravitational_acceleration: float
    bottom_topography: np.ndarray
    eps = 1e-12

    left_intermediate_velocity: np.ndarray
    right_intermediate_velocity: np.ndarray

    _discrete_solution: np.ndarray

    # help attributes for for loop
    _left_cell_index: int
    _left_swe_values: np.ndarray
    _left_height: float
    _left_discharge: float
    _left_flux: np.ndarray
    _left_velocity: float
    _right_cell_index: int
    _right_swe_values: np.ndarray
    _right_height: float
    _right_discharge: float
    _right_flux: np.ndarray
    _right_velocity: float
    _topography_step: float

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        self.left_intermediate_velocity = np.zeros(self.volume_space.edge_number)
        self.right_intermediate_velocity = np.zeros(self.volume_space.edge_number)
        right_numerical_fluxes = np.empty((2, self.volume_space.dimension))
        left_numerical_fluxes = np.empty((2, self.volume_space.dimension))
        self._discrete_solution = dof_vector

        for edge_index in range(self.volume_space.edge_number):
            self._build_help_attributes(edge_index)

            left_cell_flux, right_cell_flux = self._calculate_cell_fluxes()

            left_numerical_fluxes[:, self._left_cell_index] = left_cell_flux
            right_numerical_fluxes[:, self._right_cell_index] = right_cell_flux

        return np.array([left_numerical_fluxes, right_numerical_fluxes])

    def _build_help_attributes(self, edge_index: int):
        (
            self._left_cell_index,
            self._right_cell_index,
        ) = self.volume_space.left_right_cell(edge_index)

        self._build_swe_values()
        self._build_fluxes()
        self._build_intermediate_velocities()
        self._update_intermediate_velocities(edge_index)

        self._build_topography_step()

    def _build_swe_values(self):
        left_swe_values, right_swe_values = (
            self._discrete_solution[:, self._left_cell_index],
            self._discrete_solution[:, self._right_cell_index],
        )

        if left_swe_values[0] < self.eps:
            self._left_swe_values = np.zeros(2)
        else:
            self._left_swe_values = left_swe_values

        if right_swe_values[0] < self.eps:
            self._right_swe_values = np.zeros(2)
        else:
            self._right_swe_values = right_swe_values

        self._left_height, self._left_discharge = self._left_swe_values
        self._right_height, self._right_discharge = self._right_swe_values

    def _build_fluxes(self):
        self._left_flux = self._flux(*self._left_swe_values)
        self._right_flux = self._flux(*self._right_swe_values)

    def _flux(self, height, discharge) -> np.ndarray:
        if height != 0:
            return np.array(
                [
                    discharge,
                    discharge**2 / height
                    + self.gravitational_acceleration * height**2 / 2,
                ]
            )
        else:
            return np.zeros(2)

    def _build_intermediate_velocities(self):
        left_height, u_left = self._get_height_and_velocity(*self._left_swe_values)
        right_height, u_right = self._get_height_and_velocity(*self._right_swe_values)

        self._left_velocity = min(
            u_left - np.sqrt(self.gravitational_acceleration * left_height),
            u_right - np.sqrt(self.gravitational_acceleration * right_height),
            0,
        )
        self._right_velocity = max(
            u_left + np.sqrt(self.gravitational_acceleration * left_height),
            u_right + np.sqrt(self.gravitational_acceleration * right_height),
            0,
        )

    def _get_height_and_velocity(
        self, height: float, discharge: float
    ) -> Tuple[float, float]:
        if height != 0:
            return (height, discharge / height)
        else:
            return (0, 0)

    def _update_intermediate_velocities(self, edge_index: int):
        self.left_intermediate_velocity[edge_index] = self._left_velocity
        self.right_intermediate_velocity[edge_index] = self._right_velocity

    def _build_topography_step(self):
        self._topography_step = (
            self.bottom_topography[self._right_cell_index]
            - self.bottom_topography[self._left_cell_index]
        )

    def _calculate_cell_fluxes(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._left_velocity < self.eps and self._right_velocity < self.eps:
            return (np.zeros(2), np.zeros(2))

        left_state, right_state = self._calculate_modified_intermediate_states()

        return (
            left_state - self._left_velocity * self._left_swe_values + self._left_flux,
            right_state
            - self._right_velocity * self._right_swe_values
            + self._right_flux,
        )

    def _calculate_modified_intermediate_states(self) -> Tuple[np.ndarray, np.ndarray]:
        (
            left_modified_height,
            right_modified_height,
        ) = self._calculate_modified_intermediate_height()

        discharge = self._calculate_intermediate_discharge()

        return (
            np.array([left_modified_height, self._left_velocity * discharge]),
            np.array([right_modified_height, self._right_velocity * discharge]),
        )

    def _calculate_modified_intermediate_height(self) -> Tuple[float, float]:
        # returns intermediate_velocity * intermediate_height. The last one is
        # modified to ensure positivity. The product is returned for avoiding
        # division with small numbers.

        height_HLL = self._calculate_height_HLL()

        left_height = (
            height_HLL
            + self._right_velocity
            / (self._right_velocity - self._left_velocity)
            * self._topography_step
        )
        right_height = (
            height_HLL
            + self._left_velocity
            / (self._right_velocity - self._left_velocity)
            * self._topography_step
        )

        if self._topography_step >= 0:
            modified_right_height = self._right_velocity * max(right_height, 0)

            return (
                self._left_velocity * left_height
                - self._right_velocity * right_height
                + modified_right_height,
                modified_right_height,
            )
        else:
            modified_left_height = self._left_velocity * max(left_height, 0)

            return (
                modified_left_height,
                self._right_velocity * right_height
                - self._left_velocity * left_height
                + modified_left_height,
            )

    def _calculate_height_HLL(self) -> float:
        return (
            self._right_velocity * self._right_height
            - self._left_velocity * self._left_height
            - self._right_discharge
            + self._left_discharge
        ) / (self._right_velocity - self._left_velocity)

    def _calculate_intermediate_discharge(self) -> float:
        discharge_HLL = self._calculate_discharge_HLL()
        step_length = self.volume_space.mesh.step_length
        source_term = self._calculate_source_term_approximation()

        return (
            discharge_HLL
            - self.gravitational_acceleration
            * step_length
            * source_term
            / (self._right_velocity - self._left_velocity)
        )

    def _calculate_discharge_HLL(self) -> float:
        return (
            self._right_velocity * self._right_flux[0]
            - self._left_velocity * self._left_flux[0]
            - self._right_flux[1]
            + self._left_flux[1]
        ) / (self._right_velocity - self._left_velocity)

    def _calculate_source_term_approximation(self) -> float:
        step_length = self.volume_space.mesh.step_length

        if self._topography_step >= 0:
            return (
                (self._left_height + self._right_height)
                / (2 * step_length)
                * min(self._left_height, self._topography_step)
            )
        else:
            return (
                (self._left_height + self._right_height)
                / (2 * step_length)
                * max(-self._right_height, self._topography_step)
            )
