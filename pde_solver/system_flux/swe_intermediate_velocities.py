from typing import Tuple

import numpy as np
from defaults import EPSILON, GRAVITATIONAL_ACCELERATION
from pde_solver.discrete_solution import DiscreteSolutionObserver
from pde_solver.solver_space import FiniteVolumeSpace


class SWEIntermediateVelocities(DiscreteSolutionObserver):
    volume_space: FiniteVolumeSpace
    left_velocities: np.ndarray
    right_velocities: np.ndarray
    gravitational_acceleration = GRAVITATIONAL_ACCELERATION

    def update(self):
        self.left_velocities = np.zeros(self.volume_space.edge_number)
        self.right_velocities = np.zeros(self.volume_space.edge_number)

        for edge_index in range(self.volume_space.edge_number):
            cell_indices = self.volume_space.left_right_cell(edge_index)
            swe_values = self._build_swe_values(*cell_indices)
            left_velocity, right_velocity = self._build_wave_velocities(*swe_values)

            self.left_velocities[cell_indices[0]] = left_velocity
            self.right_velocities[cell_indices[0]] = right_velocity

    def _build_swe_values(
        self, left_cell_index: int, right_cell_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_swe_values, right_swe_values = (
            self._discrete_solution.end_values[left_cell_index],
            self._discrete_solution.end_values[right_cell_index],
        )

        if left_swe_values[0] < EPSILON:
            left_swe_values = np.zeros(2)

        if right_swe_values[0] < EPSILON:
            right_swe_values = np.zeros(2)

        return left_swe_values, right_swe_values

    def _build_wave_velocities(
        self, left_swe_values: np.ndarray, right_swe_values: np.ndarray
    ) -> Tuple[float, float]:
        left_height, u_left = self._get_height_and_velocity(*left_swe_values)
        right_height, u_right = self._get_height_and_velocity(*right_swe_values)

        left_wave_velocity = min(
            u_left - np.sqrt(self.gravitational_acceleration * left_height),
            u_right - np.sqrt(self.gravitational_acceleration * right_height),
            0,
        )
        right_wave_velocity = max(
            u_left + np.sqrt(self.gravitational_acceleration * left_height),
            u_right + np.sqrt(self.gravitational_acceleration * right_height),
            0,
        )

        return left_wave_velocity, right_wave_velocity

    def _get_height_and_velocity(
        self, height: float, discharge: float
    ) -> Tuple[float, float]:
        if height != 0:
            return (height, discharge / height)
        else:
            return (0, 0)
