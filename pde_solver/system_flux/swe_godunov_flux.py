from typing import Tuple

import numpy as np
from defaults import EPSILON, GRAVITATIONAL_ACCELERATION
from pde_solver.solver_space import FiniteVolumeSpace
from .system_flux import SystemFlux
from .swe_intermediate_velocities import SWEIntermediateVelocities
from .source_term_discretization import SourceTermDiscretization


class GodunovCellFluxesCalculator:
    dof_vector: np.ndarray
    gravitational_acceleration = GRAVITATIONAL_ACCELERATION
    bottom_topography: np.ndarray
    volume_space: FiniteVolumeSpace
    intermediate_velocities: SWEIntermediateVelocities
    source_term_discretization: SourceTermDiscretization

    cell_indices: Tuple[int, int]
    swe_values: Tuple[np.ndarray, np.ndarray]
    heights: Tuple[float, float]
    discharges: Tuple[float, float]
    fluxes: Tuple[np.ndarray, np.ndarray]
    wave_velocities: Tuple[float, float]
    cell_fluxes: Tuple[np.ndarray, np.ndarray]
    topography_step: float

    def setup(self, edge_index: int):
        self.cell_indices = self.volume_space.left_right_cell(edge_index)

        self._build_swe_values()
        self._build_fluxes()
        self._build_wave_velocities(edge_index)
        self._build_topography_step()
        self._build_cell_fluxes()

    def _build_swe_values(self):
        left_swe_values, right_swe_values = (
            self.dof_vector[self.cell_indices[0]],
            self.dof_vector[self.cell_indices[1]],
        )

        if left_swe_values[0] < EPSILON:
            left_swe_values = np.zeros(2)

        if right_swe_values[0] < EPSILON:
            right_swe_values = np.zeros(2)

        self.swe_values = (left_swe_values, right_swe_values)
        self.heights = (left_swe_values[0], right_swe_values[0])
        self.discharges = (left_swe_values[1], right_swe_values[1])

    def _build_fluxes(self):
        self.fluxes = (self._flux(*self.swe_values[0]), self._flux(*self.swe_values[1]))

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

    def _build_wave_velocities(self, edge_index: int):
        self.wave_velocities = (
            self.intermediate_velocities.left_velocities[edge_index],
            self.intermediate_velocities.right_velocities[edge_index],
        )

    def _build_topography_step(self):
        self.topography_step = (
            self.bottom_topography[self.cell_indices[1]]
            - self.bottom_topography[self.cell_indices[0]]
        )

    def _build_cell_fluxes(self):
        if self.wave_velocities[0] < EPSILON and self.wave_velocities[1] < EPSILON:
            return (np.zeros(2), np.zeros(2))

        left_state, right_state = self._calculate_modified_intermediate_states()

        self.cell_fluxes = (
            left_state - self.wave_velocities[0] * self.swe_values[0] + self.fluxes[0],
            right_state - self.wave_velocities[1] * self.swe_values[1] + self.fluxes[1],
        )

    def _calculate_modified_intermediate_states(self) -> Tuple[np.ndarray, np.ndarray]:
        (
            left_modified_height,
            right_modified_height,
        ) = self._calculate_modified_intermediate_height()

        discharge = self._calculate_intermediate_discharge()

        return (
            np.array([left_modified_height, self.wave_velocities[0] * discharge]),
            np.array([right_modified_height, self.wave_velocities[1] * discharge]),
        )

    def _calculate_modified_intermediate_height(self) -> Tuple[float, float]:
        # returns intermediate_velocity * intermediate_height. The last one is
        # modified to ensure positivity. The product is returned for avoiding
        # division with small numbers.

        height_HLL = self._calculate_height_HLL()

        left_height = (
            height_HLL
            + self.wave_velocities[1]
            / (self.wave_velocities[1] - self.wave_velocities[0])
            * self.topography_step
        )
        right_height = (
            height_HLL
            + self.wave_velocities[0]
            / (self.wave_velocities[1] - self.wave_velocities[0])
            * self.topography_step
        )

        if self.topography_step >= 0:
            modified_right_height = self.wave_velocities[1] * max(right_height, 0)

            return (
                self.wave_velocities[0] * left_height
                - self.wave_velocities[1] * right_height
                + modified_right_height,
                modified_right_height,
            )
        else:
            modified_left_height = self.wave_velocities[0] * max(left_height, 0)

            return (
                modified_left_height,
                self.wave_velocities[1] * right_height
                - self.wave_velocities[0] * left_height
                + modified_left_height,
            )

    def _calculate_height_HLL(self) -> float:
        return (
            self.wave_velocities[1] * self.heights[1]
            - self.wave_velocities[0] * self.heights[0]
            - self.discharges[1]
            + self.discharges[0]
        ) / (self.wave_velocities[1] - self.wave_velocities[0])

    def _calculate_intermediate_discharge(self) -> float:
        discharge_HLL = self._calculate_discharge_HLL()
        step_length = self.volume_space.mesh.step_length
        source_term = self.source_term_discretization(
            self.heights[0], self.heights[1], self.topography_step
        )

        return (
            discharge_HLL
            - self.gravitational_acceleration
            * step_length
            * source_term
            / (self.wave_velocities[1] - self.wave_velocities[0])
        )

    def _calculate_discharge_HLL(self) -> float:
        return (
            self.wave_velocities[1] * self.fluxes[1][0]
            - self.wave_velocities[0] * self.fluxes[0][0]
            - self.fluxes[1][1]
            + self.fluxes[0][1]
        ) / (self.wave_velocities[1] - self.wave_velocities[0])


class SWEGodunovNumericalFlux(SystemFlux):
    """Calculates the shallow-water Godunov numerical fluxes,
    i.e.

        FR_{i-1/2}, FL_{i+1/2}

    where i is the cell index.

    """

    volume_space: FiniteVolumeSpace
    cell_flux_calculator: GodunovCellFluxesCalculator

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.left_intermediate_velocity = np.zeros(self.volume_space.edge_number)
        self.right_intermediate_velocity = np.zeros(self.volume_space.edge_number)
        right_numerical_fluxes = np.empty((self.volume_space.dimension, 2))
        left_numerical_fluxes = np.empty((self.volume_space.dimension, 2))
        self.cell_flux_calculator.dof_vector = dof_vector

        for edge_index in range(self.volume_space.edge_number):
            self.cell_flux_calculator.setup(edge_index)
            left_cell_index, right_cell_index = self.cell_flux_calculator.cell_indices
            left_cell_flux, right_cell_flux = self.cell_flux_calculator.cell_fluxes

            left_numerical_fluxes[left_cell_index] = left_cell_flux
            right_numerical_fluxes[right_cell_index] = right_cell_flux

            self._update_intermediate_velocities(edge_index)

        return left_numerical_fluxes, right_numerical_fluxes

    def _update_intermediate_velocities(self, edge_index: int):
        self.left_intermediate_velocity[
            edge_index
        ] = self.cell_flux_calculator.wave_velocities[0]
        self.right_intermediate_velocity[
            edge_index
        ] = self.cell_flux_calculator.wave_velocities[1]
