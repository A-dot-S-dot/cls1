from typing import Callable, Tuple

import numpy as np
from pde_solver.discretization import DiscreteSolution
from pde_solver.discretization.finite_volume import FiniteVolumeSpace

from .numerical_flux import NumericalFlux


class ShallowWaterIntermediateVelocities:
    left_velocities: np.ndarray
    right_velocities: np.ndarray

    _last_dof_vector: np.ndarray
    _volume_space: FiniteVolumeSpace
    _gravitational_acceleration: float
    _epsilon = 1e-12

    def __init__(
        self, volume_space: FiniteVolumeSpace, gravitational_acceleration: float
    ):
        self._last_dof_vector = np.empty(1)
        self._volume_space = volume_space
        self._gravitational_acceleration = gravitational_acceleration

    def update(self, dof_vector: np.ndarray):
        if (dof_vector != self._last_dof_vector).any():
            self.left_velocities = np.zeros(self._volume_space.node_number)
            self.right_velocities = np.zeros(self._volume_space.node_number)

            for edge_index in range(self._volume_space.node_number):
                cell_indices = self._volume_space.left_right_cell(edge_index)
                swe_values = self._build_swe_values(dof_vector, *cell_indices)
                left_velocity, right_velocity = self._build_wave_velocities(*swe_values)

                self.left_velocities[cell_indices[0]] = left_velocity
                self.right_velocities[cell_indices[0]] = right_velocity

            self._last_dof_vector = dof_vector.copy()

    def _build_swe_values(
        self, dof_vector: np.ndarray, left_cell_index: int, right_cell_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_swe_values, right_swe_values = (
            dof_vector[left_cell_index],
            dof_vector[right_cell_index],
        )

        if left_swe_values[0] < self._epsilon:
            left_swe_values = np.zeros(2)

        if right_swe_values[0] < self._epsilon:
            right_swe_values = np.zeros(2)

        return left_swe_values, right_swe_values

    def _build_wave_velocities(
        self, left_swe_values: np.ndarray, right_swe_values: np.ndarray
    ) -> Tuple[float, float]:
        left_height, u_left = self._get_height_and_velocity(*left_swe_values)
        right_height, u_right = self._get_height_and_velocity(*right_swe_values)

        left_wave_velocity = min(
            u_left - np.sqrt(self._gravitational_acceleration * left_height),
            u_right - np.sqrt(self._gravitational_acceleration * right_height),
            0,
        )
        right_wave_velocity = max(
            u_left + np.sqrt(self._gravitational_acceleration * left_height),
            u_right + np.sqrt(self._gravitational_acceleration * right_height),
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


class ShallowWaterGodunovNodeFluxesCalculator:
    """Calculates the left and right fluxes of an edge."""

    _volume_space: FiniteVolumeSpace
    _intermediate_velocities: ShallowWaterIntermediateVelocities
    _gravitational_acceleration: float
    _bottom_topography: np.ndarray
    _source_term_function: Callable[[float, float, float, float], float]
    _epsilon = 1e-12

    _cell_indices: Tuple[int, int]
    _swe_values: Tuple[np.ndarray, np.ndarray]
    _heights: Tuple[float, float]
    _discharges: Tuple[float, float]
    _fluxes: Tuple[np.ndarray, np.ndarray]
    _wave_velocities: Tuple[float, float]
    _node_fluxes: Tuple[np.ndarray, np.ndarray]
    _topography_step: float

    def __init__(
        self,
        volume_space: FiniteVolumeSpace,
        gravitational_acceleration: float,
        bottom_topography: np.ndarray,
        intermediate_velocities: ShallowWaterIntermediateVelocities,
        source_term_function: Callable[[float, float, float, float], float],
    ):
        self._volume_space = volume_space
        self._gravitational_acceleration = gravitational_acceleration
        self._bottom_topography = bottom_topography
        self._intermediate_velocities = intermediate_velocities
        self._source_term_function = source_term_function

    @property
    def cell_indices(self) -> Tuple[int, int]:
        return self._cell_indices

    @property
    def node_fluxes(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._node_fluxes

    def __call__(self, dof_vector: np.ndarray, node_index: int):
        self._intermediate_velocities.update(dof_vector)
        self._cell_indices = self._volume_space.left_right_cell(node_index)

        self._build_swe_values(dof_vector)
        self._build_fluxes()
        self._build_wave_velocities(node_index)
        self._build_topography_step()
        self._build_node_fluxes()

    def _build_swe_values(self, dof_vector: np.ndarray):
        left_swe_values, right_swe_values = (
            dof_vector[self._cell_indices[0]],
            dof_vector[self._cell_indices[1]],
        )

        if left_swe_values[0] < self._epsilon:
            left_swe_values = np.zeros(2)

        if right_swe_values[0] < self._epsilon:
            right_swe_values = np.zeros(2)

        self._swe_values = (left_swe_values, right_swe_values)
        self._heights = (left_swe_values[0], right_swe_values[0])
        self._discharges = (left_swe_values[1], right_swe_values[1])

    def _build_fluxes(self):
        self._fluxes = (
            self._flux(*self._swe_values[0]),
            self._flux(*self._swe_values[1]),
        )

    def _flux(self, height, discharge) -> np.ndarray:
        if height != 0:
            return np.array(
                [
                    discharge,
                    discharge**2 / height
                    + self._gravitational_acceleration * height**2 / 2,
                ]
            )
        else:
            return np.zeros(2)

    def _build_wave_velocities(self, edge_index: int):
        self._wave_velocities = (
            self._intermediate_velocities.left_velocities[edge_index],
            self._intermediate_velocities.right_velocities[edge_index],
        )

    def _build_topography_step(self):
        try:
            self._topography_step = (
                self._bottom_topography[self._cell_indices[1]]
                - self._bottom_topography[self._cell_indices[0]]
            )
        except:
            raise ValueError

    def _build_node_fluxes(self):
        if (
            self._wave_velocities[0] < self._epsilon
            and self._wave_velocities[1] < self._epsilon
        ):
            return (np.zeros(2), np.zeros(2))

        left_state, right_state = self._calculate_modified_intermediate_states()

        self._node_fluxes = (
            left_state
            - self._wave_velocities[0] * self._swe_values[0]
            + self._fluxes[0],
            right_state
            - self._wave_velocities[1] * self._swe_values[1]
            + self._fluxes[1],
        )

    def _calculate_modified_intermediate_states(self) -> Tuple[np.ndarray, np.ndarray]:
        (
            left_modified_height,
            right_modified_height,
        ) = self._calculate_modified_intermediate_height()

        discharge = self._calculate_intermediate_discharge()

        return (
            np.array([left_modified_height, self._wave_velocities[0] * discharge]),
            np.array([right_modified_height, self._wave_velocities[1] * discharge]),
        )

    def _calculate_modified_intermediate_height(self) -> Tuple[float, float]:
        # returns intermediate_velocity * intermediate_height. The last one is
        # modified to ensure positivity. The product is returned for avoiding
        # division with small numbers.

        height_HLL = self._calculate_height_HLL()

        left_height = (
            height_HLL
            + self._wave_velocities[1]
            / (self._wave_velocities[1] - self._wave_velocities[0])
            * self._topography_step
        )
        right_height = (
            height_HLL
            + self._wave_velocities[0]
            / (self._wave_velocities[1] - self._wave_velocities[0])
            * self._topography_step
        )

        if self._topography_step >= 0:
            modified_right_height = self._wave_velocities[1] * max(right_height, 0)

            return (
                self._wave_velocities[0] * left_height
                - self._wave_velocities[1] * right_height
                + modified_right_height,
                modified_right_height,
            )
        else:
            modified_left_height = self._wave_velocities[0] * max(left_height, 0)

            return (
                modified_left_height,
                self._wave_velocities[1] * right_height
                - self._wave_velocities[0] * left_height
                + modified_left_height,
            )

    def _calculate_height_HLL(self) -> float:
        return (
            self._wave_velocities[1] * self._heights[1]
            - self._wave_velocities[0] * self._heights[0]
            - self._discharges[1]
            + self._discharges[0]
        ) / (self._wave_velocities[1] - self._wave_velocities[0])

    def _calculate_intermediate_discharge(self) -> float:
        discharge_HLL = self._calculate_discharge_HLL()
        step_length = self._volume_space.mesh.step_length
        source_term = self._source_term_function(
            self._heights[0],
            self._heights[1],
            self._topography_step,
            self._volume_space.mesh.step_length,
        )

        return (
            discharge_HLL
            - self._gravitational_acceleration
            * step_length
            * source_term
            / (self._wave_velocities[1] - self._wave_velocities[0])
        )

    def _calculate_discharge_HLL(self) -> float:
        return (
            self._wave_velocities[1] * self._fluxes[1][0]
            - self._wave_velocities[0] * self._fluxes[0][0]
            - self._fluxes[1][1]
            + self._fluxes[0][1]
        ) / (self._wave_velocities[1] - self._wave_velocities[0])


class ShallowWaterGodunovNumericalFlux(NumericalFlux):
    """Calculates the shallow-water Godunov numerical fluxes,
    i.e.

        FR_{i-1/2}, FL_{i+1/2}

    where i is the cell index.

    """

    _volume_space: FiniteVolumeSpace
    _node_fluxes_calculator: ShallowWaterGodunovNodeFluxesCalculator

    def __init__(
        self,
        volume_space: FiniteVolumeSpace,
        node_fluxes_calculator: ShallowWaterGodunovNodeFluxesCalculator,
    ):
        self._volume_space = volume_space
        self._node_fluxes_calculator = node_fluxes_calculator

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        right_numerical_fluxes = np.empty((self._volume_space.dimension, 2))
        left_numerical_fluxes = np.empty((self._volume_space.dimension, 2))

        for node_index in range(self._volume_space.node_number):
            self._node_fluxes_calculator(dof_vector, node_index)
            (
                left_cell_index,
                right_cell_index,
            ) = self._node_fluxes_calculator.cell_indices
            left_node_flux, right_node_flux = self._node_fluxes_calculator.node_fluxes

            left_numerical_fluxes[right_cell_index] = right_node_flux
            right_numerical_fluxes[left_cell_index] = left_node_flux

        return left_numerical_fluxes, right_numerical_fluxes


def calculate_natural_source_term_discretization(
    left_height: float, right_height: float, topography_step: float, step_length: float
) -> float:
    return (left_height + right_height) / (2 * step_length) * topography_step


def calculate_wet_dry_preserving_source_term_discretization(
    left_height: float, right_height: float, topography_step: float, step_length: float
) -> float:
    if topography_step >= 0:
        return (
            (left_height + right_height)
            / (2 * step_length)
            * min(left_height, topography_step)
        )
    else:
        return (
            (left_height + right_height)
            / (2 * step_length)
            * max(-right_height, topography_step)
        )


class OptimalGodunovTimeStep:
    _discrete_solution: DiscreteSolution
    _intermediate_velocities: ShallowWaterIntermediateVelocities
    _step_length: float

    def __init__(
        self,
        discrete_solution: DiscreteSolution,
        intermediate_velocities: ShallowWaterIntermediateVelocities,
        step_length: float,
    ):
        self._discrete_solution = discrete_solution
        self._intermediate_velocities = intermediate_velocities
        self._step_length = step_length

    def __call__(self) -> float:
        self._intermediate_velocities.update(self._discrete_solution.end_values)
        return self._step_length / (
            2
            * np.max(
                [
                    abs(self._intermediate_velocities.left_velocities),
                    self._intermediate_velocities.right_velocities,
                ]
            )
        )
