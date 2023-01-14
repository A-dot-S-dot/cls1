from typing import Optional, Tuple

import numpy as np
import problem.shallow_water as shallow_water
from pde_solver.discretization import DiscreteSolution
from pde_solver.discretization.finite_volume import FiniteVolumeSpace

from .numerical_flux import NumericalFlux


class GodunovNumericalFlux(NumericalFlux):
    """Calculates the shallow-water Godunov numerical fluxes,
    i.e.

        FR_{i-1/2}, FL_{i+1/2}

    where i is the cell index. For more information see 'A simple well-balanced
    and positive numerical scheme for the schallow-water system' by E. Audusse,
    C. Chalons and P. Ung.

    """

    _volume_space: FiniteVolumeSpace
    _gravitational_acceleration: float
    _topography_step: np.ndarray
    _source_term: shallow_water.SourceTermDiscretization
    _wave_speed: shallow_water.WaveSpeed
    _flux: shallow_water.Flux
    _nullifier: shallow_water.Nullifier

    def __init__(
        self,
        volume_space: FiniteVolumeSpace,
        gravitational_acceleration: float,
        bottom=None,
        source_term=None,
        wave_speed=None,
        flux=None,
        nullifier=None,
    ):
        self._volume_space = volume_space
        self._gravitational_acceleration = gravitational_acceleration
        self._wave_speed = wave_speed or shallow_water.WaveSpeed(
            volume_space, gravitational_acceleration
        )
        self._flux = flux or shallow_water.Flux(gravitational_acceleration)
        self._nullifier = nullifier or shallow_water.Nullifier()

        self._build_topography_step(bottom)
        self._build_source_term(source_term)

    def _build_topography_step(self, bottom: Optional[np.ndarray]):
        if bottom is not None:
            self._topography_step = (
                bottom[self._volume_space.right_cell_indices]
                - bottom[self._volume_space.left_cell_indices]
            )
        else:
            self._topography_step = np.zeros(self._volume_space.dimension)

    def _build_source_term(
        self, source_term: Optional[shallow_water.SourceTermDiscretization]
    ):
        if source_term is not None:
            self._source_term = source_term
        elif (self._topography_step == self._topography_step[0]).all():
            self._source_term = shallow_water.VanishingSourceTerm()
        else:
            raise ValueError(
                "The bottom seems to be non constant. In this case a source term discretization is needed."
            )

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dof_vector = self._nullifier(dof_vector)
        flux = self._flux(dof_vector)
        wave_speed_left, wave_speed_right = self._wave_speed(dof_vector)

        left_values = dof_vector[self._volume_space.left_cell_indices]
        right_values = dof_vector[self._volume_space.right_cell_indices]
        height_left = left_values[:, 0]
        height_right = right_values[:, 0]
        discharge_left = left_values[:, 1]
        discharge_right = right_values[:, 1]
        flux_left = flux[self._volume_space.left_cell_indices]
        flux_right = flux[self._volume_space.right_cell_indices]
        height_flux_left = flux_left[:, 0]
        height_flux_right = flux_right[:, 0]
        discharge_flux_left = flux_left[:, 1]
        discharge_flux_right = flux_right[:, 1]

        h_HLL = self._calculate_HLL_value(
            height_left,
            height_right,
            height_flux_left,
            height_flux_right,
            wave_speed_left,
            wave_speed_right,
        )
        h_star_left = self._calculate_h_star_left(
            h_HLL, wave_speed_left, wave_speed_right
        )
        h_star_right = self._calculate_h_star_right(
            h_HLL, wave_speed_left, wave_speed_right
        )
        modified_height_left, modified_height_right = self._calculate_modified_heights(
            h_star_left, h_star_right, wave_speed_left, wave_speed_right
        )

        q_HLL = self._calculate_HLL_value(
            discharge_left,
            discharge_right,
            discharge_flux_left,
            discharge_flux_right,
            wave_speed_left,
            wave_speed_right,
        )
        step_length = self._volume_space.mesh.step_length
        source_term = self._source_term(
            height_left, height_right, self._topography_step, step_length
        )
        q_star = self._calculate_q_star(
            q_HLL,
            step_length,
            source_term,
            wave_speed_left,
            wave_speed_right,
        )

        node_flux_left = self._calculate_node_flux_left(
            modified_height_left, q_star, wave_speed_left, left_values, flux_left
        )
        node_flux_right = self._calculate_node_flux_right(
            modified_height_right, q_star, wave_speed_right, right_values, flux_right
        )

        cell_flux_left = self._calculate_cell_flux_left(node_flux_right)
        cell_flux_right = self._calculate_cell_flux_right(node_flux_left)

        return cell_flux_left, cell_flux_right

    def _calculate_HLL_value(
        self,
        value_left: np.ndarray,
        value_right: np.ndarray,
        flux_left: np.ndarray,
        flux_right: np.ndarray,
        wave_speed_left: np.ndarray,
        wave_speed_right: np.ndarray,
    ) -> np.ndarray:
        return (
            wave_speed_right * value_right
            + -wave_speed_left * value_left
            + -flux_right
            + flux_left
        ) / (wave_speed_right + -wave_speed_left)

    def _calculate_h_star_left(
        self,
        h_HLL: np.ndarray,
        wave_speed_left: np.ndarray,
        wave_speed_right: np.ndarray,
    ) -> np.ndarray:
        return (
            h_HLL
            + wave_speed_right
            / (wave_speed_right + -wave_speed_left)
            * self._topography_step
        )

    def _calculate_h_star_right(
        self,
        h_HLL: np.ndarray,
        wave_speed_left: np.ndarray,
        wave_speed_right: np.ndarray,
    ) -> np.ndarray:
        return (
            h_HLL
            + wave_speed_left
            / (wave_speed_right + -wave_speed_left)
            * self._topography_step
        )

    def _calculate_modified_heights(
        self,
        h_star_left: np.ndarray,
        h_star_right: np.ndarray,
        wave_speed_left: np.ndarray,
        wave_speed_right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # returns wave_speed * modified_height. The last one is
        # modified to ensure positivity. The product is returned for avoiding
        # division with small numbers.

        modified_height_left = wave_speed_left * np.maximum(h_star_left, 0)
        modified_height_right = wave_speed_right * np.maximum(h_star_right, 0)

        modified_height_left[self._topography_step >= 0] = (
            wave_speed_left * h_star_left
            + -wave_speed_right * h_star_right
            + modified_height_right
        )[self._topography_step >= 0]

        modified_height_right[self._topography_step < 0] = (
            wave_speed_right * h_star_right
            + -wave_speed_left * h_star_left
            + modified_height_left
        )[self._topography_step < 0]

        return modified_height_left, modified_height_right

    def _calculate_q_star(
        self,
        q_HLL: np.ndarray,
        step_length: float,
        source_term: np.ndarray,
        wave_speed_left: np.ndarray,
        wave_speed_right: np.ndarray,
    ) -> np.ndarray:
        return q_HLL - self._gravitational_acceleration * step_length * source_term / (
            wave_speed_right + -wave_speed_left
        )

    def _calculate_node_flux_left(
        self,
        modified_height_left: np.ndarray,
        q_star: np.ndarray,
        wave_speed_left: np.ndarray,
        left_values: np.ndarray,
        flux_left: np.ndarray,
    ) -> np.ndarray:
        left_state = np.array([modified_height_left, wave_speed_left * q_star]).T

        return left_state + -wave_speed_left[:, None] * left_values + flux_left

    def _calculate_node_flux_right(
        self,
        modified_height_right: np.ndarray,
        q_star: np.ndarray,
        wave_speed_right: np.ndarray,
        right_values: np.ndarray,
        flux_right: np.ndarray,
    ):
        right_state = np.array([modified_height_right, wave_speed_right * q_star]).T
        return right_state + -wave_speed_right[:, None] * right_values + flux_right

    def _calculate_cell_flux_left(self, node_flux_right: np.ndarray) -> np.ndarray:
        return node_flux_right[self._volume_space.left_node_indices]

    def _calculate_cell_flux_right(self, node_flux_left: np.ndarray) -> np.ndarray:
        return node_flux_left[self._volume_space.right_node_indices]


class OptimalTimeStep:
    _discrete_solution: DiscreteSolution
    _wave_speed: shallow_water.WaveSpeed
    _step_length: float

    def __init__(
        self,
        discrete_solution: DiscreteSolution,
        volume_space: FiniteVolumeSpace,
        gravitational_acceleration: float,
        step_length: float,
    ):
        self._discrete_solution = discrete_solution
        self._wave_speed = shallow_water.WaveSpeed(
            volume_space, gravitational_acceleration
        )

        self._step_length = step_length

    def __call__(self) -> float:
        wave_speed_left, wave_speed_right = self._wave_speed(
            self._discrete_solution.end_values
        )
        return self._step_length / (
            2
            * np.max(
                [
                    abs(wave_speed_left),
                    wave_speed_right,
                ]
            )
        )
