from typing import Tuple

import core
import core.ode_solver as os
import defaults
import numpy as np
import shallow_water
from core import finite_volume
from lib import NumericalFlux, NumericalFluxDependentRightHandSide
from shallow_water.benchmark import ShallowWaterBenchmark

from . import godunov


class IntermediateState:
    """Calculates local Lax-Friedrich's intermediate states of Riemann problem (i.e. approximative solution), i.e. we get for each node between two cells

    u' = (uL+uR)/2 + (f(uL)-f(uR))/(2*lambda),

    where lambda denotes the wave speed and f the flux.

    """

    _volume_space: finite_volume.FiniteVolumeSpace
    _flux: core.SystemVector
    _wave_speed: core.SystemVector

    def __init__(
        self,
        volume_space: finite_volume.FiniteVolumeSpace,
        flux: core.SystemVector,
        wave_speed: core.SystemVector,
    ):
        self._volume_space = volume_space
        self._flux = flux
        self._wave_speed = wave_speed

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        values_left = dof_vector[self._volume_space.left_cell_indices]
        values_right = dof_vector[self._volume_space.right_cell_indices]
        flux_left = self._flux(values_left)
        flux_right = self._flux(values_right)
        wave_speed = self._wave_speed(dof_vector)

        return (values_left + values_right) / 2 + (flux_left + -flux_right) / (
            2 * wave_speed[:, None]
        )


class LLFNumericalFLux(NumericalFlux):
    """Calculates the shallow-water local lax friedrich numerical fluxes,
    i.e.

        FR_{i-1/2}, FL_{i+1/2}

    where i is the cell index. Note the bottom must be constant for that. For
    more information see 'Bound-preserving flux limiting for high-order explicit
    Runge-Kutta time discretizations of hyperbolic conservation laws' by
    D.Kuzmin et al.

    """

    _volume_space: finite_volume.FiniteVolumeSpace
    _flux: core.SystemVector
    _wave_speed: core.SystemVector
    _intermediate_state: core.SystemVector

    def __init__(
        self,
        volume_space: finite_volume.FiniteVolumeSpace,
        flux: core.SystemVector,
        wave_speed: core.SystemVector,
        intermediate_state: core.SystemVector,
    ):
        self._volume_space = volume_space
        self._flux = flux
        self._wave_speed = wave_speed
        self._intermediate_state = intermediate_state

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        value_left = dof_vector[self._volume_space.left_cell_indices]
        flux_left = self._flux(value_left)
        wave_speed = self._wave_speed(dof_vector)
        intermediate_state = self._intermediate_state(dof_vector)

        node_flux = self._calculate_node_flux(
            wave_speed, intermediate_state, value_left, flux_left
        )

        return (
            node_flux[self._volume_space.left_node_indices],
            -node_flux[self._volume_space.right_node_indices],
        )

    def _calculate_node_flux(
        self,
        wave_speed: np.ndarray,
        intermediate_state: np.ndarray,
        value_left: np.ndarray,
        flux_left,
    ) -> np.ndarray:
        return wave_speed[:, None] * (value_left + -intermediate_state) + flux_left


def build_llf_numerical_flux(
    volume_space: finite_volume.FiniteVolumeSpace, gravitational_acceleration=None
) -> NumericalFlux:
    g = gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
    flux = shallow_water.Flux(g)
    wave_speed = shallow_water.MaximumWaveSpeed(volume_space, g)
    intermediate_state = IntermediateState(volume_space, flux, wave_speed)

    return LLFNumericalFLux(volume_space, flux, wave_speed, intermediate_state)


class LocalLaxFriedrichsSolver(core.Solver):
    def __init__(
        self,
        benchmark: ShallowWaterBenchmark,
        name=None,
        short=None,
        mesh_size=None,
        cfl_number=None,
        adaptive=False,
        save_history=False,
    ):
        name = name or "Local Lax-Friedrichs finite volume scheme "
        short = short or "llf"
        mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        cfl_number = cfl_number or defaults.LOCAL_LAX_FRIEDRICHS_CFL_NUMBER
        adaptive = adaptive
        ode_solver_type = os.ForwardEuler

        solution = finite_volume.build_finite_volume_solution(
            benchmark, mesh_size, save_history=save_history
        )

        bottom = shallow_water.build_topography_discretization(
            benchmark, len(solution.space.mesh)
        )
        if not shallow_water.is_constant(bottom):
            raise ValueError("Bottom must be constant.")

        numerical_flux = build_llf_numerical_flux(
            solution.space, benchmark.gravitational_acceleration
        )
        right_hand_side = NumericalFluxDependentRightHandSide(
            solution.space, numerical_flux
        )

        optimal_time_step = godunov.OptimalTimeStep(
            shallow_water.MaximumWaveSpeed(
                solution.space, benchmark.gravitational_acceleration
            ),
            solution.space.mesh.step_length,
        )
        time_stepping = core.build_adaptive_time_stepping(
            benchmark, solution, optimal_time_step, cfl_number, adaptive
        )
        cfl_checker = core.CFLChecker(optimal_time_step)

        core.Solver.__init__(
            self,
            solution,
            right_hand_side,
            ode_solver_type,
            time_stepping,
            name=name,
            short=short,
            cfl_checker=cfl_checker,
        )
