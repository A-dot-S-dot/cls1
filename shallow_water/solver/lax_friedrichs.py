from typing import Tuple

import core.ode_solver as os
import defaults
import numpy as np
import shallow_water
from core import factory
from core import time_stepping as ts
from core.discretization.finite_volume import FiniteVolumeSpace
from core.numerical_flux import NumericalFlux, NumericalFluxDependentRightHandSide
from core.solver import Solver
from shallow_water.benchmark import ShallowWaterBenchmark


class LocalLaxFriedrichsIntermediateState:
    """Calculates local Lax-Friedrich's intermediate states of Riemann problem (i.e. approximative solution), i.e. we get for each node between two cells

    u' = (uL+uR)/2 + (f(uL)-f(uR))/(2*lambda),

    where lambda denotes the wave speed and f the flux.

    """

    _volume_space: FiniteVolumeSpace
    _flux: shallow_water.Flux
    _wave_speed: shallow_water.MaximumWaveSpeed

    def __init__(
        self,
        volume_space: FiniteVolumeSpace,
        gravitational_acceleration: float,
        flux=None,
        wave_speed=None,
    ):
        self._volume_space = volume_space
        self._flux = flux or shallow_water.Flux(gravitational_acceleration)
        self._wave_speed = wave_speed or shallow_water.MaximumWaveSpeed(
            volume_space, gravitational_acceleration
        )

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        values_left = dof_vector[self._volume_space.left_cell_indices]
        values_right = dof_vector[self._volume_space.right_cell_indices]
        flux_left = self._flux(values_left)
        flux_right = self._flux(values_right)
        wave_speed = self._wave_speed(dof_vector)

        return (values_left + values_right) / 2 + (flux_left - flux_right) / (
            2 * wave_speed[:, None]
        )


class LocalLaxFriedrichsFlux(NumericalFlux):
    """Calculates the shallow-water local lax friedrich numerical fluxes,
    i.e.

        FR_{i-1/2}, FL_{i+1/2}

    where i is the cell index. Note the bottom must be constant for that. For
    more information see 'Bound-preserving flux limiting for high-order explicit
    Runge-Kutta time discretizations of hyperbolic conservation laws' by
    D.Kuzmin et al.

    """

    _volume_space: FiniteVolumeSpace
    _wave_speed: shallow_water.MaximumWaveSpeed
    _flux: shallow_water.Flux

    def __init__(
        self,
        volume_space: FiniteVolumeSpace,
        gravitational_acceleration: float,
        flux=None,
        wave_speed=None,
        intermediate_state=None,
    ):
        self._volume_space = volume_space
        self._flux = flux or shallow_water.Flux(gravitational_acceleration)
        self._wave_speed = wave_speed or shallow_water.MaximumWaveSpeed(
            volume_space, gravitational_acceleration
        )
        self._intermediate_state = (
            intermediate_state
            or LocalLaxFriedrichsIntermediateState(
                volume_space,
                gravitational_acceleration,
                flux=flux,
                wave_speed=wave_speed,
            )
        )

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
            node_flux[self._volume_space.right_node_indices],
        )

    def _calculate_node_flux(
        self,
        wave_speed: np.ndarray,
        intermediate_state: np.ndarray,
        value_left: np.ndarray,
        flux_left,
    ) -> np.ndarray:
        return wave_speed[:, None] * (value_left - intermediate_state) + flux_left


def build_local_lax_friedrichs_flux(
    benchmark: ShallowWaterBenchmark, volume_space: FiniteVolumeSpace
) -> NumericalFlux:
    bottom = shallow_water.build_topography_discretization(
        benchmark, len(volume_space.mesh)
    )
    if not shallow_water.is_constant(bottom):
        raise ValueError("Bottom must be constant.")

    return LocalLaxFriedrichsFlux(volume_space, benchmark.gravitational_acceleration)


class LocalLaxFriedrichsSolver(Solver):
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

        solution = factory.build_finite_volume_solution(
            benchmark, mesh_size, save_history=save_history
        )
        numerical_flux = build_local_lax_friedrichs_flux(benchmark, solution.space)
        right_hand_side = NumericalFluxDependentRightHandSide(
            solution.space, numerical_flux
        )
        time_stepping = shallow_water.build_adaptive_time_stepping(
            benchmark, solution, cfl_number, adaptive
        )
        cfl_checker = ts.CFLChecker(
            shallow_water.OptimalTimeStep(
                solution.space, benchmark.gravitational_acceleration
            )
        )

        Solver.__init__(
            self,
            solution,
            right_hand_side,
            ode_solver_type,
            time_stepping,
            name=name,
            short=short,
            cfl_checker=cfl_checker,
        )
