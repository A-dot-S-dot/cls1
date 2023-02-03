from typing import Tuple

import core
import core.ode_solver as os
import defaults
import numpy as np
import shallow_water
from core import finite_volume
from lib import NumericalFluxDependentRightHandSide

from ..benchmark import ShallowWaterBenchmark
from ..core import *
from ..finite_volume import build_boundary_conditions_applier
from ..riemann_solver import RiemannSolver


class OptimalTimeStep:
    _riemann_solver: core.RiemannSolver
    _step_length: float

    def __init__(
        self,
        riemann_solver: core.RiemannSolver,
        step_length: float,
    ):
        self._riemann_solver = riemann_solver
        self._step_length = step_length

    def __call__(self, time: float, dof_vector: np.ndarray) -> float:
        self._riemann_solver.solve(time, dof_vector)

        return self._step_length / np.max(
            np.array(
                [
                    np.abs(self._riemann_solver.wave_speed_left),
                    self._riemann_solver.wave_speed_right,
                ]
            )
        )


class LLFNumericalFLux:
    """Calculates the shallow-water local lax friedrich numerical fluxes,
    i.e.

        FR_{i-1/2}, FL_{i+1/2}

    where i is the cell index. Note the bottom must be constant for that. For
    more information see 'Bound-preserving flux limiting for high-order explicit
    Runge-Kutta time discretizations of hyperbolic conservation laws' by
    D.Kuzmin et al.

    """

    _riemann_solver: core.RiemannSolver

    def __init__(self, riemann_solver: core.RiemannSolver):
        self._riemann_solver = riemann_solver

    @property
    def riemann_solver(self) -> core.RiemannSolver:
        return self._riemann_solver

    def __call__(
        self, time: float, dof_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._riemann_solver.solve(time, dof_vector)

        node_flux = self._riemann_solver.intermediate_flux

        return node_flux[:-1], -node_flux[1:]


def build_llf_numerical_flux(
    benchmark: shallow_water.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> LLFNumericalFLux:
    riemann_solver = shallow_water.RiemannSolver(
        build_boundary_conditions_applier(benchmark)
    )
    return LLFNumericalFLux(riemann_solver)


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
            benchmark,
            mesh_size,
            save_history=save_history,
            periodic=benchmark.boundary_conditions == "periodic",
        )

        bottom = build_topography_discretization(benchmark, len(solution.space.mesh))
        if not is_constant(bottom):
            raise ValueError("Bottom must be constant.")

        conditions_applier = build_boundary_conditions_applier(
            benchmark, cells_to_add_numbers=(1, 1)
        )
        riemann_solver = RiemannSolver(
            conditions_applier, benchmark.gravitational_acceleration
        )
        numerical_flux = LLFNumericalFLux(riemann_solver)
        right_hand_side = NumericalFluxDependentRightHandSide(
            solution.space, numerical_flux
        )

        optimal_time_step = OptimalTimeStep(
            riemann_solver, solution.space.mesh.step_length
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
