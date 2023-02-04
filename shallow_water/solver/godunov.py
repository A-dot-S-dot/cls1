from typing import Optional, Tuple

import core
import defaults
import numpy as np
import shallow_water
from lib import NumericalFluxDependentRightHandSide
from shallow_water.finite_volume import build_boundary_conditions_applier

from .lax_friedrichs import OptimalTimeStep


class GodunovNumericalFlux:
    """Calculates the shallow-water Godunov numerical fluxes,
    i.e.

        FR_{i-1/2}, FL_{i+1/2}

    where i is the cell index. For more information see 'A simple well-balanced
    and positive numerical scheme for the schallow-water system' by E. Audusse,
    C. Chalons and P. Ung.

    """

    _gravitational_acceleration: float
    _riemann_solver: shallow_water.RiemannSolver
    _topography_step: np.ndarray
    _source_term: shallow_water.SourceTermDiscretization
    _nullifier: shallow_water.Nullifier

    def __init__(
        self,
        gravitational_acceleration: float,
        boundary_conditions=None,
        bottom=None,
        source_term=None,
        nullifier=None,
    ):
        self._gravitational_acceleration = gravitational_acceleration
        self._riemann_solver = shallow_water.RiemannSolver(
            boundary_conditions,
            gravitational_acceleration=gravitational_acceleration,
            wave_speed=shallow_water.WaveSpeed(gravitational_acceleration),
        )
        self._nullifier = nullifier or shallow_water.Nullifier()

        self._build_topography_step(bottom)
        self._build_source_term(source_term)

    def _build_topography_step(self, bottom: Optional[np.ndarray]):
        if bottom is not None:
            bottom = np.array([bottom[0], *bottom, bottom[-1]])
            self._topography_step = np.diff(bottom)
        else:
            self._topography_step = np.array([0.0])

    def _build_source_term(
        self, source_term: Optional[shallow_water.SourceTermDiscretization]
    ):
        if source_term is not None:
            self._source_term = source_term
        elif (self._topography_step == 0).all():
            self._source_term = shallow_water.VanishingSourceTerm()
        else:
            raise ValueError(
                "The bottom seems to be non constant. In this case a source term discretization is needed."
            )

    @property
    def riemann_solver(self) -> shallow_water.RiemannSolver:
        return self._riemann_solver

    def __call__(
        self, time: float, dof_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._riemann_solver.solve(time, dof_vector)

        h_HLL = self._riemann_solver.intermediate_state[:, 0]
        q_HLL = self._riemann_solver.intermediate_state[:, 1]

        h_star_left, h_star_right = self._calculate_h_star(
            h_HLL,
            self._riemann_solver.wave_speed_left,
            self._riemann_solver.wave_speed_right,
        )
        modified_height_left, modified_height_right = self._calculate_modified_heights(
            h_star_left,
            h_star_right,
            self._riemann_solver.wave_speed_left,
            self._riemann_solver.wave_speed_right,
        )

        source_term = self._source_term(
            self._riemann_solver.value_left[:, 0],
            self._riemann_solver.value_right[:, 0],
            self._topography_step,
        )
        q_star = self._calculate_q_star(
            q_HLL,
            self._source_term.step_length,
            source_term,
            self._riemann_solver.wave_speed_left,
            self._riemann_solver.wave_speed_right,
        )

        node_flux_left, node_flux_right = self._calculate_node_flux(
            self._riemann_solver.value_left,
            self._riemann_solver.value_right,
            self._riemann_solver.flux_left,
            self._riemann_solver.flux_right,
            self._riemann_solver.wave_speed_left,
            self._riemann_solver.wave_speed_right,
            modified_height_left,
            modified_height_right,
            q_star,
        )

        return node_flux_right[:-1], -node_flux_left[1:]

    def _calculate_h_star(
        self,
        h_HLL: np.ndarray,
        wave_speed_left: np.ndarray,
        wave_speed_right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            h_HLL
            + wave_speed_right
            / (wave_speed_right + -wave_speed_left)
            * self._topography_step
        ), (
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

    def _calculate_node_flux(
        self,
        value_left: np.ndarray,
        value_right: np.ndarray,
        flux_left: np.ndarray,
        flux_right: np.ndarray,
        wave_speed_left: np.ndarray,
        wave_speed_right: np.ndarray,
        modified_height_left: np.ndarray,
        modified_height_right: np.ndarray,
        q_star: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_state = np.array([modified_height_left, wave_speed_left * q_star]).T
        right_state = np.array([modified_height_right, wave_speed_right * q_star]).T

        return (
            left_state + -wave_speed_left[:, None] * value_left + flux_left,
            right_state + -wave_speed_right[:, None] * value_right + flux_right,
        )


def build_godunov_numerical_flux(
    benchmark: shallow_water.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> GodunovNumericalFlux:
    topography = shallow_water.build_topography_discretization(benchmark, len(mesh))
    source_term = shallow_water.build_source_term(mesh.step_length)

    return GodunovNumericalFlux(
        benchmark.gravitational_acceleration,
        boundary_conditions=build_boundary_conditions_applier(benchmark, (1, 1)),
        bottom=topography,
        source_term=source_term,
    )


class GodunovSolver(core.Solver):
    def __init__(
        self,
        benchmark: shallow_water.ShallowWaterBenchmark,
        name=None,
        short=None,
        mesh_size=None,
        cfl_number=None,
        adaptive=False,
        save_history=False,
    ):
        name = name or "Godunov's finite volume scheme "
        short = short or "godunov"
        mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        cfl_number = cfl_number or defaults.GODUNOV_CFL_NUMBER
        adaptive = adaptive
        ode_solver_type = core.ForwardEuler

        solution = core.finite_volume.build_finite_volume_solution(
            benchmark,
            mesh_size,
            save_history=save_history,
            periodic=benchmark.boundary_conditions == "periodic",
        )

        godunov_flux = build_godunov_numerical_flux(benchmark, solution.space.mesh)
        right_hand_side = NumericalFluxDependentRightHandSide(
            solution.space, godunov_flux
        )
        optimal_time_step = OptimalTimeStep(
            godunov_flux.riemann_solver,
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
