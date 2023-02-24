from typing import Tuple

import core
import defaults
import finite_volume
import numpy as np
from finite_volume import scalar


def limit(
    antidiffusive_flux: np.ndarray,
    wave_speed: np.ndarray,
    bar_state_left: np.ndarray,
    bar_state_right: np.ndarray,
    value_max_left: np.ndarray,
    value_max_right: np.ndarray,
    value_min_left: np.ndarray,
    value_min_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    limited_flux = np.zeros(len(antidiffusive_flux))

    positive_flux_case = antidiffusive_flux >= 0
    negative_flux_case = antidiffusive_flux < 0

    limited_flux[positive_flux_case] = np.minimum(
        antidiffusive_flux[positive_flux_case],
        wave_speed[positive_flux_case]
        * np.minimum(
            value_max_left[positive_flux_case] + -bar_state_left[positive_flux_case],
            bar_state_right[positive_flux_case] + -value_min_right[positive_flux_case],
        ),
    )

    limited_flux[negative_flux_case] = np.maximum(
        antidiffusive_flux[negative_flux_case],
        wave_speed[negative_flux_case]
        * np.maximum(
            value_min_left[negative_flux_case] + -bar_state_left[negative_flux_case],
            bar_state_right[negative_flux_case] + -value_max_right[negative_flux_case],
        ),
    )

    return -limited_flux, limited_flux


class MCLFlux(finite_volume.NumericalFlux):
    _riemann_solver: core.RiemannSolver
    _low_order_flux: finite_volume.NumericalFlux
    _high_order_flux: finite_volume.NumericalFlux
    _local_maximum: core.LocalMaximum
    _local_minimum: core.LocalMinimum

    def __init__(
        self,
        riemann_solver: core.RiemannSolver,
        high_order_flux: finite_volume.NumericalFlux,
        neighbours_indices: core.NeighbourIndicesMapping,
    ):
        self.input_dimension = high_order_flux.input_dimension
        self._riemann_solver = riemann_solver
        self._low_order_flux = finite_volume.NumericalFluxWithArbitraryInput(
            finite_volume.LaxFriedrichsFlux(riemann_solver)
        )
        self._high_order_flux = high_order_flux
        self._local_maximum = core.LocalMaximum(neighbours_indices)
        self._local_minimum = core.LocalMinimum(neighbours_indices)

    @property
    def _wave_speed(self) -> np.ndarray:
        return self._riemann_solver.wave_speed_right

    @property
    def _bar_state(self) -> np.ndarray:
        return self._riemann_solver.intermediate_state

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, low_order_flux = self._low_order_flux(*values)
        _, high_order_flux = self._low_order_flux(*values)

        antidiffusive_flux = high_order_flux - low_order_flux
        value_max, value_min = self._get_bounds(finite_volume.get_dof_vector(*values))
        value_max_left, value_max_right = value_max[:-1], value_max[1:]
        value_min_left, value_min_right = value_min[:-1], value_min[1:]

        limited_flux = limit(
            antidiffusive_flux,
            self._wave_speed,
            self._bar_state,
            self._bar_state,
            value_max_left,
            value_max_right,
            value_min_left,
            value_min_right,
        )

        flux = low_order_flux + limited_flux

        return -flux, flux

    def _get_bounds(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._local_maximum(dof_vector), self._local_minimum(dof_vector)


class MCLSolver(core.Solver):
    def __init__(
        self,
        benchmark: core.Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        cfl_number=None,
        ode_solver_type=None,
        save_history=False,
    ):
        name = name or "MCL Solver"
        short = short or "mcl"
        mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        cfl_number = cfl_number or defaults.MCL_CFL_NUMBER
        ode_solver_type = ode_solver_type or core.Heun
        solution = finite_volume.get_finite_volume_solution(
            benchmark, mesh_size, save_history=save_history
        )
        step_length = solution.space.mesh.step_length

        riemann_solver = scalar.get_riemann_solver(benchmark.problem)
        numerical_flux = MCLFlux(riemann_solver, ..., solution.space.dof_neighbours)
        boundary_conditions = swe.get_boundary_conditions(
            *benchmark.boundary_conditions,
            inflow_left=benchmark.inflow_left,
            inflow_right=benchmark.inflow_right,
        )
        right_hand_side = finite_volume.NumericalFluxDependentRightHandSide(
            numerical_flux,
            step_length,
            boundary_conditions,
            swe.RiemannSolver(benchmark.gravitational_acceleration),
        )

        time_stepping = core.get_mesh_dependent_time_stepping(
            benchmark,
            solution.space.mesh,
            cfl_number or defaults.FINITE_VOLUME_CFL_NUMBER,
        )
        right_hand_side = get_mcl_right_hand_side(benchmark.problem, solution.space)
        optimal_time_step = cg_low.OptimalTimeStep(
            finite_element.LumpedMassVector(solution.space),
            finite_element.build_artificial_diffusion(
                benchmark.problem, solution.space
            ),
        )
        time_stepping = core.get_adaptive_time_stepping(
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
