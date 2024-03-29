from typing import Callable

import core
import core.ode_solver as os
import defaults
import finite_element
import numpy as np


class OptimalTimeStep:
    _lumped_mass: Callable[[np.ndarray], np.ndarray]
    _artificial_diffusion: core.SystemMatrix

    def __init__(
        self,
        lumped_mass: Callable[[np.ndarray], np.ndarray],
        artificial_diffusion: core.SystemMatrix,
    ):
        self._lumped_mass = lumped_mass
        self._artificial_diffusion = artificial_diffusion

    def __call__(self, time: float, dof_vector: np.ndarray) -> float:
        self._artificial_diffusion.assemble(dof_vector)

        return min(
            self._lumped_mass(dof_vector)
            / (2 * abs(self._artificial_diffusion().diagonal()))
        )


class LowOrderCGRightHandSide:
    """Right hand side of low order continuous Galerkin method (ri). To be more
    precise it is defined as following:

        ri = 1/mi*sum(d_ij*(uj-ui)-(fj-fi)*c_ij, j!=i)

    where mi denotes lumped mass, d_ij an artificial diffusion, ui the DOF
    entries, fi flux approximation and c_ij a discrete gradient.

    """

    _lumped_mass: Callable[[np.ndarray], np.ndarray]
    _artificial_diffusion: core.SystemMatrix
    _flux_gradient: Callable[[np.ndarray], np.ndarray]

    def __init__(
        self,
        lumped_mass: Callable[[np.ndarray], np.ndarray],
        artificial_diffusion: core.SystemMatrix,
        flux_gradient: Callable[[np.ndarray], np.ndarray],
    ):
        self._lumped_mass = lumped_mass
        self._artificial_diffusion = artificial_diffusion
        self._flux_gradient = flux_gradient

    def __call__(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        self._artificial_diffusion.assemble(dof_vector)
        return (
            self._artificial_diffusion.dot(dof_vector) + self._flux_gradient(dof_vector)
        ) / self._lumped_mass(dof_vector)


def get_cg_low_right_hand_side(
    problem: str, element_space: finite_element.LagrangeSpace
) -> LowOrderCGRightHandSide:
    lumped_mass = finite_element.LumpedMassVector(element_space)
    artificial_diffusion = finite_element.build_artificial_diffusion(
        problem, element_space
    )
    flux_gradient = finite_element.build_flux_gradient_approximation(
        problem, element_space
    )

    return LowOrderCGRightHandSide(lumped_mass, artificial_diffusion, flux_gradient)


class LowOrderContinuousGalerkinSolver(finite_element.Solver):
    def __init__(
        self,
        benchmark: core.Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        polynomial_degree=None,
        cfl_number=None,
        ode_solver_type=None,
        adaptive=False,
        save_history=False,
    ):
        name = name or "Low order Continuous Galerkin"
        short = short or "cg_low"
        cfl_number = cfl_number or defaults.FINITE_ELEMENT_CFL_NUMBER
        ode_solver_type = ode_solver_type or os.Heun

        solution = finite_element.get_finite_element_solution(
            benchmark,
            mesh_size=mesh_size,
            polynomial_degree=polynomial_degree,
            save_history=save_history,
        )

        right_hand_side = get_cg_low_right_hand_side(benchmark.problem, solution.space)
        optimal_time_step = OptimalTimeStep(
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


class CGLowParser(finite_element.SolverParser):
    prog = "cg-low"
    name = "Low order Continuous Galerkin"
    solver = LowOrderContinuousGalerkinSolver
    _cfl_default = 1.0

    def _add_arguments(self):
        self._add_adaptive_time_stepping()
        self._add_ode_solver()
