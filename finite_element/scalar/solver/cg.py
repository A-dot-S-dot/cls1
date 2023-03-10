from typing import Callable

import core
import defaults
import finite_element
import numpy as np


class CGRightHandSide:
    """Right hand side of continuous Galerkin method r. To be more
    precise it is defined as following:

       Mr = A

    where M denotes mass marix and A the discrete flux gradient.

    """

    mass: core.SystemMatrix
    flux_gradient: Callable[[np.ndarray], np.ndarray]

    def __init__(
        self, mass: core.SystemMatrix, flux_gradient: Callable[[np.ndarray], np.ndarray]
    ):
        self.mass = mass
        self.flux_gradient = flux_gradient

    def __call__(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        return self.mass.inverse(self.flux_gradient(dof_vector))


def get_cg_right_hand_side(
    problem: str,
    element_space: finite_element.LagrangeSpace,
    exact_flux=False,
) -> CGRightHandSide:
    mass = finite_element.MassMatrix(element_space)
    flux_gradient = finite_element.build_flux_gradient(
        problem, element_space, exact_flux
    )

    return CGRightHandSide(mass, flux_gradient)


class ContinuousGalerkinSolver(core.Solver):
    def __init__(
        self,
        benchmark: core.Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        polynomial_degree=None,
        cfl_number=None,
        exact_flux=False,
        save_history=False,
    ):
        name = name or "Continuous Galerkin"
        short = short or "cg"
        mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        polynomial_degree = polynomial_degree or defaults.POLYNOMIAL_DEGREE
        cfl_number = cfl_number or defaults.FINITE_ELEMENT_CFL_NUMBER
        exact_flux = exact_flux

        solution = finite_element.get_finite_element_solution(
            benchmark, mesh_size, polynomial_degree, save_history=save_history
        )

        right_hand_side = get_cg_right_hand_side(
            benchmark.problem, solution.space, exact_flux=exact_flux
        )
        ode_solver = finite_element.build_optimal_ode_solver(solution.space)
        time_stepping = core.get_mesh_dependent_time_stepping(
            benchmark, solution.space.mesh, cfl_number
        )

        core.Solver.__init__(
            self,
            solution,
            right_hand_side,
            ode_solver,
            time_stepping,
            name=name,
            short=short,
        )


class CGParser(finite_element.SolverParser):
    prog = "cg"
    name = "Continuous Galerkin"
    solver = ContinuousGalerkinSolver

    def _add_arguments(self):
        self.add_argument(
            "++exact-flux", action="store_true", help="Calculate flux matrices exactly."
        )
