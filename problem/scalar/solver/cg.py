import defaults
import numpy as np
import problem.scalar as scalar
from base import factory
from base.benchmark import Benchmark
from base.discretization import finite_element
from base.solver import Solver
from base.system import SystemMatrix, SystemVector


class CGRightHandSide(SystemVector):
    """Right hand side of continuous Galerkin method r. To be more
    precise it is defined as following:

       Mr = A

    where M denotes mass marix and A the discrete flux gradient.

    """

    mass: SystemMatrix
    flux_gradient: SystemVector

    def __init__(self, mass: SystemMatrix, flux_gradient: SystemVector):
        self.mass = mass
        self.flux_gradient = flux_gradient

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return self.mass.inverse(self.flux_gradient(dof_vector))


def build_cg_right_hand_side(
    problem: str,
    element_space: finite_element.LagrangeSpace,
    exact_flux=False,
) -> SystemVector:
    mass = scalar.MassMatrix(element_space)
    flux_gradient = scalar.build_flux_gradient(problem, element_space, exact_flux)

    return CGRightHandSide(mass, flux_gradient)


class ContinuousGalerkinSolver(Solver):
    def __init__(
        self,
        benchmark: Benchmark,
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
        cfl_number = cfl_number or defaults.CFL_NUMBER
        exact_flux = exact_flux

        solution = factory.build_finite_element_solution(
            benchmark, mesh_size, polynomial_degree, save_history=save_history
        )

        right_hand_side = build_cg_right_hand_side(
            benchmark.problem, solution.space, exact_flux=exact_flux
        )
        ode_solver = factory.build_optimal_ode_solver(solution.space)
        time_stepping = factory.build_mesh_dependent_time_stepping(
            benchmark, solution.space.mesh, cfl_number
        )

        Solver.__init__(
            self,
            solution,
            right_hand_side,
            ode_solver,
            time_stepping,
            name=name,
            short=short,
        )
