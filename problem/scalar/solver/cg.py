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


class CGRightHandSideFactory:
    def __call__(
        self,
        problem: str,
        element_space: finite_element.LagrangeSpace,
        exact_flux=False,
    ) -> SystemVector:
        mass = scalar.MassMatrix(element_space)
        flux_gradient = self.build_flux_gradient(problem, element_space, exact_flux)

        return CGRightHandSide(mass, flux_gradient)

    def build_flux_gradient(
        self,
        problem: str,
        element_space: finite_element.LagrangeSpace,
        exact_flux: bool,
    ) -> SystemVector:
        if exact_flux:
            return self.build_exact_flux_gradient(problem, element_space)
        else:
            return self.build_flux_gradient_approximation(problem, element_space)

    def build_exact_flux_gradient(
        self, problem: str, element_space: finite_element.LagrangeSpace
    ) -> SystemVector:
        flux_gradients = {
            "advection": scalar.AdvectionFluxGradient(element_space),
            "burgers": scalar.FluxGradient(element_space, lambda u: 1 / 2 * u**2),
        }
        return flux_gradients[problem]

    def build_flux_gradient_approximation(
        self, problem: str, element_space: finite_element.LagrangeSpace
    ) -> SystemVector:
        flux_gradients = {
            "advection": scalar.AdvectionFluxGradient(element_space),
            "burgers": scalar.ApproximatedFluxGradient(
                element_space, lambda u: 1 / 2 * u**2
            ),
        }
        return flux_gradients[problem]


CG_FACTORY = CGRightHandSideFactory()


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

        solution, element_space = factory.FINITE_ELEMENT_SOLUTION_FACTORY(
            benchmark, mesh_size, polynomial_degree, save_history=save_history
        )
        right_hand_side = CG_FACTORY(
            benchmark.problem, element_space, exact_flux=exact_flux
        )
        ode_solver = factory.OPTIMAL_ODE_SOLVER_FACTORY(element_space)
        time_stepping = factory.MESH_DEPENDENT_TIME_STEPPING(
            benchmark, element_space.mesh, cfl_number
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
