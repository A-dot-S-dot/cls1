import base.ode_solver as os
import defaults
import numpy as np
from base import factory
from base import time_stepping as ts
from base.benchmark import Benchmark
from base.discretization import DiscreteSolution, finite_element
from base.solver import Solver
from base.system import SystemMatrix, SystemVector
from problem import scalar

from .cg import CG_FACTORY


class OptimalTimeStep:
    _lumped_mass: SystemVector
    _artificial_diffusion: SystemMatrix

    def __init__(self, lumped_mass: SystemVector, artificial_diffusion: SystemMatrix):
        self._lumped_mass = lumped_mass
        self._artificial_diffusion = artificial_diffusion

    def __call__(self, dof_vector: np.ndarray) -> float:
        self._artificial_diffusion.assemble(dof_vector)

        return min(
            self._lumped_mass() / (2 * abs(self._artificial_diffusion().diagonal()))
        )


class LowOrderCGRightHandSide(SystemVector):
    """Right hand side of low order continuous Galerkin method (ri). To be more
    precise it is defined as following:

        ri = 1/mi*sum(d_ij*(uj-ui)-(fj-fi)*c_ij, j!=i)

    where mi denotes lumped mass, d_ij an artificial diffusion, ui the DOF
    entries, fi flux approximation and c_ij a discrete gradient.

    """

    _lumped_mass: SystemVector
    _artificial_diffusion: SystemMatrix
    _flux_gradient: SystemVector

    def __init__(
        self,
        lumped_mass: SystemVector,
        artificial_diffusion: SystemMatrix,
        flux_gradient: SystemVector,
    ):
        self._lumped_mass = lumped_mass
        self._artificial_diffusion = artificial_diffusion
        self._flux_gradient = flux_gradient

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        self._artificial_diffusion.assemble(dof_vector)
        return (
            self._artificial_diffusion.dot(dof_vector) + self._flux_gradient(dof_vector)
        ) / self._lumped_mass()


class LowCGRightHandSideFactory:
    def __call__(
        self, problem: str, element_space: finite_element.LagrangeSpace
    ) -> SystemVector:
        lumped_mass = scalar.LumpedMassVector(element_space)
        artificial_diffusion = self.build_artificial_diffusion(problem, element_space)
        flux_gradient = CG_FACTORY.build_flux_gradient_approximation(
            problem, element_space
        )
        return LowOrderCGRightHandSide(lumped_mass, artificial_diffusion, flux_gradient)

    def build_artificial_diffusion(
        self, problem: str, element_space: finite_element.LagrangeSpace
    ) -> SystemMatrix:
        diffusions = {
            "advection": scalar.DiscreteUpwind,
            "burgers": scalar.BurgersArtificialDiffusion,
        }
        return diffusions[problem](element_space)


LOW_CG_FACTORY = LowCGRightHandSideFactory()


class AdaptiveTimeSteppingFactory:
    def __call__(
        self,
        benchmark: Benchmark,
        discrete_solution: DiscreteSolution,
        cfl_number: float,
        adaptive: bool,
    ) -> ts.TimeStepping:
        lumped_mass = scalar.LumpedMassVector(discrete_solution.space)
        artificial_diffusion = LOW_CG_FACTORY.build_artificial_diffusion(
            benchmark.problem, discrete_solution.space
        )

        return ts.TimeStepping(
            benchmark.end_time,
            cfl_number,
            ts.DiscreteSolutionDependentTimeStep(
                OptimalTimeStep(
                    lumped_mass,
                    artificial_diffusion,
                ),
                discrete_solution,
            ),
            adaptive=adaptive,
            start_time=benchmark.start_time,
        )


ADAPTIVE_TIME_STEPPING_FACTORY = AdaptiveTimeSteppingFactory()


class LowOrderContinuousGalerkinSolver(Solver):
    def __init__(
        self,
        benchmark: Benchmark,
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
        short = short or "low_cg"
        mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        polynomial_degree = polynomial_degree or defaults.POLYNOMIAL_DEGREE
        cfl_number = cfl_number or defaults.MCL_CFL_NUMBER
        ode_solver_type = ode_solver_type or os.Heun

        solution, element_space = factory.FINITE_ELEMENT_SOLUTION_FACTORY(
            benchmark, mesh_size, polynomial_degree, save_history=save_history
        )
        right_hand_side = LOW_CG_FACTORY(benchmark.problem, element_space)
        time_stepping = ADAPTIVE_TIME_STEPPING_FACTORY(
            benchmark, solution, cfl_number, adaptive
        )
        cfl_checker = ts.CFLChecker(
            OptimalTimeStep(
                scalar.LumpedMassVector(element_space),
                LOW_CG_FACTORY.build_artificial_diffusion(
                    benchmark.problem, element_space
                ),
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
