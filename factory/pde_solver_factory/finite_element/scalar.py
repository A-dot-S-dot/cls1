from typing import Dict

import factory
import numpy as np
import pde_solver.system_matrix as sm
import pde_solver.system_vector as sf
from factory.pde_solver_factory import PDESolverFactory
from pde_solver import ScalarFiniteElementSolver
from pde_solver.discrete_solution import DiscreteSolutionObservable
from pde_solver.interpolate import NodeValuesInterpolator
from pde_solver.ode_solver import ExplicitRungeKuttaMethod
from pde_solver.solver_space import LagrangeFiniteElementSpace
from pde_solver.time_stepping import CFLCheckedVector


class ScalarFiniteElementSolverFactory(PDESolverFactory[int]):
    """Superclass, some methods must be implemented by subclasses."""

    _solver: ScalarFiniteElementSolver
    _solver_space: LagrangeFiniteElementSpace

    def _setup_solver(self):
        self._build_solver()
        self._build_space()
        self._build_solution()
        self._build_tqdm_kwargs()
        self._build_time_stepping()
        self._build_right_hand_side()
        self._build_ode_solver()

    def _build_solver(self):
        self._solver = ScalarFiniteElementSolver()

    def _build_space(self):
        self._solver_space = LagrangeFiniteElementSpace(
            self.mesh, self.attributes.polynomial_degree
        )
        self._solver.solver_space = self._solver_space

    def _build_solution(self):
        interpolator = NodeValuesInterpolator(*self._solver_space.basis_nodes)
        self._solver.solution = DiscreteSolutionObservable(
            self.benchmark.start_time,
            interpolator.interpolate(self.benchmark.initial_data),
        )

    def _build_tqdm_kwargs(self):
        self._solver.tqdm_kwargs = self.tqdm_kwargs

    def _build_time_stepping(self):
        raise NotImplementedError

    def _build_right_hand_side(self):
        raise NotImplementedError

    def _build_ode_solver(self):
        raise NotImplementedError

    def _setup_ode_solver(self, ode_solver: ExplicitRungeKuttaMethod):
        ode_solver.time = self.benchmark.start_time
        ode_solver.start_value = self._solver.solution.initial_data
        ode_solver.right_hand_side = self._solver.right_hand_side

    @property
    def grid(self) -> np.ndarray:
        return self._solver_space.basis_nodes

    @property
    def cell_quadrature_degree(self) -> int:
        return self.attributes.polynomial_degree + 1


class ContinuousGalerkinSolverFactory(ScalarFiniteElementSolverFactory):
    _problem_name: str
    _plot_label = ["cg"]

    @property
    def problem_name(self) -> str:
        return self._problem_name

    @problem_name.setter
    def problem_name(self, problem_name: str):
        self._problem_name = problem_name
        factory.FLUX_FACTORY.problem_name = problem_name

    def _build_time_stepping(self):
        time_stepping = factory.TIME_STEPPING_FACTORY.mesh_time_stepping
        time_stepping.start_time = self.benchmark.start_time
        time_stepping.end_time = self.benchmark.end_time
        time_stepping.mesh = self.mesh
        time_stepping.cfl_number = self.attributes.cfl_number

        self._solver.time_stepping = time_stepping

    def _build_right_hand_side(self):
        # must be created first such that this quantities updated first if DOF change
        mass = self._build_mass()
        flux_gradient = self._build_flux_gradient()

        right_hand_side = sf.CGRightHandSide()
        right_hand_side.mass = mass
        right_hand_side.flux_gradient = flux_gradient

        self._solver.right_hand_side = right_hand_side

    def _build_mass(self) -> sm.SystemMatrix:
        mass = sm.MassMatrix(self._solver_space)
        mass.build_inverse()

        return mass

    def _build_flux_gradient(self) -> sf.SystemVector:
        discrete_gradient = sm.DiscreteGradient(self._solver_space)
        factory.FLUX_FACTORY.exact_flux = self.attributes.exact_flux
        return factory.FLUX_FACTORY.get_flux_gradient(discrete_gradient)

    def _build_ode_solver(self):
        ode_solver = factory.ODE_SOLVER_FACTORY.get_optimal_ode_solver(
            self.attributes.polynomial_degree
        )
        self._setup_ode_solver(ode_solver)

        self._solver.ode_solver = ode_solver

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = {
            "desc": "Continuous Galerkin",
            "leave": False,
            "postfix": {
                "p": self.attributes.polynomial_degree,
                "exact_flux": int(self.attributes.exact_flux),
                "cfl_number": self.attributes.cfl_number,
                "DOFs": self.dimension,
            },
        }

        return tqdm_kwargs

    @property
    def info(self) -> str:
        return f"CG (p={self.attributes.polynomial_degree}, exact_flux={int(self.attributes.exact_flux)}, cfl={self.attributes.cfl_number})"


class LowOrderCGFactory(ScalarFiniteElementSolverFactory):
    _lumped_mass: sf.SystemVector
    _artificial_diffusion: sm.SystemMatrix
    _problem_name: str
    _plot_label = ["cg_low"]

    @property
    def problem_name(self) -> str:
        return self._problem_name

    @problem_name.setter
    def problem_name(self, problem_name: str):
        self._problem_name = problem_name
        factory.FLUX_FACTORY.problem_name = problem_name
        factory.DIFFUSION_FACTORY.problem_name = problem_name
        factory.TIME_STEPPING_FACTORY.problem_name = problem_name

    def _build_time_stepping(self):
        self._lumped_mass = self._build_lumped_mass()
        self._artificial_diffusion = self._build_diffusion()

        time_stepping = factory.TIME_STEPPING_FACTORY.get_mcl_time_stepping(
            self._solver.solution
        )
        time_stepping.start_time = self.benchmark.start_time
        time_stepping.end_time = self.benchmark.end_time
        time_stepping.cfl_number = self.attributes.cfl_number
        time_stepping.lumped_mass = self._lumped_mass
        time_stepping.artificial_diffusion = self._artificial_diffusion

        self._solver.time_stepping = time_stepping

    def _build_lumped_mass(self) -> sf.SystemVector:
        return sf.LumpedMassVector(self._solver_space)

    def _build_diffusion(self) -> sm.SystemMatrix:
        discrete_gradient = sm.DiscreteGradient(self._solver_space)
        return factory.DIFFUSION_FACTORY.get_artificial_diffusion(
            discrete_gradient, self._solver.solution
        )

    def _build_right_hand_side(self):
        # must be created first such that this quantities updated first if DOF change
        discrete_gradient = self._build_discrete_gradient()
        flux_approximation = self._build_flux_approximation()

        right_hand_side = sf.LowOrderCGRightHandSide()
        right_hand_side.lumped_mass = self._lumped_mass
        right_hand_side.artificial_diffusion = self._artificial_diffusion
        right_hand_side.discrete_gradient = discrete_gradient
        right_hand_side.flux_approximation = flux_approximation

        self._solver.right_hand_side = CFLCheckedVector(
            right_hand_side, self._solver.time_stepping
        )

    def _build_discrete_gradient(self) -> sm.SystemMatrix:
        return sm.DiscreteGradient(self._solver_space)

    def _build_flux_approximation(self) -> sf.SystemVector:
        return sf.FluxApproximation(factory.FLUX_FACTORY.flux)

    def _build_ode_solver(self):
        ode_solver = factory.ODE_SOLVER_FACTORY.get_ode_solver(
            self.attributes.ode_solver
        )
        self._setup_ode_solver(ode_solver)

        self._solver.ode_solver = ode_solver

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = {
            "desc": "Low Order Continuous Galerkin",
            "leave": False,
            "postfix": {
                "p": self.attributes.polynomial_degree,
                "cfl_number": self.attributes.cfl_number,
                "DOFs": self.dimension,
                "ode_solver": self.attributes.ode_solver,
            },
        }

        return tqdm_kwargs

    @property
    def info(self) -> str:
        return f"Low CG (p={self.attributes.polynomial_degree}, cfl={self.attributes.cfl_number}, ode_solver={self.attributes.ode_solver})"


class MCLSolverFactory(LowOrderCGFactory):
    _plot_label = ["MCL"]

    def _build_right_hand_side(self):
        # must be created first such that this quantities updated first if DOF change
        mass = self._build_mass()
        local_maximum = self._build_local_maximum()
        local_minimum = self._build_local_minimum()
        discrete_gradient = self._build_discrete_gradient()
        flux_approximation = self._build_flux_approximation()

        low_cg = sf.LowOrderCGRightHandSide()
        low_cg.lumped_mass = self._lumped_mass
        low_cg.artificial_diffusion = self._artificial_diffusion
        low_cg.discrete_gradient = discrete_gradient
        low_cg.flux_approximation = flux_approximation

        right_hand_side = sf.MCLRightHandSide(self._solver_space)
        right_hand_side.mass = mass
        right_hand_side.local_maximum = local_maximum
        right_hand_side.local_minimum = local_minimum
        right_hand_side.low_cg_right_hand_side = low_cg

        self._solver.right_hand_side = CFLCheckedVector(
            right_hand_side, self._solver.time_stepping
        )

    def _build_mass(self) -> sm.SystemMatrix:
        return sm.MassMatrix(self._solver_space)

    def _build_local_maximum(self) -> sf.SystemVector:
        return sf.LocalMaximum(self._solver_space)

    def _build_local_minimum(self) -> sf.SystemVector:
        return sf.LocalMinimum(self._solver_space)

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = super().tqdm_kwargs
        tqdm_kwargs["desc"] = "MCL Limiter"

        return tqdm_kwargs

    @property
    def info(self) -> str:
        info = super().info
        attributes = info[info.find("(") :]
        return f"MCL {attributes}"
