from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Callable, Dict
from math_type import MultidimensionalFunction

from fem import FiniteElementSpace, GlobalFiniteElement
from fem.lagrange import LagrangeFiniteElementSpace
from math_type import ScalarFunction
from mesh import Mesh
from pde_solver.solver import FiniteElementSolver, PDESolver
from system.matrix import SystemMatrix
from system.matrix.discrete_gradient import DiscreteGradient
from system.matrix.mass import MassMatrix
from system.vector import SystemVector
from system.vector.cg import CGRightHandSide
from system.vector.dof_vector import DOFVector
from system.vector.group_finite_element_approximation import (
    GroupFiniteElementApproximation,
)
from system.vector.local_bounds import LocalMaximum, LocalMinimum
from system.vector.low_order_cg import LowOrderCGRightHandSide
from system.vector.lumped_mass import LumpedMassVector
from system.vector.mcl import MCLRightHandSide

from .artificial_diffusion_factory import DIFFUSION_FACTORY
from .flux_factory import FLUX_FACTORY
from .ode_solver_factory import ODE_SOLVER_FACTORY
from .time_stepping_factory import TIME_STEPPING_FACTORY


class PDESolverFactory(ABC):
    attributes: Namespace
    problem_name: str
    mesh: Mesh
    initial_data: ScalarFunction
    start_time: float
    end_time: float

    _solver: PDESolver

    @property
    def solver(self) -> PDESolver:
        self._create_attributes()

        return self._solver

    @abstractmethod
    def _create_attributes(self):
        ...

    @property
    @abstractmethod
    def cell_quadrature_degree(self) -> int:
        ...

    @property
    @abstractmethod
    def dofs(self) -> int:
        ...

    @property
    @abstractmethod
    def plot_label(self) -> str:
        ...

    @property
    @abstractmethod
    def eoc_title(self) -> str:
        ...

    @property
    @abstractmethod
    def tqdm_kwargs(self) -> Dict:
        ...

    @property
    @abstractmethod
    def info(self) -> str:
        ...

    @property
    @abstractmethod
    def discrete_solution(self) -> MultidimensionalFunction:
        ...


class FiniteElementSolverFactory(PDESolverFactory):
    """Superclass, some methods must be implemented by subclasses."""

    _solver: FiniteElementSolver
    _element_space: FiniteElementSpace
    _dof_vector: DOFVector

    def _create_attributes(self):
        self._create_solver()
        self._create_element_space()
        self._create_dofs()
        self._create_right_hand_side()
        self._build_values()

        self._create_time_stepping()
        self._create_ode_solver()
        self._create_tqdm_kwargs()

    def _create_solver(self):
        self._solver = FiniteElementSolver()

    def _create_element_space(self):
        self._element_space = LagrangeFiniteElementSpace(
            self.mesh, self.attributes.polynomial_degree
        )

    def _create_dofs(self):
        self._dof_vector = DOFVector(self._element_space)
        self._solver.discrete_solution_dofs = self._dof_vector

    def _create_right_hand_side(self):
        raise NotImplementedError

    def _create_time_stepping(self):
        raise NotImplementedError

    @abstractmethod
    def _create_ode_solver(self):
        raise NotImplementedError

    def _create_tqdm_kwargs(self):
        self._solver.tqdm_kwargs = self.tqdm_kwargs

    def _build_values(self):
        self._dof_vector.dofs = self._element_space.interpolate(self.initial_data)

    @property
    def cell_quadrature_degree(self) -> int:
        return self.attributes.polynomial_degree + 1

    @property
    def dofs(self) -> int:
        return len(self._dof_vector.dofs)

    @property
    @abstractmethod
    def plot_label(self) -> str:
        ...

    @property
    @abstractmethod
    def eoc_title(self) -> str:
        ...

    @property
    @abstractmethod
    def tqdm_kwargs(self) -> Dict:
        ...

    @property
    @abstractmethod
    def info(self) -> str:
        ...

    @property
    def discrete_solution(self) -> GlobalFiniteElement:
        return GlobalFiniteElement(self._element_space, self._solver.solution)


class ContinuousGalerkinSolverFactory(FiniteElementSolverFactory):
    _problem_name: str

    @property
    def problem_name(self) -> str:
        return self._problem_name

    @problem_name.setter
    def problem_name(self, problem_name: str):
        self._problem_name = problem_name
        FLUX_FACTORY.problem_name = problem_name

    def _create_right_hand_side(self):
        # must be created first such that this quantities updated first if DOF change
        mass = self._create_mass()
        flux_gradient = self._create_flux_gradient()

        right_hand_side = CGRightHandSide(self._dof_vector)
        right_hand_side.mass = mass
        right_hand_side.flux_gradient = flux_gradient

        self._solver.right_hand_side = right_hand_side

    def _create_mass(self) -> SystemMatrix:
        mass = MassMatrix(self._element_space)
        mass.build_inverse()

        return mass

    def _create_flux_gradient(self) -> SystemVector:
        discrete_gradient = DiscreteGradient(self._element_space)
        FLUX_FACTORY.exact_flux = self.attributes.exact_flux
        return FLUX_FACTORY.get_flux_gradient(self._dof_vector, discrete_gradient)

    def _create_time_stepping(self):
        time_stepping = TIME_STEPPING_FACTORY.mesh_time_stepping
        time_stepping.start_time = self.start_time
        time_stepping.end_time = self.end_time
        time_stepping.mesh = self.mesh
        time_stepping.cfl_number = self.attributes.cfl_number

        self._solver.time_stepping = time_stepping

    def _create_ode_solver(self):
        ode_solver = ODE_SOLVER_FACTORY.get_optimal_ode_solver(
            self.attributes.polynomial_degree
        )
        self._solver.ode_solver = ode_solver

    @property
    def plot_label(self) -> str:
        if self.attributes.label is not None:
            return self.attributes.label
        else:
            return "cg"

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = {
            "desc": "Continuous Galerkin",
            "leave": False,
            "postfix": {
                "p": self.attributes.polynomial_degree,
                "exact_flux": int(self.attributes.exact_flux),
                "cfl_number": self.attributes.cfl_number,
                "DOFs": self.dofs,
            },
        }

        return tqdm_kwargs

    @property
    def eoc_title(self) -> str:
        title = self.info
        return title + "\n" + "-" * len(title)

    @property
    def info(self) -> str:
        return f"CG (p={self.attributes.polynomial_degree}, exact_flux={int(self.attributes.exact_flux)}, cfl={self.attributes.cfl_number})"


class LowOrderCGFactory(FiniteElementSolverFactory):
    _lumped_mass: SystemVector
    _artificial_diffusion: SystemMatrix
    _problem_name: str

    @property
    def problem_name(self) -> str:
        return self._problem_name

    @problem_name.setter
    def problem_name(self, problem_name: str):
        self._problem_name = problem_name
        FLUX_FACTORY.problem_name = problem_name
        DIFFUSION_FACTORY.problem_name = problem_name
        TIME_STEPPING_FACTORY.problem_name = problem_name

    def _create_right_hand_side(self):
        # must be created first such that this quantities updated first if DOF change
        self._lumped_mass = self._create_lumped_mass()
        self._artificial_diffusion = self._create_diffusion()
        discrete_gradient = self._create_discrete_gradient()
        flux_approximation = self._create_flux_approximation()

        right_hand_side = LowOrderCGRightHandSide(self._dof_vector)
        right_hand_side.lumped_mass = self._lumped_mass
        right_hand_side.artificial_diffusion = self._artificial_diffusion
        right_hand_side.discrete_gradient = discrete_gradient
        right_hand_side.flux_approximation = flux_approximation

        self._solver.right_hand_side = right_hand_side

    def _create_lumped_mass(self) -> SystemVector:
        return LumpedMassVector(self._element_space)

    def _create_diffusion(self) -> SystemMatrix:
        discrete_gradient = DiscreteGradient(self._element_space)
        return DIFFUSION_FACTORY.get_artificial_diffusion(
            self._dof_vector, discrete_gradient
        )

    def _create_discrete_gradient(self) -> SystemMatrix:
        return DiscreteGradient(self._element_space)

    def _create_flux_approximation(self) -> SystemVector:
        return GroupFiniteElementApproximation(self._dof_vector, FLUX_FACTORY.flux)

    def _create_time_stepping(self):
        time_stepping = TIME_STEPPING_FACTORY.mcl_time_stepping
        time_stepping.start_time = self.start_time
        time_stepping.end_time = self.end_time
        time_stepping.cfl_number = self.attributes.cfl_number
        time_stepping.lumped_mass = self._lumped_mass
        time_stepping.artificial_diffusion = self._artificial_diffusion
        time_stepping.setup_delta_t()

        self._solver.time_stepping = time_stepping

    def _create_ode_solver(self):
        ode_solver = ODE_SOLVER_FACTORY.get_ode_solver(self.attributes.ode_solver)
        self._solver.ode_solver = ode_solver

    @property
    def plot_label(self) -> str:
        if self.attributes.label is not None:
            return self.attributes.label
        else:
            return "cg_low"

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = {
            "desc": "Low Order Continuous Galerkin",
            "leave": False,
            "postfix": {
                "p": self.attributes.polynomial_degree,
                "cfl_number": self.attributes.cfl_number,
                "DOFs": self.dofs,
                "ode_solver": self.attributes.ode_solver,
            },
        }

        return tqdm_kwargs

    @property
    def eoc_title(self) -> str:
        title = self.info
        return title + "\n" + "-" * len(title)

    @property
    def info(self) -> str:
        return f"Low CG (p={self.attributes.polynomial_degree}, cfl={self.attributes.cfl_number}, ode_solver={self.attributes.ode_solver})"


class MCLSolverFactory(LowOrderCGFactory):
    def _create_right_hand_side(self):
        # must be created first such that this quantities updated first if DOF change
        mass = self._create_mass()
        local_maximum = self._create_local_maximum()
        local_minimum = self._create_local_minimum()
        self._lumped_mass = self._create_lumped_mass()
        self._artificial_diffusion = self._create_diffusion()
        discrete_gradient = self._create_discrete_gradient()
        flux_approximation = self._create_flux_approximation()

        low_cg = LowOrderCGRightHandSide(self._dof_vector)
        low_cg.lumped_mass = self._lumped_mass
        low_cg.artificial_diffusion = self._artificial_diffusion
        low_cg.discrete_gradient = discrete_gradient
        low_cg.flux_approximation = flux_approximation

        right_hand_side = MCLRightHandSide(self._dof_vector)
        right_hand_side.mass = mass
        right_hand_side.local_maximum = local_maximum
        right_hand_side.local_minimum = local_minimum
        right_hand_side.low_cg_right_hand_side = low_cg

        self._solver.right_hand_side = right_hand_side

    def _create_mass(self) -> SystemMatrix:
        return MassMatrix(self._element_space)

    def _create_local_maximum(self) -> SystemVector:
        return LocalMaximum(self._dof_vector)

    def _create_local_minimum(self) -> SystemVector:
        return LocalMinimum(self._dof_vector)

    @property
    def plot_label(self) -> str:
        if self.attributes.label is not None:
            return self.attributes.label
        else:
            return "MCL"

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = super().tqdm_kwargs
        tqdm_kwargs["desc"] = "MCL Limiter"

        return tqdm_kwargs

    @property
    def eoc_title(self) -> str:
        title = self.info
        return title + "\n" + "-" * len(title)

    @property
    def info(self) -> str:
        info = super().info
        attributes = info[info.find("(") :]
        return f"MCL {attributes}"
