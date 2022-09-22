from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict

from fem import FiniteElementSpace, GlobalFiniteElement
from fem.lagrange import LagrangeFiniteElementSpace
from math_type import FunctionRealToReal
from mesh import Mesh
from pde_solver.solver import PDESolver
from pde_solver.time_stepping import SpatialMeshDependendetTimeStepping
from system.matrix import SystemMatrix
from system.matrix.discrete_gradient import DiscreteGradient
from system.matrix.mass import MassMatrix
from system.vector import SystemVector
from system.vector.cg import CGRightHandSide
from system.vector.dof_vector import DOFVector
from system.vector.group_finite_element_approximation import (
    GroupFiniteElementApproximation,
)
from system.vector.low_order_cg import LowOrderCGRightHandSide
from system.vector.lumped_mass import LumpedMassVector

from .artificial_diffusion_factory import ArtificialDiffusionFactory
from .flux_factory import FluxFactory
from .ode_solver_factory import ODESolverFactory


class PDESolverFactory(ABC):
    attributes: Namespace
    problem_name: str
    mesh: Mesh
    initial_data: FunctionRealToReal
    start_time: float
    end_time: float

    _solver: PDESolver
    _element_space: FiniteElementSpace
    _dof_vector: DOFVector

    @property
    def solver(self) -> PDESolver:
        self._create_attributes()

        return self._solver

    def _create_attributes(self):
        self._create_solver()
        self._create_element_space()
        self._create_dofs()
        self._create_right_hand_side()
        self._create_time_stepping()
        self._create_ode_solver()
        self._create_tqdm_kwargs()

    def _create_solver(self):
        self._solver = PDESolver()

    def _create_element_space(self):
        self._element_space = LagrangeFiniteElementSpace(
            self.mesh, self.attributes.polynomial_degree
        )

    def _create_dofs(self):
        self._dof_vector = DOFVector(self._element_space)
        self._solver.discrete_solution_dofs = self._dof_vector
        self._dof_vector.dofs = self._element_space.interpolate(self.initial_data)

    @abstractmethod
    def _create_right_hand_side(self):
        ...

    @abstractmethod
    def _create_time_stepping(self):
        ...

    @abstractmethod
    def _create_ode_solver(self):
        ...

    def _create_tqdm_kwargs(self):
        self._solver.tqdm_kwargs = self.tqdm_kwargs

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


class ContinuousGalerkinSolverFactory(PDESolverFactory):
    flux_factory: FluxFactory
    ode_solver_factory: ODESolverFactory

    _problem_name: str

    @property
    def problem_name(self) -> str:
        return self._problem_name

    @problem_name.setter
    def problem_name(self, problem_name: str):
        self._problem_name = problem_name
        self.flux_factory.problem_name = problem_name

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
        self.flux_factory.exact_flux = self.attributes.exact_flux
        return self.flux_factory.get_flux_gradient(self._dof_vector, discrete_gradient)

    def _create_time_stepping(self):
        self._solver.time_stepping = SpatialMeshDependendetTimeStepping(
            self.start_time, self.end_time, self.mesh, self.attributes.cfl_number
        )

    def _create_ode_solver(self):
        ode_solver = self.ode_solver_factory.get_optimal_ode_solver(
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


class LowOrderCGFactory(PDESolverFactory):
    ode_solver_factory: ODESolverFactory
    flux_factory: FluxFactory
    artificial_diffusion_factory: ArtificialDiffusionFactory

    _problem_name: str

    @property
    def problem_name(self) -> str:
        return self._problem_name

    @problem_name.setter
    def problem_name(self, problem_name: str):
        self._problem_name = problem_name
        self.flux_factory.problem_name = problem_name
        self.artificial_diffusion_factory.problem_name = problem_name

    def _create_right_hand_side(self):
        # must be created first such that this quantities updated first if DOF change
        lumped_mass = self._create_lumped_mass()
        artificial_diffusion = self._create_diffusion()
        discrete_gradient = self._create_discrete_gradient()
        flux_approximation = self._create_flux_approximation()

        right_hand_side = LowOrderCGRightHandSide(self._dof_vector)
        right_hand_side.lumped_mass = lumped_mass
        right_hand_side.artificial_diffusion = artificial_diffusion
        right_hand_side.discrete_gradient = discrete_gradient
        right_hand_side.flux_approximation = flux_approximation

        self._solver.right_hand_side = right_hand_side

    def _create_lumped_mass(self) -> SystemVector:
        return LumpedMassVector(self._element_space)

    def _create_diffusion(self) -> SystemMatrix:
        discrete_gradient = DiscreteGradient(self._element_space)
        return self.artificial_diffusion_factory.get_artificial_diffusion(
            self._dof_vector, discrete_gradient
        )

    def _create_discrete_gradient(self) -> SystemMatrix:
        return DiscreteGradient(self._element_space)

    def _create_flux_approximation(self) -> SystemVector:
        return GroupFiniteElementApproximation(self._dof_vector, self.flux_factory.flux)

    def _create_time_stepping(self):
        self._solver.time_stepping = SpatialMeshDependendetTimeStepping(
            self.start_time, self.end_time, self.mesh, self.attributes.cfl_number
        )

    def _create_ode_solver(self):
        ode_solver = self.ode_solver_factory.get_ode_solver("euler")
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
            },
        }

        return tqdm_kwargs

    @property
    def eoc_title(self) -> str:
        title = self.info
        return title + "\n" + "-" * len(title)

    @property
    def info(self) -> str:
        return f"Low CG (p={self.attributes.polynomial_degree}, cfl={self.attributes.cfl_number})"
