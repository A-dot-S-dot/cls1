from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict

from fem import FiniteElementSpace, GlobalFiniteElement
from fem.lagrange import LagrangeFiniteElementSpace
from math_type import FunctionRealToReal
from mesh import Mesh
from pde_solver.cg_solver import ContinuousGalerkinSolver
from pde_solver.solver import PDESolver
from pde_solver.time_stepping import SpatialMeshDependendetTimeStepping
from system.matrix.discrete_gradient import DiscreteGradient
from system.matrix.mass import MassMatrix
from system.vector.dof_vector import DOFVector

from factory import FluxGradientFactory, ODESolverFactory


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
        self._create_system_objects()
        self._create_time_stepping()
        self._create_ode_solver()
        self._create_tqdm_kwargs()

    @abstractmethod
    def _create_solver(self):
        ...

    def _create_element_space(self):
        self._element_space = LagrangeFiniteElementSpace(
            self.mesh, self.attributes.polynomial_degree
        )

    def _create_dofs(self):
        self._dof_vector = DOFVector(self._element_space)
        self._solver.discrete_solution_dofs = self._dof_vector
        self._dof_vector.dofs = self._element_space.interpolate(self.initial_data)

    @abstractmethod
    def _create_system_objects(self):
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
    flux_gradient_factory: FluxGradientFactory
    ode_solver_factory: ODESolverFactory

    _problem_name: str
    _solver: ContinuousGalerkinSolver

    @property
    def problem_name(self) -> str:
        return self._problem_name

    @problem_name.setter
    def problem_name(self, problem_name: str):
        self._problem_name = problem_name
        self.flux_gradient_factory.problem_name = problem_name

    def _create_solver(self):
        self._solver = ContinuousGalerkinSolver()

    def _create_system_objects(self):
        self._create_mass()
        self._create_flux_gradient()

    def _create_mass(self):
        mass = MassMatrix(self._element_space)
        mass.build_inverse()

        self._solver.mass = mass

    def _create_flux_gradient(self):
        discrete_gradient = DiscreteGradient(self._element_space)
        self.flux_gradient_factory.exact_flux = self.attributes.exact_flux
        flux_gradient = self.flux_gradient_factory.get_flux_gradient(
            self._dof_vector, discrete_gradient
        )
        self._solver.flux_gradient = flux_gradient

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
