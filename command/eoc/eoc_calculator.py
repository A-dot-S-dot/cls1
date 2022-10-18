from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np
from benchmark import Benchmark
from custom_type import ScalarFunction
from factory.pde_solver_factory import (
    PDESolverFactory,
    ScalarFiniteElementSolverFactory,
)
from pde_solver.discretization.finite_element import LagrangeFiniteElement
from pde_solver.error import Norm
from pde_solver.mesh import Mesh
from tqdm import trange


class EOCCalculator(ABC):
    benchmark: Benchmark
    refine_number: int
    solver_factory: PDESolverFactory

    _norm: Tuple[Norm, ...]

    def add_norms(self, *norm: Norm):
        self._norm = norm

    @property
    def norm_names(self) -> Sequence[str]:
        return [norm.name for norm in self._norm]

    @property
    def mesh(self):
        return self.solver_factory.mesh

    @mesh.setter
    def mesh(self, mesh: Mesh):
        self.solver_factory.mesh = mesh

        for norm in self._norm:
            norm.mesh = mesh

    @property
    @abstractmethod
    def dofs(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def errors(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def eocs(self) -> np.ndarray:
        ...

    @abstractmethod
    def calculate(self, refinie_number: int) -> np.ndarray:
        ...

    def refine_mesh(self):
        self.mesh = self.mesh.refine()


class ScalarEOCCalculator(EOCCalculator):
    _solver_factory: ScalarFiniteElementSolverFactory

    _dofs: np.ndarray
    _eocs: np.ndarray
    _errors: np.ndarray

    @property
    def solver_factory(self) -> PDESolverFactory:
        return self._solver_factory

    @solver_factory.setter
    def solver_factory(self, solver_factory: PDESolverFactory):
        if isinstance(solver_factory, ScalarFiniteElementSolverFactory):
            self._solver_factory = solver_factory
        else:
            raise NotImplementedError(
                "ScalarEOCCalculator not designed for non finite element solvers."
            )

        self.mesh = self.solver_factory.mesh

    @property
    def dofs(self) -> np.ndarray:
        return self._dofs

    @property
    def errors(self) -> np.ndarray:
        return self._errors

    @property
    def eocs(self) -> np.ndarray:
        return self._eocs

    def calculate(self):
        self._dofs = np.empty(self.refine_number + 1)
        self._errors = np.empty((len(self._norm), self.refine_number + 1))
        self._eocs = np.empty((len(self._norm), self.refine_number + 1))

        for index in trange(
            self.refine_number + 1,
            desc=f"EOC calculation",
            unit="refinements",
            leave=False,
        ):
            discrete_solution = self._calculate_discrete_solution()

            self._dofs[index] = self.solver_factory.dimension
            self._errors[:, index] = self._calculate_error(discrete_solution)
            self._eocs[:, index] = self._calculate_eoc(index)

            if index != self.refine_number:
                self.refine_mesh()

    def _calculate_discrete_solution(self) -> ScalarFunction:
        solver = self.solver_factory.solver
        solver.solve()

        return LagrangeFiniteElement(
            self._solver_factory.solver_space, solver.solution.end_values
        )

    def _calculate_error(self, discrete_solution: ScalarFunction) -> np.ndarray:
        exact_solution = self.benchmark.exact_solution_at_end_time
        function = lambda x: discrete_solution(x) - exact_solution(x)

        return np.array([norm(function) for norm in self._norm])

    def _calculate_eoc(self, current_index: int) -> np.ndarray:
        if current_index == 0:
            return np.array([np.nan for _ in self._norm])
        else:
            return np.array(
                [
                    np.log2(old_error / new_error)
                    for old_error, new_error in zip(
                        self.errors[:, current_index - 1], self.errors[:, current_index]
                    )
                ]
            )
