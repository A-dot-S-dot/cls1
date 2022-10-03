"""This module provides a solver task for analyzing error between an exact
solution and a discrete one of a PDE with diffrent norms.

"""
from abc import ABC
from argparse import Namespace
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd
from benchmark import Benchmark
from factory.pde_solver_factory import PDESolverFactory
from mesh import Mesh
from pde_solver.solver_components import SolverComponents
from quadrature.norm import L1Norm, L2Norm, LInfinityNorm, Norm
from tqdm import tqdm, trange

from .command import Command

ScalarFunction = Callable[[float], float]


class EOCCalculator(ABC):
    ...


class ScalarEOCCalculator(EOCCalculator):
    mesh: Mesh
    benchmark: Benchmark
    solver_factory: PDESolverFactory

    _norm: Tuple[Norm, ...]
    _end_time: float
    _time_steps_number: int

    def add_norm(self, *norm: Norm):
        self._norm = norm

    def eoc(self, refine_number: int) -> np.ndarray:
        data_frame = np.empty((refine_number + 1, 9))

        for index in trange(
            refine_number + 1,
            desc=f"EOC calculation",
            unit="refinements",
            leave=False,
        ):
            discrete_solution = self._calculate_discrete_solution()

            data_frame[index, [0, 3, 6]] = discrete_solution.degree_of_freedom
            for i, norm in enumerate(self._norm):
                data_frame[index, 1 + 3 * i] = self._calculate_error(
                    norm, discrete_solution
                )
                data_frame[index, 2 + 3 * i] = self._calculate_eoc(
                    index, 1 + 3 * i, data_frame
                )

            if index != refine_number:
                self._refine_mesh()

        return data_frame

    def _calculate_discrete_solution(self) -> ScalarFunction:
        solver = self.solver_factory.solver
        solver.solve()

        return self.solver_factory.discrete_solution

    def _calculate_error(self, norm: Norm, discrete_solution: ScalarFunction) -> float:
        exact_solution = self.benchmark.exact_solution_at_end_time
        function = lambda x: discrete_solution(x) - exact_solution(x)

        return norm(function)

    def _calculate_eoc(
        self, index: int, error_column: int, data_frame: np.ndarray
    ) -> float:
        if index == 0:
            return np.nan
        else:
            old_error = data_frame[index - 1, error_column]
            new_error = data_frame[index, error_column]
            return np.log2(old_error / new_error)

    def _refine_mesh(self):
        self.mesh = self.mesh.refine()
        self.solver_factory.mesh = self.mesh

        for norm in self._norm:
            norm.set_mesh(self.mesh)


class EOCCommand(Command):
    _components: SolverComponents
    _args: Namespace
    _benchmark: Benchmark
    _solver_factories: Sequence[PDESolverFactory]

    def __init__(self, args: Namespace):
        self._args = args

        self._components = SolverComponents(args)
        self._benchmark = self._components.benchmark
        self._solver_factories = self._components.solver_factories

    def execute(self):
        if not self._benchmark.has_exact_solution():
            raise ValueError(
                f"Benchmark {self._args.benchmark} of {self._args.problem} has no exact solution. No EOC can be calculated."
            )

        if self._args.quite or len(self._solver_factories) == 0:
            print("WARNING: Nothing to do...")
        else:
            self._print_eocs()

    def _print_eocs(self):
        eocs = []
        titles = []

        for solver_factory in tqdm(
            self._components.solver_factories,
            desc="Calculate solutions",
            unit="solver",
            leave=False,
        ):
            eocs.append(self._calculate_eoc(solver_factory))
            titles.append(solver_factory.eoc_title)

        for eoc, title in zip(eocs, titles):
            print()
            print(title)
            print(eoc)

    def _calculate_eoc(self, solver_factory: PDESolverFactory) -> pd.DataFrame:
        raw_eoc = self._calculate_raw_eoc(solver_factory)

        columns = [
            np.array(["L2", "L2", "L2", "L1", "L1", "L1", "Linf", "Linf", "Linf"]),
            np.array(
                ["DOFs", "error", "eoc", "DOFs", "error", "eoc", "DOFs", "error", "eoc"]
            ),
        ]

        data_frame = pd.DataFrame(raw_eoc, columns=columns)
        return self._format_data_frame(data_frame)

    def _calculate_raw_eoc(self, solver_factory: PDESolverFactory) -> np.ndarray:
        mesh = solver_factory.mesh

        raw_eoc = ScalarEOCCalculator()
        raw_eoc.mesh = mesh
        raw_eoc.solver_factory = solver_factory
        raw_eoc.benchmark = self._components.benchmark

        l2_norm = L2Norm(mesh, solver_factory.cell_quadrature_degree)
        l1_norm = L1Norm(mesh, solver_factory.cell_quadrature_degree)
        linf_norm = LInfinityNorm(mesh, solver_factory.cell_quadrature_degree + 5)
        raw_eoc.add_norm(l2_norm, l1_norm, linf_norm)

        return raw_eoc.eoc(self._args.eoc)

    def _format_data_frame(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        dofs_format = "{:.0f}"
        error_format = "{:.2e}"
        eoc_format = "{:.2f}"

        for norm in ["L2", "L1", "Linf"]:
            data_frame.loc[:, (norm, "DOFs")] = data_frame[(norm, "DOFs")].map(
                dofs_format.format
            )
            data_frame.loc[:, (norm, "error")] = data_frame[(norm, "error")].map(
                error_format.format
            )
            data_frame.loc[:, (norm, "eoc")] = data_frame[(norm, "eoc")].map(
                eoc_format.format
            )

        return data_frame
