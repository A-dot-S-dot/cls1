from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from benchmark import Benchmark
from factory.pde_solver_factory import PDESolverFactory
from factory.solver_components import SolverComponents
from math_type import FunctionRealToReal
from mesh import Interval
from tqdm import tqdm

from .task import Task


class FunctionPlotter:
    _grid: np.ndarray
    _suptitle: str

    def set_grid(self, interval: Interval, dof_number: int = 0):

        grid_number = max(500, dof_number * 2)
        self._grid = np.linspace(interval.a, interval.b, grid_number)

    @property
    def title(self) -> str:
        ...

    @title.setter
    def title(self, title: str):
        plt.title(title)

    @property
    def suptitle(self) -> str:
        ...

    @suptitle.setter
    def suptitle(self, suptitle: str):
        plt.suptitle(suptitle, fontsize=14, fontweight="bold")

    def add_function(self, function: FunctionRealToReal, function_label: str):
        function_values = np.array([function(x) for x in self._grid])

        plt.plot(self._grid, function_values, label=function_label)

    def save(self, path="output/plot.png"):
        self._setup()
        plt.savefig(path)

    def show(self):
        self._setup()
        plt.show()
        plt.close()

    def _setup(self):
        plt.xlabel("x")
        plt.legend()


class PlotTask(Task):
    _args: Namespace
    _components: SolverComponents
    _benchmark: Benchmark
    _plotter: FunctionPlotter

    def __init__(self, args: Namespace):
        self._args = args
        self._components = SolverComponents(args)
        self._benchmark = self._components.benchmark
        self._build_plotter()

    def _build_plotter(self):
        self._plotter = FunctionPlotter()

        domain = self._benchmark.domain
        dofs_number = self._get_maximal_polynomial_degree() * self._args.elements_number

        self._plotter.set_grid(domain, dofs_number)

    def _get_maximal_polynomial_degree(self) -> int:
        maximal_polynomial_degree = 0
        for solver_args in self._args.solver:
            maximal_polynomial_degree = max(
                solver_args.polynomial_degree, maximal_polynomial_degree
            )

        return maximal_polynomial_degree

    def execute(self):
        self._add_functions()
        self._plotter.title = f"{len(self._components.mesh)} elements"

        if not self._args.no_plot:
            self._plotter.show()

    def _add_functions(self):
        self._add_exact_solution()
        self._add_discrete_solutions()

    def _add_exact_solution(self):
        benchmark = self._components.benchmark

        if benchmark.has_exact_solution():
            self._plotter.add_function(
                benchmark.exact_solution_at_end_time,
                f"$u(\cdot, {self._benchmark.end_time:.1f})$",
            )
        elif len(self._components.solver_factories) == 0:
            raise ValueError("Nothing to plot.")

    def _add_discrete_solutions(self):
        for solver_factory in tqdm(
            self._components.solver_factories,
            desc="Calculate solutions",
            unit="solver",
            position=0,
            leave=False,
        ):
            self._add_discrete_solution(solver_factory)

    def _add_discrete_solution(self, solver_factory: PDESolverFactory):
        solver = solver_factory.solver
        solver.solve()

        label = solver_factory.plot_label
        tqdm.write(
            f"Solved {solver_factory.info} with {solver_factory.dofs} DOFs and {solver.time_steps} time steps."
        )

        self._plotter.add_function(solver_factory.discrete_solution, label)
