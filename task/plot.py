from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from factory.pde_solver_factory import PDESolverFactory
from factory.solver_components import SolverComponents
from math_type import FunctionRealToReal
from mesh import Interval
from tqdm import tqdm

from .task import Task


class FunctionPlotter:
    _grid: np.ndarray
    _title: str
    _suptitle: str

    def __init__(self, interval: Interval):
        self._grid = np.linspace(interval.a, interval.b)

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, title: str):
        self._title = title
        plt.title(title)

    @property
    def suptitle(self) -> str:
        return self._suptitle

    @suptitle.setter
    def suptitle(self, suptitle: str):
        self._suptitle = suptitle
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
    _plotter: FunctionPlotter
    _target_time: float
    _time_steps_number: int

    def __init__(self, args: Namespace):
        self._args = args
        self._components = SolverComponents(args)
        mesh = self._components.mesh
        benchmark = self._components.benchmark

        self._plotter = FunctionPlotter(mesh.domain)
        self._target_time = benchmark.T
        self._time_steps_number = len(mesh) * args.courant_factor

    def execute(self):
        self._add_functions()

        if not self._args.no_plot:
            self._plotter.show()

    def _add_functions(self):
        self._add_exact_solution()
        self._add_discrete_solutions()

    def _add_exact_solution(self):
        benchmark = self._components.benchmark

        if benchmark.has_exact_solution():
            self._plotter.add_function(
                benchmark.exact_solution_at_T,
                f"$u(\cdot, {self._target_time:.2f})$",
            )
        elif len(self._components.solver_factories) == 0:
            raise ValueError("Nothing to plot.")

    def _add_discrete_solutions(self):
        for solver_factory in tqdm(
            self._components.solver_factories,
            desc="Calculate solutions",
            unit="solver",
            leave=False,
        ):
            self._add_discrete_solution(solver_factory)

    def _add_discrete_solution(self, solver_factory: PDESolverFactory):
        solver = solver_factory.solver
        solver.solve(self._target_time, self._time_steps_number)

        label = solver_factory.plot_label

        self._plotter.add_function(solver_factory.discrete_solution, label)
