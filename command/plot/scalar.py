from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from benchmark import Benchmark

from .plotter import SolutionPlotter


class ScalarFunctionPlotter(SolutionPlotter[float]):
    _benchmark: Benchmark

    def __init__(self, benchmark: Benchmark):
        self._benchmark = benchmark

    def set_suptitle(self, suptitle: str):
        plt.suptitle(suptitle, fontsize=14, fontweight="bold")

    def add_initial_data(self):
        self.add_function(
            self._benchmark.initial_data,
            "$u_0$",
        )

    def add_exact_solution(self):
        self.add_function(
            self._benchmark.exact_solution_at_end_time,
            f"$u(t={self._benchmark.end_time:.1f})$",
        )

    def add_function(self, function: Callable[[float], float], label: str):
        function_values = np.array([function(x) for x in self.grid])
        self.add_function_values(self.grid, function_values, label)

    def add_function_values(
        self, grid: np.ndarray, function_values: np.ndarray, label: str
    ):
        plt.plot(grid, function_values, label=label)
        self._plot_available = True

    def _setup(self):
        plt.xlabel("x")
        plt.legend()
