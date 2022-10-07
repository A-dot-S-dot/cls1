from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from benchmark import SWEBenchmark

from .plotter import SolutionPlotter


class SWEFunctionPlotter(SolutionPlotter[np.ndarray]):
    _benchmark: SWEBenchmark
    _figure: plt.Figure
    _height_axes: plt.Axes
    _discharge_axes: plt.Axes

    def __init__(self, benchmark: SWEBenchmark):
        self._benchmark = benchmark
        self._figure, (self._height_axes, self._discharge_axes) = plt.subplots(1, 2)

    def set_suptitle(self, suptitle: str):
        self._figure.suptitle(suptitle, fontsize=14, fontweight="bold")

    def add_initial_data(self):
        self.add_function(self._benchmark.initial_data, "$h_0$", "$q_0")

    def add_exact_solution(self):
        end_time = self._benchmark.end_time
        self.add_function(
            self._benchmark.exact_solution_at_end_time,
            f"$h+b(t={end_time:.1f})$",
            f"$q(t={end_time:.1f})$",
        )

    def add_function(
        self,
        function: Callable[[float], np.ndarray],
        height_label: str,
        discharge_label: str,
    ):
        swe_values = np.array([function(x) for x in self.grid])
        self.add_function_values(self.grid, swe_values.T, height_label, discharge_label)

    def add_function_values(
        self,
        grid: np.ndarray,
        swe_values: np.ndarray,
        height_label: str,
        discharge_label: str,
    ):
        height = swe_values[0]
        discharge = swe_values[1]

        self.add_height(grid, height, height_label)
        self.add_discharge(grid, discharge, discharge_label)

        self._plot_available = True

    def add_height(self, grid: np.ndarray, height: np.ndarray, label: str):
        total_height = height + [self._benchmark.topography(x) for x in grid]
        self._height_axes.plot(grid, total_height, label=label)

    def add_discharge(self, grid: np.ndarray, discharge: np.ndarray, label: str):
        self._discharge_axes.plot(grid, discharge, label=label)

    def _setup(self):
        self.add_topography()

        self._height_axes.set_xlabel("x")
        self._height_axes.set_ylabel("h+b")
        self._height_axes.legend()

        self._discharge_axes.set_xlabel("x")
        self._discharge_axes.set_ylabel("discharge")
        self._discharge_axes.legend()

    def add_topography(self):
        topography_values = np.array([self._benchmark.topography(x) for x in self.grid])
        self._height_axes.plot(self.grid, topography_values, label="$b$")
