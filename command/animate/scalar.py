from typing import Callable, List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from benchmark import Benchmark
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .animator import SolutionAnimator


class ScalarFunctionAnimator(SolutionAnimator[float]):
    _benchmark: Benchmark
    _values: List[np.ndarray]
    _times: np.ndarray
    _lines: List[Line2D]
    _animation: animation.FuncAnimation
    _figure: plt.Figure
    _axes: plt.Axes
    _time_info: Text

    def __init__(self, benchmark: Benchmark):
        self._benchmark = benchmark
        self._values, self._lines = [], []
        self._figure, self._axes = plt.subplots()
        self._time_info = self._axes.text(
            0.05,
            0.95,
            f"T={self._benchmark.start_time:.2f}",
            size=14,
            transform=self._axes.transAxes,
        )

    def _animate(self, time_index: int):
        for index, values in enumerate(self._values):
            self._lines[index].set_ydata(values[time_index])

        self._time_info.set_text(f"T={self.temporal_grid[time_index]:.2f}")

        return [*self._lines, self._time_info]

    def set_suptitle(self, suptitle: str):
        self._figure.suptitle(
            suptitle,
            fontsize=14,
            fontweight="bold",
        )

    def add_initial_data(self):
        self.add_function(
            lambda x, t: self._benchmark.initial_data(x),
            "$u_0$",
        )

    def add_exact_solution(self):
        self.add_function(
            self._benchmark.exact_solution,
            "exact",
        )

    def add_function(
        self,
        function: Callable[[float, float], float],
        label: str,
    ):
        function_values = np.array(
            [[function(x, t) for x in self.spatial_grid] for t in self.temporal_grid]
        )
        self.add_function_values(
            self.spatial_grid, self.temporal_grid, function_values, label
        )

    def add_function_values(
        self,
        spatial_grid: np.ndarray,
        temporal_grid: np.ndarray,
        function_values: np.ndarray,
        label: str,
    ):

        (line,) = self._axes.plot(spatial_grid, function_values[0], label=label)

        self.temporal_grid = temporal_grid
        self._values.append(function_values)
        self._lines.append(line)

        self._animation_available = True

    def _setup(self):
        self._animation = animation.FuncAnimation(
            self._figure,
            self._animate,
            interval=self.interval,
            blit=True,
            frames=range(self.start_index, len(self.temporal_grid)),
        )
        self._axes.set_xlabel("x")
        self._axes.legend()
