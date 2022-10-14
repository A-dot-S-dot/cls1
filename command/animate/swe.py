from typing import Callable, List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from benchmark import SWEBenchmark
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .animator import SolutionAnimator


class SWEFunctionAnimator(SolutionAnimator[np.ndarray]):
    _benchmark: SWEBenchmark
    _bottom_topography: np.ndarray
    _values: List[np.ndarray]
    _grids: List[np.ndarray]
    _times: np.ndarray
    _height_lines: List[Line2D]
    _discharge_lines: List[Line2D]
    _animation: animation.FuncAnimation
    _figure: plt.Figure
    _height_axes: plt.Axes
    _discharge_axes: plt.Axes
    _time_info: Text

    def __init__(self, benchmark: SWEBenchmark):
        self._benchmark = benchmark
        self._grids, self._values, self._height_lines, self._discharge_lines = (
            [],
            [],
            [],
            [],
        )
        self._figure, (self._height_axes, self._discharge_axes) = plt.subplots(1, 2)
        self._time_info = self._height_axes.text(
            0.05,
            0.95,
            f"T={self._benchmark.start_time:.2f}",
            size=14,
            transform=self._height_axes.transAxes,
        )

    def _animate(self, time_index: int):
        for index, values in enumerate(self._values):
            height = self._get_total_height(
                self._grids[index], values[time_index, :, 0]
            )

            self._height_lines[index].set_ydata(height)
            self._discharge_lines[index].set_ydata(values[time_index, :, 1])

        self._time_info.set_text(f"T={self.temporal_grid[time_index]:.2f}")

        return [*self._height_lines, *self._discharge_lines, self._time_info]

    def _get_total_height(self, grid: np.ndarray, heights: np.ndarray) -> np.ndarray:
        return np.array(
            [heights[i] + self._benchmark.topography(x) for i, x in enumerate(grid)]
        )

    def set_suptitle(self, suptitle: str):
        self._figure.suptitle(
            suptitle,
            fontsize=14,
            fontweight="bold",
        )

    def add_initial_data(self):
        self.add_function(
            lambda x, t: self._benchmark.initial_data(x), "$h_0$", "$q_0$"
        )

    def add_exact_solution(self):
        self.add_function(self._benchmark.exact_solution, "$h+b$", "$q$")

    def add_function(
        self,
        function: Callable[[float, float], np.ndarray],
        height_label: str,
        discharge_label: str,
    ):
        function_values = np.array(
            [[function(x, t) for x in self.spatial_grid] for t in self.temporal_grid]
        )
        self.add_function_values(
            self.spatial_grid,
            self.temporal_grid,
            function_values,
            height_label,
            discharge_label,
        )

    def add_function_values(
        self,
        spatial_grid: np.ndarray,
        temporal_grid: np.ndarray,
        function_values: np.ndarray,
        height_label: str,
        discharge_label: str,
    ):

        (height_line,) = self._height_axes.plot(
            spatial_grid, function_values[0].T[0], label=height_label
        )
        (discharge_line,) = self._discharge_axes.plot(
            spatial_grid, function_values[0].T[1], label=discharge_label
        )

        self.temporal_grid = temporal_grid
        self._grids.append(spatial_grid)
        self._values.append(function_values)
        self._height_lines.append(height_line)
        self._discharge_lines.append(discharge_line)

        self._animation_available = True

    def _setup(self):
        self._add_topography()
        self._add_animation()
        self._add_axes_descriptions()

    def _add_topography(self):
        topography_values = np.array(
            [self._benchmark.topography(x) for x in self.spatial_grid]
        )
        self._height_axes.plot(self.spatial_grid, topography_values, label="$b$")

    def _add_animation(self):
        self._animation = animation.FuncAnimation(
            self._figure,
            self._animate,
            interval=self.interval,
            blit=True,
            frames=len(self.temporal_grid),
        )

    def _add_axes_descriptions(self):
        self._height_axes.set_xlabel("x")
        self._height_axes.set_ylabel("h+b")
        self._height_axes.legend()

        self._discharge_axes.set_xlabel("x")
        self._discharge_axes.set_ylabel("discharge")
        self._discharge_axes.legend()
