from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar

import defaults
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from core.benchmark import Benchmark, NoExactSolutionError
from core.discretization import DiscreteSolution
from core.interpolate import TemporalInterpolator
from core.solver import Solver
from matplotlib import animation
from shallow_water.benchmark import ShallowWaterBenchmark
from tqdm.auto import tqdm

from .command import Command

T = TypeVar("T", float, np.ndarray)


class NothingToAnimateError(Exception):
    ...


class Animator(ABC, Generic[T]):
    _benchmark: Benchmark
    _spatial_grid: np.ndarray
    _temporal_grid: np.ndarray
    _save: Optional[str] = None
    _start_time: float
    _duration: float

    _animatables: List[DiscreteSolution | Callable]
    _values: List[np.ndarray]
    _spatial_grids: List[np.ndarray]
    _temporal_grids: List[np.ndarray]
    _labels: List[Tuple[str, ...]]

    _animation: ...
    _figure: ...

    def __init__(
        self,
        benchmark: Benchmark,
        mesh_size=None,
        time_steps=None,
        save=None,
        start_time=None,
        duration=None,
    ):
        self._benchmark = benchmark
        self._spatial_grid = np.linspace(
            self._benchmark.domain.a,
            self._benchmark.domain.b,
            mesh_size or defaults.PLOT_MESH_SIZE,
        )
        self._temporal_grid = np.linspace(
            self._benchmark.start_time,
            self._benchmark.end_time,
            time_steps or defaults.TIME_STEPS,
        )
        self._save = save
        self._start_time = start_time or self._benchmark.start_time
        self._duration = duration or defaults.DURATION

        self._animatables = []
        self._spatial_grids = []
        self._temporal_grids = []
        self._labels = []

        plt.close()

    @property
    def start_index(self) -> int:
        return np.where(self._temporal_grid >= self._start_time)[0][0]

    @property
    def frames_per_second(self) -> int:
        return int((len(self._temporal_grid) - self.start_index) / self._duration)

    def set_suptitle(self, suptitle: str):
        self._figure.suptitle(
            suptitle,
            fontsize=14,
            fontweight="bold",
        )

    @abstractmethod
    def add_initial_data(self):
        ...

    @abstractmethod
    def add_exact_solution(self):
        ...

    def add_animatable(
        self,
        animatable: DiscreteSolution | Callable,
        *label: str,
    ):
        self._labels.append(label)
        self._animatables.append(animatable)

        if isinstance(animatable, DiscreteSolution):
            self._spatial_grids.append(animatable.grid)
            self._temporal_grids.append(np.array(animatable.time_history))
        else:
            self._spatial_grids.append(self._spatial_grid)
            self._temporal_grids.append(self._temporal_grid)

    def show(self):
        if len(self._animatables) > 0:
            self._setup()
            plt.show()

            if self._save:
                self._animation.save(
                    self._save, writer="ffmpeg", fps=self.frames_per_second
                )
                tqdm.write(f"Animation is saved in '{self._save}'.")

        else:
            raise NothingToAnimateError

    def _setup(self):
        self._temporal_grid = reduce(np.union1d, self._temporal_grids)
        self._adjust_values()
        self._adjust_axes()
        self._build_animation()

    def _adjust_values(self):
        self._values = []
        interpolator = TemporalInterpolator()

        for animatable in self._animatables:
            if isinstance(animatable, DiscreteSolution):
                self._values.append(
                    interpolator(
                        animatable.time_history,
                        animatable.value_history,
                        self._temporal_grid,
                    )
                )
            else:
                self._values.append(
                    np.array(
                        [
                            [animatable(x, t) for x in self._spatial_grid]
                            for t in self._temporal_grid
                        ]
                    )
                )

    def _build_animation(self):
        frames = range(self.start_index, len(self._temporal_grid))
        interval = int(self._duration / len(frames) * 1e3)
        self._animation = animation.FuncAnimation(
            self._figure,
            self._animate,
            interval=interval,
            blit=True,
            frames=range(self.start_index, len(self._temporal_grid)),
        )

    @abstractmethod
    def _animate(self, time_index: int):
        ...

    @abstractmethod
    def _adjust_axes(self):
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class ScalarAnimator(Animator[float]):
    _lines: ...
    _axes: ...
    _time_info: ...

    def __init__(
        self,
        benchmark: Benchmark,
        mesh_size=None,
        time_steps=None,
        save=None,
        start_time=None,
        duration=None,
    ):
        super().__init__(benchmark, mesh_size, time_steps, save, start_time, duration)
        self._values, self._lines = [], []
        self._figure, self._axes = plt.subplots()
        self._time_info = self._axes.text(
            0.05,
            0.95,
            f"T={self._benchmark.start_time:.2f}",
            size=14,
            transform=self._axes.transAxes,
        )

    def add_initial_data(self):
        self.add_animatable(
            lambda x, t: self._benchmark.initial_data(x),
            "$u_0$",
        )

    def add_exact_solution(self):
        self.add_animatable(
            self._benchmark.exact_solution,
            "exact",
        )

    def _animate(self, time_index: int):
        for index, values in enumerate(self._values):
            self._lines[index].set_ydata(values[time_index])

        self._time_info.set_text(f"T={self._temporal_grid[time_index]:.2f}")

        return [*self._lines, self._time_info]

    def _adjust_axes(self):
        for spatial_grid, values, label in zip(
            self._spatial_grids, self._values, self._labels
        ):
            (line,) = self._axes.plot(spatial_grid, values[0], label=label[0])
            self._lines.append(line)

        self._axes.set_xlabel("x")
        self._axes.legend()


class ShallowWaterAnimator(Animator[np.ndarray]):
    _benchmark: ShallowWaterBenchmark

    _height_lines: List
    _discharge_lines: List
    _height_axes: ...
    _discharge_axes: ...
    _time_info: ...

    def __init__(
        self,
        benchmark: ShallowWaterBenchmark,
        mesh_size=None,
        time_steps=None,
        save=None,
        start_time=None,
        duration=None,
    ):
        super().__init__(benchmark, mesh_size, time_steps, save, start_time, duration)
        self._benchmark = benchmark
        self._height_lines, self._discharge_lines = ([], [])
        self._figure, (self._height_axes, self._discharge_axes) = plt.subplots(1, 2)
        self._time_info = self._height_axes.text(
            0.05,
            0.95,
            f"T={self._benchmark.start_time:.2f}",
            size=14,
            transform=self._height_axes.transAxes,
        )

    def add_initial_data(self):
        self.add_animatable(
            lambda x, t: self._benchmark.initial_data(x), "$h_0+b$", "$q_0$"
        )

    def add_exact_solution(self):
        self.add_animatable(self._benchmark.exact_solution, "$h+b$", "$q$")

    def _animate(self, time_index: int):
        for index, values in enumerate(self._values):
            height = self._get_total_height(
                self._spatial_grids[index], values[time_index, :, 0]
            )

            self._height_lines[index].set_ydata(height)
            self._discharge_lines[index].set_ydata(values[time_index, :, 1])

        self._time_info.set_text(f"T={self._temporal_grid[time_index]:.2f}")

        return [*self._height_lines, *self._discharge_lines, self._time_info]

    def _get_total_height(self, grid: np.ndarray, heights: np.ndarray) -> np.ndarray:
        return np.array(
            [heights[i] + self._benchmark.topography(x) for i, x in enumerate(grid)]
        )

    def _adjust_axes(self):
        for spatial_grid, values, label in zip(
            self._spatial_grids, self._values, self._labels
        ):

            height_label = label[0]

            if len(label) == 1:
                discharge_label = label[0]
            else:
                discharge_label = label[1]

            (height_line,) = self._height_axes.plot(
                spatial_grid, values[0].T[0], label=height_label
            )
            (discharge_line,) = self._discharge_axes.plot(
                spatial_grid, values[0].T[1], label=discharge_label
            )
            self._height_lines.append(height_line)
            self._discharge_lines.append(discharge_line)

        self._add_topography()

        additional_border_factor = 0.05
        max_values, min_values = self._get_min_max_values()

        self._height_axes.set_xlabel("x")
        self._height_axes.set_ylabel("h+b")
        self._height_axes.set_ylim(
            top=max_values[0] + additional_border_factor * max_values[0]
        )
        self._height_axes.legend()

        self._discharge_axes.set_xlabel("x")
        self._discharge_axes.set_ylabel("discharge")
        self._discharge_axes.set_ylim(
            bottom=min_values[1] - additional_border_factor * abs(min_values[1]),
            top=max_values[1] + additional_border_factor * abs(max_values[1]),
        )
        self._discharge_axes.legend()

    def _get_min_max_values(self) -> Tuple[np.ndarray, np.ndarray]:
        max_values = np.array([np.max(value, axis=(0, 1)) for value in self._values])
        max_values = np.max(max_values, axis=0)
        min_values = np.array([np.min(value, axis=(0, 1)) for value in self._values])
        min_values = np.min(min_values, axis=0)

        return max_values, min_values

    def _add_topography(self):
        topography_values = np.array(
            [self._benchmark.topography(x) for x in self._spatial_grid]
        )
        self._height_axes.plot(self._spatial_grid, topography_values, label="$b$")


class Animate(Command):
    _benchmark: Benchmark
    _solver: Sequence[Solver]
    _animator: Animator
    _initial: bool

    def __init__(
        self,
        benchmark: Benchmark,
        solver: Solver | Sequence[Solver],
        animator: Animator,
        initial=False,
    ):
        self._benchmark = benchmark
        self._animator = animator
        self._initial = initial

        if isinstance(solver, Solver):
            self._solver = [solver]
        else:
            self._solver = solver

    def execute(self):
        self._add_animations()

        try:
            self._animator.show()
        except NothingToAnimateError:
            tqdm.write("WARNING: Nothing to animate...")

    def _add_animations(self):
        self._add_discrete_solutions()
        self._add_exact_solution()

        if self._initial:
            self._animator.add_initial_data()

    def _add_exact_solution(self):
        try:
            self._benchmark.exact_solution(0, 0)
            self._animator.add_exact_solution()
        except NoExactSolutionError as error:
            tqdm.write("WARNING: " + str(error))

    def _add_discrete_solutions(self):
        for solver in self._solver:
            self._animator.add_animatable(solver._solution, solver.short)
