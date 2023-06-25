from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar, Any

import argparse
import core
import defaults
import finite_volume.shallow_water as swe
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm.auto import tqdm

from .calculate import Calculate
from .command import Command
from .plot import PlotParser

T = TypeVar("T", float, np.ndarray)


class NothingToAnimateError(Exception):
    ...


class Animator(ABC, Generic[T]):
    _benchmark: core.Benchmark
    _spatial_grid: np.ndarray
    _temporal_grid: np.ndarray
    _save: Optional[str] = None
    _start_time: float
    _duration: float
    _show: bool

    _animatables: List[core.DiscreteSolutionWithHistory | Callable]
    _values_per_solver: List[np.ndarray]
    _spatial_grids: List[np.ndarray]
    _temporal_grids: List[np.ndarray]
    _labels: List[Tuple[str, ...]]

    _animation: ...
    _figure: ...

    def __init__(
        self,
        benchmark: core.Benchmark,
        mesh_size=None,
        time_steps=None,
        save=None,
        start_time=None,
        duration=None,
        show=True,
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
        self._show = show

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
        animatable: core.DiscreteSolutionWithHistory | Callable,
        *label: str,
    ):
        self._labels.append(label)
        self._animatables.append(animatable)

        if isinstance(animatable, core.DiscreteSolutionWithHistory):
            self._spatial_grids.append(animatable.grid)
            self._temporal_grids.append(np.array(animatable.time_history))
        else:
            self._spatial_grids.append(self._spatial_grid)
            self._temporal_grids.append(self._temporal_grid)

    def show(self):
        if len(self._animatables) > 0:
            self._setup()

            if self._save:
                self._animation.save(
                    self._save, writer="ffmpeg", fps=self.frames_per_second
                )
                tqdm.write(f"Animation is saved in '{self._save}'.")

            plt.show() if self._show else plt.close()
        else:
            raise NothingToAnimateError

    def _setup(self):
        self._temporal_grid = reduce(np.union1d, self._temporal_grids)
        self._adjust_values()
        self._adjust_axes()
        self._build_animation()

    def _adjust_values(self):
        self._values_per_solver = []
        interpolator = core.TemporalInterpolator()

        for animatable in self._animatables:
            if isinstance(animatable, core.DiscreteSolutionWithHistory):
                self._values_per_solver.append(
                    interpolator(
                        animatable.time_history,
                        animatable.value_history,
                        self._temporal_grid,
                    )
                )
            else:
                self._values_per_solver.append(
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

    def __init__(self, benchmark: core.Benchmark, **kwargs):
        super().__init__(benchmark, **kwargs)
        self._values_per_solver, self._lines = [], []
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
        for index, values in enumerate(self._values_per_solver):
            self._lines[index].set_ydata(values[time_index])

        self._time_info.set_text(f"T={self._temporal_grid[time_index]:.2f}")

        return [*self._lines, self._time_info]

    def _adjust_axes(self):
        for spatial_grid, values, label in zip(
            self._spatial_grids, self._values_per_solver, self._labels
        ):
            (line,) = self._axes.plot(spatial_grid, values[0], label=label[0])
            self._lines.append(line)

        self._axes.set_xlabel("x")
        self._axes.legend()


class ShallowWaterAnimator(Animator[np.ndarray]):
    _benchmark: swe.ShallowWaterBenchmark

    _height_lines: List
    _discharge_lines: List
    _height_axes: ...
    _discharge_axes: ...
    _velocity_axes: ...
    _time_info: ...

    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        super().__init__(benchmark, **kwargs)
        self._benchmark = benchmark
        self._height_lines, self._discharge_lines, self._velocity_lines = ([], [], [])
        self._figure, (
            self._height_axes,
            self._discharge_axes,
            self._velocity_axes,
        ) = plt.subplots(1, 3)
        self._time_info = self._height_axes.text(
            0.05,
            0.95,
            f"T={self._benchmark.start_time:.2f}",
            size=14,
            transform=self._height_axes.transAxes,
        )

    def add_initial_data(self):
        self.add_animatable(
            lambda x, t: self._benchmark.initial_data(x), "$h_0+b$", "$q_0$", "$u_0$"
        )

    def add_exact_solution(self):
        self.add_animatable(self._benchmark.exact_solution, "$h+b$", "$q$", "$u$")

    def _animate(self, time_index: int):
        for index, values in enumerate(self._values_per_solver):
            height = self._get_total_height(
                self._spatial_grids[index], swe.get_height(values[time_index])
            )

            self._height_lines[index].set_ydata(height)
            self._discharge_lines[index].set_ydata(
                swe.get_discharge(values[time_index])
            )
            self._velocity_lines[index].set_ydata(swe.get_velocity(values[time_index]))

        self._time_info.set_text(f"T={self._temporal_grid[time_index]:.2f}")

        return [
            *self._height_lines,
            *self._discharge_lines,
            *self._velocity_lines,
            self._time_info,
        ]

    def _get_total_height(self, grid: np.ndarray, height: np.ndarray) -> np.ndarray:
        return height + np.array([self._benchmark.bathymetry(x) for x in grid])

    def _adjust_axes(self):
        for spatial_grid, values, label in zip(
            self._spatial_grids, self._values_per_solver, self._labels
        ):
            height_label = label[0]

            if len(label) == 1:
                discharge_label = label[0]
                velocity_label = label[0]
            else:
                discharge_label = label[1]
                velocity_label = label[2]

            (height_line,) = self._height_axes.plot(
                spatial_grid, swe.get_height(values[0]), label=height_label
            )
            (discharge_line,) = self._discharge_axes.plot(
                spatial_grid, swe.get_discharge(values[0]), label=discharge_label
            )
            (velocity_line,) = self._velocity_axes.plot(
                spatial_grid, swe.get_velocity(values[0]), label=velocity_label
            )
            self._height_lines.append(height_line)
            self._discharge_lines.append(discharge_line)
            self._velocity_lines.append(velocity_line)

        self._add_bathymetry()

        additional_border_factor = 0.05
        values_min, values_max = self._get_min_max_values()
        v_min, v_max = self._get_min_max_velocities()

        self._height_axes.set_xlabel("x")
        self._height_axes.set_ylabel("h+b")
        self._height_axes.set_ylim(
            top=swe.get_height(values_max)
            + additional_border_factor * swe.get_height(values_max)
        )
        self._height_axes.legend()

        self._discharge_axes.set_xlabel("x")
        self._discharge_axes.set_ylabel("discharge")
        self._discharge_axes.set_ylim(
            bottom=swe.get_discharge(values_min)
            - additional_border_factor * abs(swe.get_discharge(values_min)),
            top=swe.get_discharge(values_max)
            + additional_border_factor * abs(swe.get_discharge(values_max)),
        )
        self._discharge_axes.legend()

        self._velocity_axes.set_xlabel("x")
        self._velocity_axes.set_ylabel("velocity")
        self._velocity_axes.set_ylim(
            bottom=v_min - additional_border_factor * abs(v_min),
            top=v_max + additional_border_factor * abs(v_max),
        )
        self._velocity_axes.legend()

    def _get_min_max_values(self) -> Tuple[np.ndarray, np.ndarray]:
        values_max = np.array(
            [np.max(value, axis=(0, 1)) for value in self._values_per_solver]
        )
        values_max = np.max(values_max, axis=0)
        values_min = np.array(
            [np.min(value, axis=(0, 1)) for value in self._values_per_solver]
        )
        values_min = np.min(values_min, axis=0)

        return values_min, values_max

    def _get_min_max_velocities(self) -> Tuple[float, float]:
        velocities_per_solver = [
            swe.get_velocities(*value) for value in self._values_per_solver
        ]

        return (
            np.min([np.min(v) for v in velocities_per_solver]),
            np.max([np.max(v) for v in velocities_per_solver]),
        )

    def _add_bathymetry(self):
        bathymetry_values = np.array(
            [self._benchmark.bathymetry(x) for x in self._spatial_grid]
        )
        self._height_axes.plot(self._spatial_grid, bathymetry_values, label="$b$")


class Animate(Command):
    _benchmark: core.Benchmark
    _solver: Sequence[core.Solver]
    _animator: Animator
    _initial: bool
    _solver_executed: bool
    _write_warnings: bool

    def __init__(
        self,
        benchmark: core.Benchmark,
        solver: core.Solver | Sequence[core.Solver],
        animator: Animator,
        initial=False,
        solver_executed=False,
        write_warnings=True,
    ):
        self._benchmark = benchmark
        self._animator = animator
        self._initial = initial
        self._solver_executed = solver_executed
        self._write_warnings = write_warnings

        if isinstance(solver, core.Solver):
            self._solver = [solver]
        else:
            self._solver = solver

    def execute(self):
        if not self._solver_executed:
            self._calculate_solutions()

        self._add_animations()

        try:
            self._animator.show()
        except NothingToAnimateError:
            if self._write_warnings:
                tqdm.write("WARNING: Nothing to animate...")

    def _calculate_solutions(self):
        for solver in tqdm(self._solver, desc="Calculate", unit="solver", leave=False):
            try:
                Calculate(solver).execute()
            except Exception as error:
                if self._write_warnings:
                    tqdm.write(
                        f"WARNING: {str(error)} Solution calculated until t={solver.solution.time:.3e}."
                    )

    def _add_animations(self):
        self._add_discrete_solutions()
        self._add_exact_solution()

        if self._initial:
            self._animator.add_initial_data()

    def _add_exact_solution(self):
        try:
            self._benchmark.exact_solution(0, 0)
            self._animator.add_exact_solution()
        except Exception as error:
            if self._write_warnings:
                tqdm.write("WARNING: " + str(error))

    def _add_discrete_solutions(self):
        for solver in self._solver:
            self._animator.add_animatable(solver.solution, solver.short)


class AnimateParser(PlotParser):
    _benchmark_parser = "animate"
    _save_default = defaults.ANIMATE_TARGET

    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "animate",
            help="Calculate and animate solutions.",
            description="Animate benchmarks and computed solutions.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_command_arguments(self, parser):
        self._add_plot_mesh_size(parser)
        self._add_initial_data(parser)
        self._add_save(parser)
        self._add_hide(parser)
        self._add_time_steps(parser)
        self._add_start_time(parser)
        self._add_duration(parser)

    def _add_time_steps(self, parser):
        parser.add_argument(
            "--time_steps",
            help="Specify how many time steps should be at least simulated.",
            type=core.positive_int,
            metavar="<number>",
            default=defaults.TIME_STEPS,
        )

    def _add_start_time(self, parser):
        parser.add_argument(
            "-t",
            "--start-time",
            help="Set start time for animation.",
            type=core.positive_float,
            metavar="<time>",
        )

    def _add_duration(self, parser):
        parser.add_argument(
            "--duration",
            help="""Specifies how many second an animation should last.""",
            type=core.positive_float,
            default=defaults.DURATION,
            metavar="<factor>",
        )

    def postprocess(self, arguments):
        self._adjust_end_time(arguments)
        self._add_save_history_argument(arguments)
        self._build_solver(arguments)
        self._build_animator(arguments)
        arguments.command = Animate

        del arguments.problem

    def _build_animator(self, arguments):
        animator = {
            "advection": ScalarAnimator,
            "burgers": ScalarAnimator,
            "swe": ShallowWaterAnimator,
        }

        arguments.animator = animator[arguments.problem](
            arguments.benchmark,
            mesh_size=arguments.mesh_size,
            time_steps=arguments.time_steps,
            save=arguments.save,
            start_time=arguments.start_time,
            duration=arguments.duration,
            show=arguments.show,
        )

        del arguments.mesh_size
        del arguments.time_steps
        del arguments.save
        del arguments.start_time
        del arguments.duration
        del arguments.show
