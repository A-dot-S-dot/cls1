from typing import Dict, Iterable

import factory
import numpy as np
import pde_solver.system_vector as sv
from benchmark import SWEBenchmark
from factory.pde_solver_factory import PDESolverFactory
from pde_solver.discrete_solution import DiscreteSolution, DiscreteSolutionObservable
from pde_solver.interpolate import CellAverageInterpolator
from pde_solver.solver.finite_volume import SWEGodunovSolver
from pde_solver.solver_space import FiniteVolumeSpace


class SWEGodunovSolverFactory(PDESolverFactory[np.ndarray]):
    """Superclass, some methods must be implemented by subclasses."""

    benchmark: SWEBenchmark

    _solver: SWEGodunovSolver
    _volume_space: FiniteVolumeSpace
    _bottom_topography: np.ndarray

    def _setup_solver(self):
        self._build_solver()
        self._build_space()
        self._interpolate()
        self._build_tqdm_kwargs()
        self._build_numerical_flux()
        self._build_time_stepping()

    def _build_solver(self):
        self._solver = SWEGodunovSolver()
        self._solver.mesh = self.mesh

    def _build_space(self):
        self._volume_space = FiniteVolumeSpace(self.mesh)

    def _interpolate(self):
        interpolator = CellAverageInterpolator(self.mesh, 2)
        height = interpolator.interpolate(lambda x: self.benchmark.initial_data(x)[0])
        discharge = interpolator.interpolate(
            lambda x: self.benchmark.initial_data(x)[1]
        )
        self._bottom_topography = interpolator.interpolate(self.benchmark.topography)

        discrete_solution = DiscreteSolution(
            self.benchmark.start_time, np.array([height, discharge]).T
        )
        self._solver.solution = DiscreteSolutionObservable(discrete_solution)

    def _build_tqdm_kwargs(self):
        self._solver.tqdm_kwargs = self.tqdm_kwargs

    def _build_numerical_flux(self):
        numerical_flux = sv.SWEGodunovNumericalFlux()
        numerical_flux.volume_space = self._volume_space
        numerical_flux.bottom_topography = self._bottom_topography

        self._solver.numerical_flux = numerical_flux

    def _build_time_stepping(self):
        time_stepping = factory.TIME_STEPPING_FACTORY.godunov_time_stepping
        time_stepping.start_time = self.benchmark.start_time
        time_stepping.end_time = self.benchmark.end_time
        time_stepping.mesh = self.mesh
        time_stepping.cfl_number = self.attributes.cfl_number
        time_stepping.numerical_flux = self._solver.numerical_flux

        self._solver.time_stepping = time_stepping

    @property
    def grid(self) -> np.ndarray:
        return self._volume_space.cell_centers

    @property
    def cell_quadrature_degree(self) -> int:
        return 1

    @property
    def dimension(self) -> int:
        return self._volume_space.dimension

    @property
    def plot_label(self) -> Iterable[str]:
        if self.attributes.label:
            return [self.attributes.label]
        else:
            return ["godunov", "godunov"]

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = {
            "desc": "Godunov",
            "leave": False,
            "postfix": {
                "cfl_number": self.attributes.cfl_number,
                "DOFs": self.dimension,
            },
        }

        return tqdm_kwargs

    @property
    def eoc_title(self) -> str:
        title = self.info
        return title + "\n" + "-" * len(title)

    @property
    def info(self) -> str:
        return f"Godunov (cfl={self.attributes.cfl_number})"
