from typing import Dict

import factory
import numpy as np
import pde_solver.system_flux as sf
from benchmark import SWEBenchmark
from factory.pde_solver_factory import PDESolverFactory
from pde_solver import ExactCoarseSolver, FiniteVolumeSolver
from pde_solver.discrete_solution import DiscreteSolutionObservable
from pde_solver.interpolate import CellAverageInterpolator
from pde_solver.solver_space import CoarsenedFiniteVolumeSpace, FiniteVolumeSpace
from pde_solver.time_stepping import CFLCheckedFlux


class SWESolverFactory(PDESolverFactory[np.ndarray]):
    benchmark: SWEBenchmark

    _solver: FiniteVolumeSolver
    _solver_space: FiniteVolumeSpace
    _bottom_topography: np.ndarray

    @property
    def solver(self) -> FiniteVolumeSolver:
        self._setup_solver()
        return self._solver

    def _setup_solver(self):
        self._build_solver()
        self._build_space()
        self._interpolate()
        self._build_tqdm_kwargs()
        self._build_time_stepping()
        self._build_numerical_flux()

    def _build_solver(self):
        self._solver = FiniteVolumeSolver()

    def _build_space(self):
        self._solver_space = FiniteVolumeSpace(self.mesh)
        self._solver.solver_space = FiniteVolumeSpace(self.mesh)

    def _interpolate(self):
        interpolator = CellAverageInterpolator(self.mesh, 2)
        height = interpolator.interpolate(lambda x: self.benchmark.initial_data(x)[0])
        discharge = interpolator.interpolate(
            lambda x: self.benchmark.initial_data(x)[1]
        )
        self._bottom_topography = interpolator.interpolate(self.benchmark.topography)

        self._solver.solution = DiscreteSolutionObservable(
            self.benchmark.start_time, np.array([height, discharge]).T
        )

    def _build_tqdm_kwargs(self):
        self._solver.tqdm_kwargs = self.tqdm_kwargs

    def _build_time_stepping(self):
        raise NotImplementedError

    def _build_numerical_flux(self):
        raise NotImplementedError

    @property
    def grid(self) -> np.ndarray:
        return self._solver_space.cell_centers

    @property
    def cell_quadrature_degree(self) -> int:
        return 1


class SWEGodunovSolverFactory(SWESolverFactory):
    _plot_label = ["godunov", "godunov"]
    _intermediate_velocities: sf.SWEIntermediateVelocities

    def _build_time_stepping(self):
        self._build_intermediate_velocities()

        time_stepping = factory.TIME_STEPPING_FACTORY.get_godunov_time_stepping(
            self.attributes.adaptive
        )
        time_stepping.start_time = self.benchmark.start_time
        time_stepping.end_time = self.benchmark.end_time
        time_stepping.mesh = self.mesh
        time_stepping.cfl_number = self.attributes.cfl_number
        time_stepping.intermediate_velocities = self._intermediate_velocities
        time_stepping.update_time_step()

        self._solver.time_stepping = time_stepping

    def _build_intermediate_velocities(self):
        self._intermediate_velocities = sf.SWEIntermediateVelocities(
            self._solver.solution
        )
        self._intermediate_velocities.volume_space = self._solver_space
        self._intermediate_velocities.update()

    def _build_numerical_flux(self):
        numerical_flux = sf.SWEGodunovNumericalFlux()
        numerical_flux.volume_space = self._solver_space
        numerical_flux.cell_flux_calculator = self._build_cell_flux_calculator()

        self._solver.numerical_flux = CFLCheckedFlux(
            numerical_flux, self._solver.time_stepping
        )

    def _build_cell_flux_calculator(self) -> sf.GodunovCellFluxesCalculator:
        cell_flux_calculator = sf.GodunovCellFluxesCalculator()
        cell_flux_calculator.bottom_topography = self._bottom_topography
        cell_flux_calculator.volume_space = self._solver_space
        cell_flux_calculator.intermediate_velocities = self._intermediate_velocities
        cell_flux_calculator.source_term_discretization = (
            self._build_source_term_discretization()
        )

        return cell_flux_calculator

    def _build_source_term_discretization(self) -> sf.SourceTermDiscretization:
        discretization = sf.NaturalSourceTermDiscretization()
        discretization.step_length = self._solver_space.mesh.step_length
        return discretization

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = {
            "desc": "Godunov",
            "leave": False,
            "postfix": {
                "cfl_number": self.attributes.cfl_number,
                "DOFs": self.dimension,
                "adaptive": self.attributes.adaptive,
            },
        }

        return tqdm_kwargs

    @property
    def info(self) -> str:
        return f"Godunov (cfl={self.attributes.cfl_number}, adaptive={self.attributes.adaptive})"


class SWECoarseExactSolverFactory(SWEGodunovSolverFactory):
    _plot_label = ["coarse-exact", "coarse-exact"]
    _solver: ExactCoarseSolver

    def _build_solver(self):
        self._solver = ExactCoarseSolver()
        self._solver.coarsening_degree = self.attributes.coarsening_degree

    def _build_time_stepping(self):
        self._build_intermediate_velocities()

        time_stepping = factory.TIME_STEPPING_FACTORY.get_godunov_time_stepping(
            adaptive=False
        )
        time_stepping.start_time = self.benchmark.start_time
        time_stepping.end_time = self.benchmark.end_time
        time_stepping.mesh = self.mesh
        time_stepping.cfl_number = self.attributes.cfl_number
        time_stepping.intermediate_velocities = self._intermediate_velocities
        time_stepping.update_time_step()

        self._solver.time_stepping = time_stepping

    def _build_intermediate_velocities(self):
        self._intermediate_velocities = sf.SWEIntermediateVelocities(
            self._solver._fine_solution
        )
        self._intermediate_velocities.volume_space = self._solver_space
        self._intermediate_velocities.update()

    @property
    def grid(self) -> np.ndarray:
        return CoarsenedFiniteVolumeSpace(
            self._solver_space, self.attributes.coarsening_degree
        ).cell_centers

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = {
            "desc": "Coarse-Exact",
            "leave": False,
            "postfix": {
                "coarsening_degree": self.attributes.coarsening_degree,
                "cfl_number": self.attributes.cfl_number,
                "DOFs": self.dimension,
            },
        }

        return tqdm_kwargs

    @property
    def info(self) -> str:
        return f"Coarse Exact (cfl={self.attributes.cfl_number}, coarsening_degree={self.attributes.coarsening_degree})"


class SWECoarseNetworkSolverFactory(SWEGodunovSolverFactory):
    _plot_label = ["coarse-network", "coarse-network"]
    _intermediate_velocities: sf.SWEIntermediateVelocities

    def _build_space(self):
        self._solver_space = CoarsenedFiniteVolumeSpace(
            FiniteVolumeSpace(self.mesh), self.attributes.coarsening_degree
        )
        self._solver.solver_space = CoarsenedFiniteVolumeSpace(
            FiniteVolumeSpace(self.mesh), self.attributes.coarsening_degree
        )

    def _interpolate(self):
        interpolator = CellAverageInterpolator(
            self.mesh.coarsen(self.attributes.coarsening_degree), 2
        )
        height = interpolator.interpolate(lambda x: self.benchmark.initial_data(x)[0])
        discharge = interpolator.interpolate(
            lambda x: self.benchmark.initial_data(x)[1]
        )
        self._bottom_topography = interpolator.interpolate(self.benchmark.topography)

        self._solver.solution = DiscreteSolutionObservable(
            self.benchmark.start_time, np.array([height, discharge]).T
        )

    def _build_time_stepping(self):

        time_stepping = factory.TIME_STEPPING_FACTORY.get_godunov_time_stepping(
            adaptive=False
        )
        time_stepping.start_time = self.benchmark.start_time
        time_stepping.end_time = self.benchmark.end_time
        time_stepping.mesh = self.mesh
        time_stepping.cfl_number = self.attributes.cfl_number
        time_stepping.intermediate_velocities = self._build_fine_velocities()
        time_stepping.update_time_step()

        self._solver.time_stepping = time_stepping

    def _build_fine_velocities(self) -> sf.SWEIntermediateVelocities:
        interpolator = CellAverageInterpolator(self.mesh, 2)
        height = interpolator.interpolate(lambda x: self.benchmark.initial_data(x)[0])
        discharge = interpolator.interpolate(
            lambda x: self.benchmark.initial_data(x)[1]
        )
        intermediate_velocities = sf.SWEIntermediateVelocities(
            DiscreteSolutionObservable(
                self.benchmark.start_time, np.array([height, discharge]).T
            )
        )
        intermediate_velocities.volume_space = FiniteVolumeSpace(self.mesh)
        intermediate_velocities.update()

        return intermediate_velocities

    def _build_numerical_flux(self):
        self._build_intermediate_velocities()

        numerical_flux = sf.CorrectedFlux()
        numerical_flux.coarse_flux = self._build_coarse_flux()
        numerical_flux.subgrid_flux = self._build_subgrid_flux()

        self._solver.numerical_flux = CFLCheckedFlux(
            numerical_flux, self._solver.time_stepping
        )

    def _build_coarse_flux(self) -> sf.SystemFlux:
        numerical_flux = sf.SWEGodunovNumericalFlux()
        numerical_flux.volume_space = self._solver_space
        numerical_flux.cell_flux_calculator = self._build_cell_flux_calculator()

        return numerical_flux

    def _build_subgrid_flux(self) -> sf.SystemFlux:
        subgrid_flux = sf.NetworkApproximatedFlatBottomSubgridFlux(
            self.attributes.network_path
        )
        subgrid_flux.volume_space = self._solver_space

        return subgrid_flux

    @property
    def tqdm_kwargs(self) -> Dict:
        tqdm_kwargs = {
            "desc": "Coarse-Network",
            "leave": False,
            "postfix": {
                "coarsening_degree": self.attributes.coarsening_degree,
                "cfl_number": self.attributes.cfl_number,
                "DOFs": self.dimension,
            },
        }

        return tqdm_kwargs

    @property
    def info(self) -> str:
        return f"Coarse Network (cfl={self.attributes.cfl_number}, coarsening_degree={self.attributes.coarsening_degree})"
