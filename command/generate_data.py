from typing import Callable, Dict, Tuple

import core
import defaults
import lib
import numpy as np
import pandas as pd
import shallow_water
from shallow_water.solver import lax_friedrichs
from tqdm.auto import tqdm, trange

from .calculate import Calculate
from .command import Command


class SubgridFluxDataBuilder:
    _subgrid_flux: lib.NumericalFlux
    _coarsener: core.VectorCoarsener
    _skip: int
    _update_coarse_solution: Callable[[np.ndarray, np.ndarray], np.ndarray]
    _print_output: bool

    def __init__(
        self,
        subgrid_flux: lib.NumericalFlux,
        coarsening_degree: int,
        skip: int,
        print_output=True,
    ):
        self._subgrid_flux = subgrid_flux
        self._coarsener = core.VectorCoarsener(coarsening_degree)
        self._skip = skip
        self._update_coarse_solution = self._initialize_coarse_solution
        self._print_output = print_output

    def _initialize_coarse_solution(
        self, dof_vector: np.ndarray, coarse_solution: np.ndarray
    ) -> np.ndarray:
        self._update_coarse_solution = self._add_coarse_solution

        return np.array([self._coarsener(dof_vector)])

    def _add_coarse_solution(
        self, dof_vector: np.ndarray, coarse_solution: np.ndarray
    ) -> np.ndarray:
        return np.append(
            coarse_solution, np.array([self._coarsener(dof_vector)]), axis=0
        )

    def __call__(
        self,
        solver: core.Solver[core.DiscreteSolutionWithHistory],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns coarse solution and subgrid fluxes at node."""
        Calculate(solver, disable=not self._print_output, leave=False).execute()
        flux_history = lib.NumericalFluxWithHistory(self._subgrid_flux)
        coarse_solution = np.empty(0)

        for time, dof_vector in tqdm(
            zip(
                solver.solution.time_history[:: self._skip],
                solver.solution.value_history[:: self._skip],
            ),
            desc="Create Subgrid Flux Data",
            disable=not self._print_output,
            leave=False,
        ):
            flux_history(time, dof_vector)
            coarse_solution = self._update_coarse_solution(dof_vector, coarse_solution)

        self._update_coarse_solution = self._initialize_coarse_solution

        return coarse_solution, flux_history.flux_left_history

    def __repr__(self) -> str:
        return self.__class__.__name__


class SubgridFluxDataFrameCreator:
    _local_degree: int
    _node_index: int
    _print_output: bool

    def __init__(self, local_degree: int, node_index: int, print_output=True):
        self._local_degree = local_degree
        self._node_index = node_index
        self._print_output = print_output

        self._data = pd.DataFrame(columns=self._create_columns())

    def __call__(
        self,
        coarse_solution: np.ndarray,
        subgrid_flux_left: np.ndarray,
        data=None,
    ) -> pd.DataFrame:
        time_indices = range(len(coarse_solution))
        new_data = pd.DataFrame(columns=self._create_columns(), index=time_indices)
        for time_index in tqdm(
            time_indices,
            desc="Create Data Frame" if data is None else "Updata Data Frame",
            disable=not self._print_output,
            leave=False,
        ):
            self._append_values(
                new_data,
                time_index,
                coarse_solution[time_index],
                subgrid_flux_left[time_index],
            )

        return pd.concat([data, new_data]) if data is not None else new_data

    def _create_columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            [
                [
                    *[
                        f"U{i}"
                        for i in range(
                            self._node_index - self._local_degree,
                            self._node_index + self._local_degree,
                        )
                    ],
                    f"G_{self._node_index-1/2}",
                ],
                ["h", "q"],
            ],
        )

    def _append_values(
        self,
        new_data: pd.DataFrame,
        time_index: int,
        coarse_solution: np.ndarray,
        left_subgrid_flux: np.ndarray,
    ):
        new_data.loc[
            time_index, (f"G_{self._node_index-1/2}", "h")
        ] = left_subgrid_flux[self._node_index, 0]

        new_data.loc[
            time_index, (f"G_{self._node_index-1/2}", "q")
        ] = left_subgrid_flux[self._node_index, 1]

        for i in range(
            self._node_index - self._local_degree,
            self._node_index + self._local_degree,
        ):
            new_data.loc[time_index, (f"U{i}", "h")] = coarse_solution[i, 0]
            new_data.loc[time_index, (f"U{i}", "q")] = coarse_solution[i, 1]

    def __repr__(self) -> str:
        return self.__class__.__name__


class BenchmarkParametersDataCreator:
    def __call__(
        self,
        benchmark: shallow_water.OscillationNoTopographyBenchmark,
        data=None,
    ) -> pd.DataFrame:
        if data is None:
            data = pd.DataFrame(columns=self._create_columns())

        self._append_values(data, benchmark)

        return data

    def _create_columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            [
                ["height", "velocity"],
                ["average", "amplitude", "wave_number", "phase_shift"],
            ],
        )

    def _append_values(
        self,
        data: pd.DataFrame,
        benchmark: shallow_water.OscillationNoTopographyBenchmark,
    ):
        data.loc[len(data.index)] = [
            benchmark.height_average,
            benchmark.height_amplitude,
            benchmark.height_wave_number,
            benchmark.height_phase_shift,
            benchmark.velocity_average,
            benchmark.velocity_amplitude,
            benchmark.velocity_wave_number,
            benchmark.velocity_phase_shift,
        ]

    def __repr__(self) -> str:
        return self.__class__.__name__


class GenerateData(Command):
    _solution_number: int
    _solver_builder: Callable[..., core.Solver]
    _create_benchmark: Callable[[], shallow_water.OscillationNoTopographyBenchmark]
    _subgrid_flux_data_path: str
    _benchmark_data_path: str
    _overwrite: bool
    _print_output: bool
    _solver_kwargs: Dict
    _subgrid_flux_data_builder: SubgridFluxDataBuilder
    _subgrid_flux_data_frame_creator: SubgridFluxDataFrameCreator
    _benchmark_parameters_data_frame_creator: BenchmarkParametersDataCreator

    def __init__(
        self,
        solution_number: int,
        create_benchmark=None,
        solver_builder=None,
        fine_flux_builder=None,
        coarse_flux_builder=None,
        coarsening_degree=None,
        skip=None,
        input_radius=None,
        node_index=None,
        subgrid_flux_data_path=None,
        benchmark_data_path=None,
        overwrite=None,
        print_output=True,
        **solver_kwargs,
    ):
        self._solution_number = solution_number
        self._solver_builder = solver_builder or lax_friedrichs.LaxFriedrichsSolver
        self._create_benchmark = create_benchmark or (
            lambda: shallow_water.RandomOscillationNoTopographyBenchmark(
                height_average=shallow_water.HEIGHT_AVERAGE
            )
        )
        fine_flux_builder = fine_flux_builder or lax_friedrichs.get_lax_friedrichs_flux
        coarse_flux_builder = coarse_flux_builder or fine_flux_builder
        coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        skip = skip or defaults.SKIP
        input_radius = input_radius or defaults.INPUT_DIMENSION
        node_index = node_index or input_radius
        self._subgrid_flux_data_path = (
            subgrid_flux_data_path or defaults.SUBGRID_FLUX_DATA_PATH
        )
        self._benchmark_data_path = benchmark_data_path or defaults.BENCHMARK_DATA_PATH
        self._overwrite = overwrite if overwrite is not None else defaults.OVERWRITE
        self._print_output = print_output
        self._solver_kwargs = solver_kwargs

        self._assert_coarsening_degree(coarsening_degree)

        subgrid_flux = self._create_subgrid_flux(
            fine_flux_builder, coarse_flux_builder, coarsening_degree
        )
        self._subgrid_flux_data_builder = SubgridFluxDataBuilder(
            subgrid_flux, coarsening_degree, skip, print_output=print_output
        )
        self._subgrid_flux_data_frame_creator = SubgridFluxDataFrameCreator(
            input_radius, node_index, print_output=print_output
        )
        self._benchmark_parameters_data_frame_creator = BenchmarkParametersDataCreator()

    def _assert_coarsening_degree(self, coarsening_degree: int):
        solver = self._create_solver()
        mesh_size = len(solver.solution.space.mesh)
        if mesh_size % coarsening_degree != 0:
            raise ValueError(
                f"Mesh size {mesh_size} is not dividable by coarsening degree {coarsening_degree}."
            )

    def _create_solver(
        self, benchmark=None
    ) -> core.Solver[core.DiscreteSolutionWithHistory]:
        benchmark = benchmark or self._create_benchmark()
        return self._solver_builder(benchmark, save_history=True, **self._solver_kwargs)

    def _create_subgrid_flux(
        self,
        fine_flux_builder: Callable[
            [shallow_water.ShallowWaterBenchmark, core.Mesh], lib.NumericalFlux
        ],
        coarse_flux_builder: Callable[
            [shallow_water.ShallowWaterBenchmark, core.Mesh], lib.NumericalFlux
        ],
        coarsening_degree: int,
    ) -> lib.NumericalFlux:
        benchmark = self._create_benchmark()
        solver = self._create_solver(benchmark)
        fine_flux = fine_flux_builder(benchmark, solver.solution.space.mesh)

        fine_mesh = solver.solution.space.mesh
        coarse_mesh = core.UniformMesh(
            fine_mesh.domain, len(fine_mesh) // coarsening_degree
        )

        coarse_flux = coarse_flux_builder(benchmark, coarse_mesh)

        return lib.SubgridFlux(fine_flux, coarse_flux, coarsening_degree)

    def execute(self):
        if self._solution_number == 1:
            subgrid_flux_data, benchmark_data = self._build_data()
            self._save_data(subgrid_flux_data, benchmark_data, self._overwrite)
        else:
            for i in trange(
                self._solution_number,
                desc="Generate Data",
                unit="solution",
                disable=not self._print_output,
            ):
                subgrid_flux_data, benchmark_data = self._build_data()
                self._save_data(
                    subgrid_flux_data,
                    benchmark_data,
                    self._overwrite if i == 0 else False,
                )

        if self._print_output:
            tqdm.write(f"Subgrid Flux Data is saved in {self._subgrid_flux_data_path}.")
            tqdm.write(f"Becnhmark Data is saved in {self._benchmark_data_path}.")

    def _build_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        benchmark = self._create_benchmark()
        solver = self._create_solver(benchmark)
        subgrid_flux_data = self._subgrid_flux_data_frame_creator(
            *self._subgrid_flux_data_builder(solver)
        )
        benchmark_data = self._benchmark_parameters_data_frame_creator(benchmark)

        return subgrid_flux_data, benchmark_data

    def _save_data(
        self,
        subgrid_flux_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        overwrite: bool,
    ):
        self._save(subgrid_flux_data, self._subgrid_flux_data_path, overwrite)
        self._save(benchmark_data, self._benchmark_data_path, overwrite)

    def _save(self, data: pd.DataFrame, path: str, overwrite: bool):
        data.to_csv(
            path,
            mode="w" if overwrite else "a",
            header=overwrite,
        )


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path, header=[0, 1], skipinitialspace=True, index_col=0)
