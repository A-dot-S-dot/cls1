from typing import Callable, List, Tuple

import defaults
import numpy as np
import pandas as pd
from core.discretization import DiscreteSolution
from tqdm.auto import tqdm, trange

from .calculate import Calculate
from .command import Command

# class ExactSubgridFlux(NumericalFlux):
#     """Calculates exat subgrid flux for shallow water equations with flat
#     bottom.

#     """

#     _fine_numerical_fluxes: NumericalFluxContainer
#     _coarse_solution: DiscreteSolution
#     _coarse_numerical_flux: ObservedNumericalFlux
#     _coarsening_degree: int
#     _time_index: int

#     def __init__(
#         self,
#         fine_numerical_fluxes: NumericalFluxContainer,
#         coarse_solution: DiscreteSolution,
#         coarse_numerical_flux: ObservedNumericalFlux,
#         coarsening_degree: int,
#     ):
#         self._fine_numerical_fluxes = fine_numerical_fluxes
#         self._coarse_solution = coarse_solution
#         self._coarse_numerical_flux = coarse_numerical_flux
#         self._coarsening_degree = coarsening_degree
#         self._time_index = 0

#     def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         left_subgrid_flux = self._get_left_subgrid_flux()
#         self._time_index += 1

#         return left_subgrid_flux, np.roll(left_subgrid_flux, -1, axis=0)

#     def _get_left_subgrid_flux(
#         self,
#     ) -> np.ndarray:
#         left_coarse_flux = self._coarse_numerical_flux.left_numerical_flux
#         left_fine_flux = self._fine_numerical_fluxes.left_numerical_fluxes[
#             self._time_index
#         ]

#         return left_fine_flux[:: self._coarsening_degree] + -left_coarse_flux


class DataBuilder:
    def __call__(
        self, solver: solver.ReducedExactSolver
    ) -> Tuple[DiscreteSolution, List[np.ndarray]]:
        solver.subgrid_flux = vector.ObservedNumericalFlux(solver.subgrid_flux)
        solver.right_hand_side = vector.NumericalFluxContainer(
            vector.NumericalFluxDependentRightHandSide(
                solver.space,
                vector.CorrectedNumericalFlux(
                    solver.numerical_flux, solver.subgrid_flux
                ),
            ),
            solver.subgrid_flux,
        )
        Calculate(solver).execute()

        return solver._solution, solver.right_hand_side.left_numerical_fluxes

    def __repr__(self) -> str:
        return self.__class__.__name__


class SubgridDataSaver:
    _data: pd.DataFrame
    _local_degree: int
    _vertex_index: int
    _data_path: str
    _overwrite: bool

    def __init__(
        self, local_degree: int, vertex_index: int, data_path: str, overwrite: bool
    ):
        self._local_degree = local_degree
        self._vertex_index = vertex_index
        self._data_path = data_path
        self._overwrite = overwrite

        self._data = pd.DataFrame(columns=self._create_columns())

    def _create_columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            [
                [
                    *[
                        f"U{i}"
                        for i in range(
                            self._vertex_index - self._local_degree,
                            self._vertex_index + self._local_degree,
                        )
                    ],
                    f"G_{self._vertex_index-1/2}",
                ],
                ["h", "q"],
            ],
        )

    def append(
        self,
        coarse_solution: DiscreteSolution,
        left_subgrid_flux: List[np.ndarray],
    ):
        time_indices = list(range(len(coarse_solution.time) - 1))
        new_data = pd.DataFrame(columns=self._create_columns(), index=time_indices)
        for time_index in tqdm(time_indices, desc="Append Values"):
            self._append_values(
                new_data,
                time_index,
                coarse_solution.values[time_index],
                left_subgrid_flux[time_index],
            )

        self._data = pd.concat([self._data, new_data])

    def _append_values(
        self,
        new_data: pd.DataFrame,
        time_index: int,
        coarse_solution: np.ndarray,
        left_subgrid_flux: np.ndarray,
    ):
        new_data.loc[
            time_index, (f"G_{self._vertex_index-1/2}", "h")
        ] = left_subgrid_flux[self._vertex_index, 0]

        new_data.loc[
            time_index, (f"G_{self._vertex_index-1/2}", "q")
        ] = left_subgrid_flux[self._vertex_index, 1]

        for i in range(
            self._vertex_index - self._local_degree,
            self._vertex_index + self._local_degree,
        ):
            new_data.loc[time_index, (f"U{i}", "h")] = coarse_solution[i, 0]
            new_data.loc[time_index, (f"U{i}", "q")] = coarse_solution[i, 1]

    def save(self):
        self._data.to_csv(
            self._data_path,
            mode="w" if self._overwrite else "a",
            header=self._overwrite,
        )
        tqdm.write(f"Data is saved in {self._data_path}.")

    def __repr__(self) -> str:
        return self.__class__.__name__


class BenchmarkParametersSaver:
    _data: pd.DataFrame
    _data_path: str
    _overwrite: bool

    def __init__(self, data_path: str, overwrite: bool):
        self._data = pd.DataFrame(columns=self._create_columns())
        self._data_path = data_path
        self._overwrite = overwrite

    def _create_columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            [
                ["height", "velocity"],
                ["average", "amplitude", "wave_number", "phase_shift"],
            ],
        )

    def append(self, benchmark: benchmark.ShallowWaterOscillationNoTopographyBenchmark):
        self._data.loc[len(self._data.index)] = [
            benchmark.height_average,
            benchmark.height_amplitude,
            benchmark.height_wave_number,
            benchmark.height_phase_shift,
            benchmark.velocity_average,
            benchmark.velocity_amplitude,
            benchmark.velocity_wave_number,
            benchmark.velocity_phase_shift,
        ]

    def save(self):
        self._data.to_csv(
            self._data_path,
            mode="w" if self._overwrite else "a",
            header=self._overwrite,
        )
        tqdm.write(f"Benchmark parameters are saved in {self._data_path}.")

    def __repr__(self) -> str:
        return self.__class__.__name__


class GenerateData(Command):
    _data_builder: DataBuilder
    _subgrid_saver: SubgridDataSaver
    _benchmark_parameters_saver: BenchmarkParametersSaver

    _solver: List[solver.ReducedExactSolver]
    _benchmarks: List[benchmark.ShallowWaterOscillationNoTopographyBenchmark]
    _fine_numerical_fluxes: List

    def __init__(
        self,
        solution_number: int,
        benchmark_generator=None,
        local_degree=None,
        vertex_index=None,
        data_path=None,
        benchmark_parameters_path=None,
        overwrite=None,
    ):
        benchmark_generator = benchmark_generator or (
            lambda _: benchmark.ShallowWaterRandomOscillationNoTopographyBenchmark(
                height_average=benchmark.shallow_water.HEIGHT_AVERAGE
            )
        )
        local_degree = local_degree or defaults.LOCAL_DEGREE
        vertex_index = vertex_index or local_degree
        data_path = data_path or defaults.DATA_PATH
        benchmark_parameters_path = (
            benchmark_parameters_path or defaults.BENCHMARK_PARAMETERS_PATH
        )
        overwrite = overwrite if overwrite is not None else defaults.OVERWRITE

        self._data_builder = DataBuilder()
        self._subgrid_saver = SubgridDataSaver(
            local_degree, vertex_index, data_path, overwrite
        )
        self._benchmark_parameters_saver = BenchmarkParametersSaver(
            benchmark_parameters_path, overwrite
        )

        self._fine_numerical_fluxes = []

        self._build_data(benchmark_generator, solution_number)

    def _build_data(self, benchmark_generator: Callable, solution_number: int):
        if solution_number == 1:
            title = f"Generate Data"
            title += "\n" + len(title) * "-"
            tqdm.write(title)

            self._add_solution(benchmark_generator)

            tqdm.write("")
        else:
            for i in trange(solution_number, desc="Generate Data", unit="solution"):
                title = f"Solver {i}"
                title += "\n" + len(title) * "-"
                tqdm.write(title)

                self._add_solution(benchmark_generator())
                tqdm.write("")

    def _add_solution(self, shallow_water_benchmark):
        resolved_exact_solver = solver.ReducedExactSolver(
            "swe",
            shallow_water_benchmark,
        )
        self._subgrid_saver.append(*self._data_builder(resolved_exact_solver))
        self._benchmark_parameters_saver.append(shallow_water_benchmark)

    def execute(self):
        self._subgrid_saver.save()
        self._benchmark_parameters_saver.save()
