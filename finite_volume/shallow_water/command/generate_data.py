import random
from typing import Callable, Tuple

import benchmark.shallow_water as swe
import core
import defaults
import finite_volume
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange

import command


class SubgridFluxDataBuilder:
    coarsening_degree: int
    input_radius: int
    node_index: int
    skip: int
    print_output: bool

    _fine_flux: finite_volume.NumericalFlux
    _coarse_flux: finite_volume.NumericalFlux
    _coarsener: core.VectorCoarsener

    def __init__(
        self,
        fine_flux: finite_volume.NumericalFlux,
        coarse_flux: finite_volume.NumericalFlux,
        coarsening_degree=None,
        input_radius=None,
        node_index=None,
        skip=None,
        print_output=True,
    ):
        self._fine_flux = finite_volume.NumericalFluxWithArbitraryInput(fine_flux)
        self._coarse_flux = finite_volume.NumericalFluxWithArbitraryInput(coarse_flux)
        self.coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        self._coarsener = core.VectorCoarsener(self.coarsening_degree)
        self.input_radius = input_radius or defaults.INPUT_RADIUS
        self.node_index = node_index or self.input_radius
        self.skip = skip or defaults.SKIP
        self.print_output = print_output

        assert self.node_index >= self.input_radius

    def __call__(
        self, solution: core.DiscreteSolutionWithHistory
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """Returns coarse solution and subgrid fluxes at node."""
        fine_values, coarse_values = self._get_values(solution)

        _, fine_flux = self._fine_flux(*fine_values)
        _, coarse_flux = self._coarse_flux(*coarse_values)

        return coarse_values, fine_flux - coarse_flux

    def _get_values(
        self, solution: core.DiscreteSolutionWithHistory
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        anchor_cell_index = self.node_index * self.coarsening_degree
        input_radius = self.coarsening_degree * self._coarse_flux.input_dimension // 2
        left_cell_index = anchor_cell_index - input_radius
        right_cell_index = anchor_cell_index + input_radius

        fine_values = np.moveaxis(
            solution.value_history[:: self.skip, left_cell_index:right_cell_index],
            1,
            0,
        )

        coarse_values = self._coarsener(fine_values)

        return tuple(fine_values), tuple(coarse_values)

    def __repr__(self) -> str:
        return self.__class__.__name__


class SubgridFluxDataFrameBuilder:
    def __init__(self, data_builder: SubgridFluxDataBuilder):
        self._data_builder = data_builder

    def get_data_frame(
        self,
        solution: core.DiscreteSolutionWithHistory,
        data=None,
    ) -> pd.DataFrame:
        time = solution.time_history[:: self._data_builder.skip]
        coarse_values, subgrid_flux = self._data_builder(solution)

        new_data = pd.DataFrame(
            np.concatenate([*coarse_values, subgrid_flux], axis=1),
            columns=self._create_columns(),
            index=time,
        )

        return pd.concat([data, new_data]) if data is not None else new_data

    def _create_columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            [
                [
                    *[
                        f"U{i}"
                        for i in range(
                            self._data_builder.node_index
                            - self._data_builder.input_radius,
                            self._data_builder.node_index
                            + self._data_builder.input_radius,
                        )
                    ],
                    f"G_{self._data_builder.node_index-1/2}",
                ],
                ["h", "q"],
            ],
        )

    def __repr__(self) -> str:
        return self.__class__.__name__


class BenchparkParameterDataFrameBuilder:
    def get_data_frame(self, benchmark, data=None) -> pd.DataFrame:
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
        benchmark: swe.OscillationNoTopographyBenchmark,
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


class RandomBenchmarkGenerator:
    def __init__(self, seed=None):
        random.seed(seed)

    def __call__(self) -> swe.OscillationNoTopographyBenchmark:
        return swe.RandomOscillationNoTopographyBenchmark(
            height_average=swe.HEIGHT_AVERAGE
        )

    def __repr__(self) -> str:
        return self.__class__.__name__


class GenerateData(command.Command):
    _solution_number: int
    _get_benchmark: RandomBenchmarkGenerator
    _subgrid_flux_df_builder: SubgridFluxDataFrameBuilder
    _benchmark_df_builder: BenchparkParameterDataFrameBuilder
    _subgrid_flux_data_path: str
    _benchmark_data_path: str
    _overwrite: bool
    _print_output: bool

    def __init__(
        self,
        solver: finite_volume.Solver,
        solution_number=None,
        seed=None,
        coarsening_degree=None,
        skip=None,
        input_radius=None,
        node_index=None,
        subgrid_flux_data_path=None,
        benchmark_data_path=None,
        overwrite=True,
        print_output=True,
    ):
        self._solver = solver
        self._solution_number = solution_number or defaults.SOLUTION_NUMBER
        self._get_benchmark = RandomBenchmarkGenerator(seed or defaults.SEED)
        self._benchmark_df_builder = BenchparkParameterDataFrameBuilder()
        self._subgrid_flux_df_builder = self._get_subgrid_flux_data_frame_builder(
            coarsening_degree or defaults.COARSENING_DEGREE,
            skip=skip,
            input_radius=input_radius,
            node_index=node_index,
            print_output=print_output,
        )
        self._subgrid_flux_data_path = (
            subgrid_flux_data_path or defaults.SUBGRID_FLUX_DATA_PATH
        )
        self._benchmark_data_path = benchmark_data_path or defaults.BENCHMARK_DATA_PATH
        self._overwrite = overwrite
        self._print_output = print_output

    def _get_subgrid_flux_data_frame_builder(self, coarsening_degree: int, **kwargs):
        fine_flux, coarse_flux = self._get_fluxes(coarsening_degree)
        data_builder = SubgridFluxDataBuilder(
            fine_flux, coarse_flux, coarsening_degree=coarsening_degree, **kwargs
        )

        return SubgridFluxDataFrameBuilder(data_builder)

    def _get_fluxes(
        self, coarsening_degree: int
    ) -> Tuple[finite_volume.NumericalFlux, finite_volume.NumericalFlux]:
        fine_space = self._solver.solution.space
        coarse_space = fine_space.coarsen(coarsening_degree)

        fine_flux = self._solver.flux_getter(self._get_benchmark(), fine_space)
        coarse_flux = self._solver.flux_getter(self._get_benchmark(), coarse_space)

        return fine_flux, coarse_flux

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
        benchmark = self._get_benchmark()
        benchmark_data = self._benchmark_df_builder.get_data_frame(benchmark)

        self._solver.reinitialize(benchmark)
        command.Calculate(
            self._solver, disable=not self._print_output, leave=False
        ).execute()

        subgrid_flux_data = self._subgrid_flux_df_builder.get_data_frame(
            self._solver.solution
        )

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
