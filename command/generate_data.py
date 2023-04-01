import argparse
import os
from typing import Any, Tuple

import core
from core.discrete_solution import DiscreteSolutionWithNodeNeighboursHistory
import defaults
import finite_volume
import numpy as np
import pandas as pd
from benchmark import shallow_water
from tqdm.auto import tqdm, trange

from .calculate import Calculate, CalculateParser
from .command import Command

DIRECTORIES = {
    "llf": "data/reduced-llf/",
    "llf2": "data/reduced-llf-2/",
    "es1": "data/reduced-es1/",
}


class SubgridFluxDataBuilder:
    coarsening_degree: int
    input_radius: int
    node_index: int
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
        print_output=True,
    ):
        self._fine_flux = finite_volume.NumericalFluxWithArbitraryInput(fine_flux)
        self._coarse_flux = finite_volume.NumericalFluxWithArbitraryInput(coarse_flux)
        self.coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        self._coarsener = core.VectorCoarsener(self.coarsening_degree)
        self.input_radius = input_radius or defaults.INPUT_RADIUS
        self.node_index = node_index or self.input_radius
        self.print_output = print_output

        assert self.node_index >= self.input_radius

    def __call__(
        self, solver: core.Solver
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """Returns coarse solution and subgrid fluxes at node."""
        fine_values, coarse_values = self._get_values(solver)

        _, fine_flux = self._fine_flux(*fine_values)
        _, coarse_flux = self._coarse_flux(*coarse_values)

        return coarse_values, fine_flux - coarse_flux

    def _get_values(
        self, solver: core.Solver
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        solver.solution = DiscreteSolutionWithNodeNeighboursHistory(
            solver.solution,
            self.coarsening_degree * self.input_radius,
            node_index=self.node_index * self.coarsening_degree,
        )
        Calculate(solver, disable=not self.print_output, leave=False).execute()

        fine_values = np.moveaxis(solver.solution.node_neighbours_history, 1, 0)
        coarse_values = self._coarsener(fine_values)

        return tuple(fine_values), tuple(coarse_values)

    def __repr__(self) -> str:
        return self.__class__.__name__


class SubgridFluxDataFrameBuilder:
    def __init__(self, data_builder: SubgridFluxDataBuilder):
        self._data_builder = data_builder

    def get_data_frame(
        self,
        solver: core.Solver[DiscreteSolutionWithNodeNeighboursHistory],
        data=None,
    ) -> pd.DataFrame:
        coarse_values, subgrid_flux = self._data_builder(solver)
        time = solver.solution.time_history

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
    def get_data_frame(
        self, benchmark: shallow_water.RandomOscillationNoTopographyBenchmark, data=None
    ) -> pd.DataFrame:
        if data is None:
            data = pd.DataFrame(columns=self._create_columns())

        self._append_values(data, benchmark)

        return data

    def _create_columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(
            [
                ("seed", ""),
                *pd.MultiIndex.from_product(
                    [
                        ["height", "velocity"],
                        ["average", "amplitude", "wave_number", "phase_shift"],
                    ]
                ),
            ]
        )

    def _append_values(
        self,
        data: pd.DataFrame,
        benchmark: shallow_water.RandomOscillationNoTopographyBenchmark,
    ):
        data.loc[len(data.index)] = [
            benchmark.seed,
            benchmark.height_average,
            benchmark.height_amplitude,
            benchmark.height_wave_number,
            benchmark.height_phase_shift,
            benchmark.velocity_average,
            benchmark.velocity_amplitude,
            benchmark.velocity_wave_number,
            benchmark.velocity_phase_shift,
        ]


class GenerateData(Command):
    _solution_number: int
    _get_benchmark: shallow_water.RandomBenchmarkGenerator
    _subgrid_flux_df_builder: SubgridFluxDataFrameBuilder
    _benchmark_df_builder: BenchparkParameterDataFrameBuilder
    _subgrid_flux_path: str
    _benchmark_path: str
    _overwrite: bool
    _print_output: bool

    def __init__(
        self,
        solver: finite_volume.Solver,
        directory: str,
        solution_number=None,
        end_time=None,
        seed=None,
        coarsening_degree=None,
        input_radius=None,
        node_index=None,
        overwrite=True,
        print_output=True,
    ):
        self._solver = solver
        self._solution_number = solution_number or defaults.SOLUTION_NUMBER
        self._get_benchmark = shallow_water.RandomBenchmarkGenerator(
            seed or defaults.SEED, end_time
        )
        self._benchmark_df_builder = BenchparkParameterDataFrameBuilder()
        self._subgrid_flux_df_builder = self._get_subgrid_flux_data_frame_builder(
            coarsening_degree or defaults.COARSENING_DEGREE,
            input_radius=input_radius,
            node_index=node_index,
            print_output=print_output,
        )
        self._subgrid_flux_path = os.path.join(directory, "data.csv")
        self._benchmark_path = os.path.join(directory, "benchmark.csv")
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
            tqdm.write(f"Subgrid Flux Data is saved in {self._subgrid_flux_path}.")
            tqdm.write(f"Benchmark Data is saved in {self._benchmark_path}.")

    def _build_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        benchmark = self._get_benchmark()
        self._solver.reinitialize(benchmark)

        benchmark_data = self._benchmark_df_builder.get_data_frame(benchmark)
        subgrid_flux_data = self._subgrid_flux_df_builder.get_data_frame(self._solver)

        return subgrid_flux_data, benchmark_data

    def _save_data(
        self,
        subgrid_flux_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        overwrite: bool,
    ):
        core.save_data(subgrid_flux_data, self._subgrid_flux_path, overwrite)
        core.save_data(benchmark_data, self._benchmark_path, overwrite)


class GenerateDataParser(CalculateParser):
    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "generate-data",
            help="Generate data for reduced shallow water models.",
            description="""Generates data for shallow water models which uses
            neural networks subgrid fluxes.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_arguments(self, parser):
        self._add_directory(parser)
        self._add_end_time(parser)
        self._add_coarsening_degree(parser)
        self._add_shallow_water_solver(parser)
        self._add_solution_number(parser)
        self._add_seed(parser)
        self._add_input_radius(parser)
        self._add_node_index(parser)
        self._add_append(parser)
        self._add_general_arguments(parser)

    def _add_directory(self, parser):
        parser.add_argument(
            "-d",
            "--directory",
            help="Specify directory for storing data.",
        )

    def _add_coarsening_degree(self, parser):
        parser.add_argument(
            "-c",
            "--coarsening-degree",
            help="Specify the coarsening degree.",
            type=core.positive_int,
            metavar="<degree>",
            default=defaults.COARSENING_DEGREE,
        )

    def _add_solution_number(self, parser):
        parser.add_argument(
            "-n",
            "--solution-number",
            help="Number of generated solutions.",
            type=core.positive_int,
            default=defaults.SOLUTION_NUMBER,
            metavar="<number>",
        )

    def _add_seed(self, parser):
        parser.add_argument(
            "--seed",
            help="Seed for generating random benchmarks",
            type=core.positive_int,
            default=defaults.SEED,
            metavar="<seed>",
        )

    def _add_input_radius(self, parser):
        parser.add_argument(
            "-i",
            "--input-radius",
            help="Number of considered cells on each node side.",
            type=core.positive_int,
            default=defaults.INPUT_RADIUS,
            metavar="<radius>",
        )

    def _add_node_index(self, parser):
        parser.add_argument(
            "--node-index",
            help="Determines for which node the subgrid fluxes are calculated. The most left node has the index 0.",
            type=core.positive_int,
            default=defaults.INPUT_RADIUS,
            metavar="<index>",
        )

    def _add_append(self, parser):
        parser.add_argument(
            "--append",
            help="Appends generated data to the specified paths",
            action="store_true",
        )

    def postprocess(self, arguments):
        self._assert_one_solver(arguments)
        self._build_directory(arguments)
        arguments.overwrite = not arguments.append
        arguments.benchmark = shallow_water.OscillationNoTopographyBenchmark()
        self._build_solver(arguments)
        arguments.solver = arguments.solver[0]

        arguments.command = GenerateData

        del arguments.append
        del arguments.benchmark

    def _assert_one_solver(self, arguments):
        solver_num = 0 if arguments.solver is None else len(arguments.solver)
        assert (
            solver_num == 1
        ), f"Exactly one solver must be given. There are {solver_num}."

    def _build_directory(self, arguments):
        if arguments.directory is None:
            arguments.directory = DIRECTORIES[arguments.solver[0].short]
