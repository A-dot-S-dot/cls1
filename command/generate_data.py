import os
import time
from argparse import Namespace
from typing import List, Tuple

import numpy as np
import pandas as pd
from benchmark import SWEOscillationNoTopographyBenchmark
from defaults import *
from factory.pde_solver_factory import SWEGodunovSolverFactory
from pde_solver import FiniteVolumeSolver
from pde_solver.discrete_solution import CoarseSolution
from pde_solver.discrete_solution.discrete_solution import DiscreteSolution
from pde_solver.mesh.uniform import UniformMesh
from pde_solver.system_flux import SystemFlux
from pde_solver.time_stepping import TimeStepTooSmallError
from tqdm import tqdm, trange

from .command import Command


class DataBuilder:
    benchmark = SWEOscillationNoTopographyBenchmark()

    _solver_factory: SWEGodunovSolverFactory
    _fine_solver: FiniteVolumeSolver
    _coarse_numerical_flux: SystemFlux
    _fine_left_numerical_flux: List[np.ndarray]

    _coarsening_degree: int
    _mesh_size: int
    _cfl_number: float

    def __init__(self, coarsening_degree: int, mesh_size: int, cfl_number: float):
        self._coarsening_degree = coarsening_degree
        self._mesh_size = mesh_size
        self._cfl_number = cfl_number

        self._build_solver_factory()
        self._build_coarse_numerical_flux()

    def _build_solver_factory(self):
        self._solver_factory = SWEGodunovSolverFactory()
        self._solver_factory.attributes = Namespace(
            cfl_number=self._cfl_number, adaptive=False
        )
        self._solver_factory.mesh = UniformMesh(self.benchmark.domain, self._mesh_size)
        self._solver_factory.benchmark = self.benchmark

    def _build_coarse_numerical_flux(
        self,
    ):
        coarse_solver_factory = SWEGodunovSolverFactory()
        coarse_solver_factory.attributes = Namespace(
            cfl_number=self._cfl_number, adaptive=False
        )
        coarse_solver_factory.benchmark = self.benchmark
        coarse_solver_factory.mesh = self._solver_factory.mesh.coarsen(
            self._coarsening_degree
        )
        solver = coarse_solver_factory.solver
        self._coarse_numerical_flux = solver.numerical_flux

    def __call__(self) -> Tuple[CoarseSolution, np.ndarray]:
        self.benchmark.random_parameters()
        self._calculate_solution()

        coarse_solution = CoarseSolution(
            self._fine_solver.solution, self._coarsening_degree
        )
        left_subgrid_flux = self._get_subgrid_fluxes(coarse_solution)

        return coarse_solution, left_subgrid_flux

    def _calculate_solution(self):
        self._fine_solver = self._solver_factory.solver
        self._fine_left_numerical_flux = []

        start_time = time.time()

        for _ in tqdm(self._fine_solver.time_stepping, **self._fine_solver.tqdm_kwargs):

            try:
                self._fine_solver.update()
                self._fine_left_numerical_flux.append(self._fine_solver.left_flux)
            except TimeStepTooSmallError:
                tqdm.write("WARNING: time step is too small calculation is interrupted")
                break

        tqdm.write(
            f"Solved {self._solver_factory.info} with {self._solver_factory.dimension} DOFs and {self._fine_solver.time_stepping.time_steps} time steps in {time.time()-start_time:.2f}s."
        )

    def _get_subgrid_fluxes(self, coarse_solution: CoarseSolution) -> np.ndarray:
        return np.array(
            [
                self._get_left_subgrid_flux(time_index, coarse_solution)
                for time_index in range(len(coarse_solution.time) - 1)
            ]
        )

    def _get_left_subgrid_flux(
        self,
        time_index: int,
        coarse_solution: CoarseSolution,
    ) -> np.ndarray:
        left_coarse_flux = self._coarse_numerical_flux(
            coarse_solution.values[time_index]
        )[0]
        left_fine_flux = self._fine_left_numerical_flux[time_index]

        return left_fine_flux[:: self._coarsening_degree] + -left_coarse_flux


class SubgridDataSaver:
    data: pd.DataFrame

    _skip_time_steps: int
    _local_degree: int

    def __init__(self, skip_time_steps: int, local_degree: int):
        self._skip_time_steps = skip_time_steps
        self._local_degree = local_degree

        self.data = pd.DataFrame(columns=self._create_columns())

    def _create_columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            [
                [
                    *[f"U{i}" for i in range(2 * self._local_degree)],
                    f"G_{self._local_degree-1/2}",
                ],
                ["h", "q"],
            ],
        )

    def append(
        self,
        coarse_solution: DiscreteSolution,
        left_subgrid_flux: np.ndarray,
    ):
        time_indices = list(
            range(0, len(coarse_solution.time) - 1, self._skip_time_steps)
        )
        new_data = pd.DataFrame(columns=self._create_columns(), index=time_indices)
        for time_index in time_indices:
            self._append_values(
                new_data,
                time_index,
                coarse_solution.values[time_index],
                left_subgrid_flux[time_index],
            )

        self.data = pd.concat([self.data, new_data])

    def _append_values(
        self,
        new_data: pd.DataFrame,
        time_index: int,
        coarse_solution: np.ndarray,
        left_subgrid_flux: np.ndarray,
    ):
        new_data.loc[
            time_index, (f"G_{self._local_degree-1/2}", "h")
        ] = left_subgrid_flux[self._local_degree, 0]

        new_data.loc[
            time_index, (f"G_{self._local_degree-1/2}", "q")
        ] = left_subgrid_flux[self._local_degree, 1]

        for i in range(2 * self._local_degree):
            new_data.loc[time_index, (f"U{i}", "h")] = coarse_solution[i, 0]
            new_data.loc[time_index, (f"U{i}", "q")] = coarse_solution[i, 1]

    def save(self, train_path: str, validate_path: str, append=False):
        train_data = self.data.sample(frac=0.8)
        test_data = pd.concat([self.data, train_data]).drop_duplicates(keep=False)

        for data, file_path in [
            (train_data, train_path),
            (test_data, validate_path),
        ]:
            if not append:
                os.remove(file_path)

            data.to_csv(file_path, mode="a", header=not os.path.exists(file_path))


class BenchmarkParametersSaver:
    data: pd.DataFrame

    def __init__(self):
        self.data = pd.DataFrame(columns=self._create_columns())

    def _create_columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_product(
            [
                ["height", "velocity"],
                ["average", "amplitude", "wave_number", "phase_shift"],
            ],
        )

    def append(self, benchmark: SWEOscillationNoTopographyBenchmark):
        self.data.loc[len(self.data.index)] = [
            benchmark.height_average,
            benchmark.height_amplitude,
            benchmark.height_wave_number,
            benchmark.height_phase_shift,
            benchmark.velocity_average,
            benchmark.velocity_amplitude,
            benchmark.velocity_wave_number,
            benchmark.velocity_phase_shift,
        ]

    def save(self, file_path: str, append=False):
        if not append:
            os.remove(file_path)

        self.data.to_csv(file_path, mode="a", header=not os.path.exists(file_path))


class GenerateData(Command):
    def execute(self):
        builder = DataBuilder(
            self._args.coarsening_degree, self._args.mesh_size, self._args.cfl_number
        )
        subgrid_saver = SubgridDataSaver(self._args.skip_steps, self._args.local_degree)
        benchmark_saver = BenchmarkParametersSaver()

        for _ in trange(
            self._args.solution_number,
            desc="Create Data",
            unit="solutions",
            leave=False,
        ):
            subgrid_saver.append(*builder())
            benchmark_saver.append(builder.benchmark)

        subgrid_saver.save(
            self._args.train_path, self._args.validate_path, self._args.overwrite
        )
        benchmark_saver.save(self._args.benchmark_parameters_path, self._args.overwrite)
