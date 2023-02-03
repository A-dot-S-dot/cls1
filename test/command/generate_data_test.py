from typing import Tuple
from unittest import TestCase

import numpy as np
import pandas as pd
import shallow_water
from shallow_water.solver import lax_friedrichs as llf

from command.generate_data import *


class TestSubgridFlux:
    def __call__(
        self, time: float, dof_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return dof_vector[::2], dof_vector[1::2]


class TestSubgridFluxDataBuilder(TestCase):
    def test_builder(self):
        solver = llf.LocalLaxFriedrichsSolver(
            shallow_water.OscillationNoTopographyBenchmark(end_time=4.0),
            mesh_size=4,
            save_history=True,
        )
        builder = SubgridFluxDataBuilder(
            TestSubgridFlux(), coarsening_degree=2, skip=2, print_output=False
        )
        coarse_solution, subgrid_node_flux = builder(solver)

        expected_coarse_solution = [
            [[2.02954248, 2.02954248], [1.97045752, 1.97045752]],
            [[2.06204443, 2.10954305], [1.93795557, 1.89045695]],
        ]
        expected_subgrid_node_flux = [
            [[2.02954248, 2.69599941], [1.97045752, 1.36561964]],
            [[2.0526527, 2.22108788], [1.94563737, 1.77443754]],
        ]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.assertAlmostEqual(
                        coarse_solution[i, j, k], expected_coarse_solution[i][j][k]
                    )
                    self.assertAlmostEqual(
                        subgrid_node_flux[i, j, k], expected_subgrid_node_flux[i][j][k]
                    )


class TestSubgridFluxDataFrameCreator(TestCase):
    def test_creator(self):
        coarse_solution = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0]],
            ]
        )
        flux = coarse_solution.copy()
        expected_df = pd.DataFrame(
            np.array(
                [
                    [1.0, -1.0, 1.0, -1.0],
                    [2.0, -2.0, 2.0, -2.0],
                    [3.0, -3.0, 3.0, -3.0],
                    [4.0, -4.0, 4.0, -4.0],
                    [3.0, -3.0, 3.0, -3.0],
                    [4.0, -4.0, 4.0, -4.0],
                ]
            ).T,
            columns=pd.MultiIndex.from_product([["U0", "U1", "G_0.5"], ["h", "q"]]),
            index=[0, 1, 0, 1],
        )
        df_creator = SubgridFluxDataFrameCreator(1, 1, print_output=False)
        df = df_creator(coarse_solution, flux)
        df = df_creator(coarse_solution, flux, data=df)

        self.assertTrue((df.values == expected_df.values).all())
        self.assertTrue(df.columns.equals(expected_df.columns))
        self.assertTrue(df.index.equals(expected_df.index))


class TestGenerateData(TestCase):
    def test_assert_coarsening_degree(self):
        self.assertRaises(ValueError, GenerateData, 1, coarsening_degree=2, mesh_size=5)

    def test_data_shape(self):
        subgrid_flux_data_path = "test/command/subgrid_flux_data.csv"
        benchmark_data_path = "test/command/benchmark_data.csv"
        create_benchmark = (
            lambda: shallow_water.benchmark.OscillationNoTopographyBenchmark(
                end_time=0.5
            )
        )
        command = GenerateData(
            2,
            create_benchmark=create_benchmark,
            subgrid_flux_data_path=subgrid_flux_data_path,
            benchmark_data_path=benchmark_data_path,
            skip=1,
            mesh_size=8,
            coarsening_degree=2,
            local_degree=1,
            print_output=False,
        )
        command.execute()

        subgrid_flux_data = load_data(subgrid_flux_data_path)
        benchmark_data = load_data(benchmark_data_path)

        self.assertTupleEqual(subgrid_flux_data.shape, (4, 6), msg="Subgrid Flux Data")
        self.assertTupleEqual(benchmark_data.shape, (2, 8), msg="Benchmark Data")
