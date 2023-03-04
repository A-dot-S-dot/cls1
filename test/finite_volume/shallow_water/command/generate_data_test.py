from typing import Tuple
from unittest import TestCase

import finite_volume
import numpy as np
import pandas as pd
from benchmark.shallow_water import OscillationNoTopographyBenchmark
from finite_volume.shallow_water.command.generate_data import *
from finite_volume.shallow_water.solver import LaxFriedrichsSolver
from numpy.testing import assert_equal


class TestNumericalFlux(finite_volume.NumericalFlux):
    input_dimension = 2

    def __call__(self, value_left, value_right) -> Tuple[np.ndarray, np.ndarray]:
        flux = (value_left + value_right) / 2

        return -flux, flux


class TestSubgridFlux:
    def __call__(
        self, time: float, dof_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return dof_vector[::2], dof_vector[1::2]


class TestSubgridFluxDataBuilder(TestCase):
    def test_builder(self):
        value = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        solution = core.DiscreteSolutionWithHistory(value)
        solution.update(1.0, 2 * value)
        solution.update(1.0, 3 * value)
        solution.update(1.0, 4 * value)
        builder = SubgridFluxDataBuilder(
            TestNumericalFlux(),
            TestNumericalFlux(),
            coarsening_degree=2,
            input_radius=1,
            node_index=1,
            skip=2,
            print_output=False,
        )
        coarse_values, subgrid_flux = builder(solution)

        expected_coarse_values = [[[1.5, 1.5], [4.5, 4.5]], [[3.5, 3.5], [10.5, 10.5]]]
        expected_subgrid_flux = [[0.0, 0.0], [0.0, 0.0]]

        assert_equal(np.array([*coarse_values]), expected_coarse_values)
        assert_equal(subgrid_flux, expected_subgrid_flux)


class TestSubgridFluxDataFrameBuilder(TestCase):
    def test_creator(self):
        value = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        solution = core.DiscreteSolutionWithHistory(value)
        solution.update(1.0, 2 * value)
        solution.update(1.0, 3 * value)
        solution.update(1.0, 4 * value)
        data_builder = SubgridFluxDataBuilder(
            TestNumericalFlux(),
            TestNumericalFlux(),
            coarsening_degree=2,
            input_radius=1,
            node_index=1,
            skip=2,
            print_output=False,
        )
        df_builder = SubgridFluxDataFrameBuilder(data_builder)

        expected_df = pd.DataFrame(
            np.array(
                [
                    [1.5, 1.5, 3.5, 3.5, 0.0, 0.0],
                    [4.5, 4.5, 10.5, 10.5, 0.0, 0.0],
                    [1.5, 1.5, 3.5, 3.5, 0.0, 0.0],
                    [4.5, 4.5, 10.5, 10.5, 0.0, 0.0],
                ]
            ),
            columns=pd.MultiIndex.from_product([["U0", "U1", "G_0.5"], ["h", "q"]]),
            index=[0.0, 2.0, 0.0, 2.0],
        )
        df = df_builder.get_data_frame(solution)
        df = df_builder.get_data_frame(solution, data=df)

        self.assertTrue(df.equals(expected_df))


class TestGenerateData(TestCase):
    def test_data_shape(self):
        subgrid_flux_data_path = (
            "test/finite_volume/shallow_water/command/subgrid_flux_data.csv"
        )
        benchmark_data_path = (
            "test/finite_volume/shallow_water/command/benchmark_data.csv"
        )
        command = GenerateData(
            LaxFriedrichsSolver(
                OscillationNoTopographyBenchmark(end_time=2.0),
                mesh_size=8,
                save_history=True,
            ),
            solution_number=2,
            subgrid_flux_data_path=subgrid_flux_data_path,
            benchmark_data_path=benchmark_data_path,
            skip=2,
            coarsening_degree=2,
            input_radius=1,
            print_output=False,
        )
        command.execute()

        subgrid_flux_data = load_data(subgrid_flux_data_path)
        benchmark_data = load_data(benchmark_data_path)

        assert_equal(subgrid_flux_data.shape, (4, 6))
        assert_equal(benchmark_data.shape, (2, 8))
