from unittest import TestCase

import core
import pandas as pd
import numpy as np
from benchmark.shallow_water import OscillationNoTopographyBenchmark
from finite_volume.shallow_water.solver import LaxFriedrichsSolver
from numpy.testing import assert_equal, assert_almost_equal

from command.generate_data import *


class TestDiscreteSolutionWithInputData(TestCase):
    def test_history(self):
        solution = core.DiscreteSolution(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        )
        solution = DiscreteSolutionWithInputData(solution, 1, 1, 2, 1)
        solution.update(1.0, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))
        expected_coarse_node_neighbours_history = np.array(
            [[[2.0, 3.0], [6.0, 7.0]], [[2.0, 3.0], [6.0, 7.0]]]
        )
        expected_fine_node_neighbours_history = np.array(
            [[[3.0, 4.0], [5.0, 6.0]], [[3.0, 4.0], [5.0, 6.0]]]
        )

        assert_equal(
            solution.coarse_node_neighbours_history,
            expected_coarse_node_neighbours_history,
        )
        assert_equal(
            solution.fine_node_neighbours_history, expected_fine_node_neighbours_history
        )


class TestGenerateData(TestCase):
    subgrid_flux_data: pd.DataFrame
    benchmark_data: pd.DataFrame

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        directory = "test/command/"
        command = GenerateData(
            LaxFriedrichsSolver(
                OscillationNoTopographyBenchmark(end_time=2.0),
                mesh_size=8,
                save_history=True,
            ),
            directory=directory,
            solution_number=2,
            coarsening_degree=2,
            input_radius=1,
            print_output=False,
        )
        command.execute()

        self.subgrid_flux_data = core.load_data(directory + "data.csv")
        self.benchmark_data = core.load_data(directory + "benchmark.csv")

    def test_data_shape(self):
        assert_equal(self.subgrid_flux_data.shape, (6, 6))
        assert_equal(self.benchmark_data.shape, (2, 9))

    def test_subgrid_flux_data(self):
        assert_almost_equal(
            self.subgrid_flux_data.values,
            np.array(
                [
                    [
                        2.05341912578735,
                        4.009948800709973,
                        1.9423340940669758,
                        3.0099961674400175,
                        0.7731992250419286,
                        0.24408021488486753,
                    ],
                    [
                        2.062632555208639,
                        4.083980592199165,
                        1.9886541670047864,
                        3.27068054481027,
                        0.27649250153536054,
                        -0.6507100470346749,
                    ],
                    [
                        2.063067950546262,
                        4.077482405361546,
                        2.008924264266433,
                        3.3978269154529475,
                        0.16129690087679638,
                        -0.42372426373137984,
                    ],
                    [
                        1.940765194075499,
                        3.567835806794359,
                        1.9262130652514138,
                        3.998558864798165,
                        0.6211389488443362,
                        5.275958456797383,
                    ],
                    [
                        1.9300040382207517,
                        3.4395265527333923,
                        1.9730351134190152,
                        3.9559423208679534,
                        0.34892148221070185,
                        2.773936512472435,
                    ],
                    [
                        1.9234541328717465,
                        3.3589209449377284,
                        1.986969073905322,
                        3.8914811065030763,
                        0.25040395170189367,
                        1.624589673819795,
                    ],
                ]
            ),
        )

    def test_subgrid_flux_data_indices(self):
        assert_almost_equal(self.subgrid_flux_data.index, [0, 1.25, 2.0, 0, 1.25, 2.0])

    def test_benchmark_data(self):
        assert_almost_equal(
            self.benchmark_data.values,
            np.array(
                [
                    [
                        34852552.0,
                        2.0,
                        0.4851642968757138,
                        5.0,
                        1.6090077127632307,
                        1.8935056304934081,
                        0.38229161043686544,
                        1.0,
                        2.1663315083138595,
                    ],
                    [
                        144159612.0,
                        2.0,
                        0.3180326063380434,
                        3.0,
                        3.2505559750084467,
                        1.7153910632881437,
                        0.41637357366107797,
                        1.0,
                        5.788343204618924,
                    ],
                ]
            ),
        )
