from unittest import TestCase

import numpy as np
import torch
from core import finite_volume
from shallow_water.solver.subgrid_network import *
from numpy.testing import assert_almost_equal


class TestNormalization(TestCase):
    def test_loading(self):
        normalize = Normalization(torch.Tensor([0]), torch.Tensor([1]))
        input = torch.Tensor([1])
        expected_output = 1

        torch.save(
            normalize.state_dict(), "test/shallow_water/solver/test_normalize.pth"
        )

        loaded_normalize = Normalization()
        loaded_normalize.load_state_dict(
            torch.load("test/shallow_water/solver/test_normalize.pth")
        )

        self.assertEqual(loaded_normalize(input), expected_output)


class TestCurvature(TestCase):
    def test_scalar_input(self):
        input = [-1, 0, 1, 4]
        expected_output = 16 / (65 ** (3 / 2))
        curvature = Curvature(0.25)
        output = curvature(*input)

        self.assertEqual(output, expected_output)

    def test_multidimensional_input(self):
        input = np.array(
            [[[-1, -1], [-1, -1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]], [[4, 4], [4, 4]]]
        )
        expected_output = 16 / (65 ** (3 / 2)) * np.ones((2, 2))
        curvature = Curvature(0.25)
        output = curvature(*input)

        for i in range(2):
            self.assertListEqual(list(output[i]), list(expected_output[i]))


class TestNetwork(TestCase):
    input_dimension = 10

    def test_loading(self):
        model = NeuralNetwork(
            torch.Tensor(self.input_dimension * [0]),
            torch.Tensor(self.input_dimension * [1]),
        )
        input = torch.Tensor([i for i in range(self.input_dimension)])
        model.eval()
        expected_output = model(input)

        torch.save(model.state_dict(), "test/shallow_water/solver/test_network.pth")

        loaded_model = NeuralNetwork()
        loaded_model.load_state_dict(
            torch.load("test/shallow_water/solver/test_network.pth")
        )

        loaded_model.eval()

        self.assertSequenceEqual(list(loaded_model(input)), list(expected_output))


class TestSubgridNetworkNumericalFlux(TestCase):
    subgrid_flux = NetworkSubgridFlux(
        2,
        finite_volume.PeriodicBoundaryConditionsApplier((2, 2)),
        Curvature(0.25),
        NeuralNetwork(),
        "test/shallow_water/solver/test_subgrid_flux_network.pth",
    )
    vector = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    def test_get_input(self):
        expected_input = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0],
            ]
        )

        assert_almost_equal(
            self.subgrid_flux._get_input(self.vector), expected_input, decimal=4
        )

    def test_flux(self):
        flux_left, flux_right = self.subgrid_flux(0, self.vector)
        self.assertTupleEqual(flux_left.shape, (4, 2))
        self.assertTupleEqual(flux_right.shape, (4, 2))
