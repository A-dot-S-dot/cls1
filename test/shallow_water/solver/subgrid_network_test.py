from unittest import TestCase

import numpy as np
import torch
from shallow_water.solver import subgrid_network


class TestNormalization(TestCase):
    def test_loading(self):
        normalize = subgrid_network.Normalization(torch.Tensor([0]), torch.Tensor([1]))
        input = torch.Tensor([1])
        expected_output = 1

        torch.save(
            normalize.state_dict(), "test/shallow_water/solver/test_normalize.pth"
        )

        loaded_normalize = subgrid_network.Normalization()
        loaded_normalize.load_state_dict(
            torch.load("test/shallow_water/solver/test_normalize.pth")
        )

        self.assertEqual(loaded_normalize(input), expected_output)


class TestCurvature(TestCase):
    def test_scalar_input(self):
        input = [-1, 0, 1, 4]
        expected_output = 16 / (65 ** (3 / 2))
        curvature = subgrid_network.Curvature(0.25)
        output = curvature(*input)

        self.assertEqual(output, expected_output)

    def test_multidimensional_input(self):
        input = np.array(
            [[[-1, -1], [-1, -1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]], [[4, 4], [4, 4]]]
        )
        expected_output = 16 / (65 ** (3 / 2)) * np.ones((2, 2))
        curvature = subgrid_network.Curvature(0.25)
        output = curvature(*input)

        for i in range(2):
            self.assertListEqual(list(output[i]), list(expected_output[i]))


class TestNetwork(TestCase):
    input_dimension = 10

    def test_loading(self):
        model = subgrid_network.NeuralNetwork(
            torch.Tensor(self.input_dimension * [0]),
            torch.Tensor(self.input_dimension * [1]),
        )
        input = torch.Tensor([i for i in range(self.input_dimension)])
        model.eval()
        expected_output = model(input)

        torch.save(model.state_dict(), "test/shallow_water/solver/test_network.pth")

        loaded_model = subgrid_network.NeuralNetwork()
        loaded_model.load_state_dict(
            torch.load("test/shallow_water/solver/test_network.pth")
        )

        loaded_model.eval()

        self.assertSequenceEqual(list(loaded_model(input)), list(expected_output))
