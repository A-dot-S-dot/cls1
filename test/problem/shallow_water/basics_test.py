from unittest import TestCase

import numpy as np
import problem.shallow_water as shallow_water


class TestNullifier(TestCase):
    def test_nullifier(self):
        nullifier = shallow_water.Nullifier(eps=1)
        dof_vector = np.array([[0.5, 2], [0.5, 0.2], [2, 0.5], [2, 2]])
        expected_modified_dof_vector = np.array([[0, 0], [0, 0], [2, 0.5], [2, 2]])

        nullified_dof_vector = nullifier(dof_vector)

        for i in range(2):
            self.assertSequenceEqual(
                list(nullified_dof_vector[:, i]),
                list(expected_modified_dof_vector[:, i]),
            )


class TestDischargeToVelocityTransformer(TestCase):
    def test_transformer(self):
        transformer = shallow_water.DischargeToVelocityTransformer()
        dof_vector = np.array([[2.0, -4.0], [2.0, 4.0], [0.0, 0.0]])
        expected_transformed_dof_vector = np.array([[2, -2], [2, 2], [0, 0]])

        transformed_dof_vector = transformer(dof_vector)

        for i in range(2):
            self.assertSequenceEqual(
                list(transformed_dof_vector[:, i]),
                list(expected_transformed_dof_vector[:, i]),
            )

    def test_negative_height_error(self):
        self.assertRaises(
            shallow_water.NegativeHeightError,
            shallow_water.DischargeToVelocityTransformer(),
            np.array([[-1, 2], [2, 3]]),
        )


class TestFlux(TestCase):
    def test_flux(self):
        flux = shallow_water.Flux(1)
        dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
        calculated_flux = flux(dof_vector)
        expected_flux = np.array([[1.0, 1.5], [0.0, 0.0], [-1.0, 1.5], [0.0, 2.0]])

        for i in range(2):
            self.assertSequenceEqual(
                list(calculated_flux[:, i]), list(expected_flux[:, i])
            )

    def test_negative_height_error(self):
        self.assertRaises(
            shallow_water.NegativeHeightError,
            shallow_water.Flux(1),
            np.array([[-1, 2], [2, 3]]),
        )


class TestIsConstant(TestCase):
    def test_not_constant(self):
        bottom = np.array([0.0, 0.0, 1.0, 0.0])
        self.assertFalse(shallow_water.is_constant(bottom))

    def test_is_constant(self):
        bottom = np.array([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(shallow_water.is_constant(bottom))
