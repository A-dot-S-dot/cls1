from unittest import TestCase

import numpy as np
from pde_solver.discretization.finite_volume import FiniteVolumeSpace
from pde_solver.mesh import Interval, UniformMesh

import problem.shallow_water as shallow_water


class TestNullifier(TestCase):
    def test_nullifier(self):
        nullifier = shallow_water.Nullifier(epsilon=1)
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
    def test_transformer(self):
        flux = shallow_water.Flux(1)
        dof_vector = np.array([[2.0, -4.0], [2.0, 4.0], [0.0, 0.0]])
        expected_flux = np.array([[-4.0, 10], [4, 10], [0, 0]])

        calculated_flux = flux(dof_vector)

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


class TestWaveSpeed(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    volume_space = FiniteVolumeSpace(mesh)
    wave_speed = shallow_water.WaveSpeed(volume_space, 1)

    def test_wave_speed(self):
        wave_speed_left, wave_speed_right = self.wave_speed(
            np.array([[1.0, -1.0], [4.0, 4.0]])
        )
        expected_wave_speed_left = np.array([-2, -2])
        expected_wave_speed_right = np.array([3, 3])

        for i in range(self.volume_space.node_number):
            self.assertAlmostEqual(
                wave_speed_left[i],
                expected_wave_speed_left[i],
                msg=f"left wave flux, index={i}",
            )
            self.assertAlmostEqual(
                wave_speed_right[i],
                expected_wave_speed_right[i],
                msg=f"right wave flux, index={i}",
            )


class TestNaturalSourceTerm(TestCase):
    def test_numerical_flux(self):
        source_term = shallow_water.NaturalSouceTerm()

        step_length = 0.5
        left_height = np.array([1, 4])
        right_height = np.array([4, 1])
        topography_step = np.array([1, -1])

        expected_discretization = [5, -5]
        discretization = source_term(
            left_height, right_height, topography_step, step_length
        )

        for i in range(len(expected_discretization)):
            self.assertAlmostEqual(
                discretization[i],
                expected_discretization[i],
                msg=f"source term discretization, index={i}",
            )


# class TestWetDryPreservingSourceTermDiscretization(TestNaturalSourceTermDiscretization):
#     step_length = 0.5
#     left_height = [1, 1]
#     right_height = [4, 4]
#     topography_step = [1, -1]
#     expected_source_term = [5, -5]

#     def test_numerical_flux(self):
#         for i in range(len(self.expected_source_term)):
#             self.assertAlmostEqual(
#                 vector.calculate_wet_dry_preserving_source_term_discretization(
#                     self.left_height[i],
#                     self.right_height[i],
#                     self.topography_step[i],
#                     self.step_length,
#                 ),
#                 self.expected_source_term[i],
#                 msg=f"right flux, index={i}",
#             )
