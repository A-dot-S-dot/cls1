from unittest import TestCase

import numpy as np
import shallow_water
from numpy.testing import assert_almost_equal
from shallow_water.solver import godunov


class TestGodunovFlux(TestCase):
    dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
    numerical_flux = godunov.GodunovNumericalFlux(
        1.0,
        bottom=np.array([0.0, 1.0, 0.0, 0.0]),
        source_term=shallow_water.NaturalSouceTerm(0.25),
    )

    numerical_flux(0.0, dof_vector)

    height_left = np.array([2, 1, 0, 1, 2])
    height_right = np.array([1, 0, 1, 2, 1])
    discharge_left = np.array([0, 1, 0, -1, 0])
    discharge_right = np.array([1, 0, -1, 0, 1])
    flux_left = np.array([[0, 2.0], [1, 1.5], [0, 0], [-1, 1.5], [0.0, 2.0]])
    flux_right = np.array([[1, 1.5], [0, 0], [-1, 1.5], [0, 2.0], [1.0, 1.5]])
    wave_speed_left = np.array([-np.sqrt(2), 0, -2, -2, -np.sqrt(2)])
    wave_speed_right = np.array([2.0, 2.0, 0.0, np.sqrt(2), 2.0])
    source_term = np.array([0.0, 2.0, -2.0, 0.0, 0.0])
    h_HLL = np.array([1.12132034, 0.5, 0.5, 1.12132034, 1.12132034])
    q_HLL = np.array([0.73223305, 0.75, -0.75, -0.73223305, 0.73223305])

    h_star_left, h_star_right = numerical_flux._calculate_h_star(
        h_HLL, wave_speed_left, wave_speed_right
    )
    (
        modified_height_left,
        modified_height_right,
    ) = numerical_flux._calculate_modified_heights(
        h_star_left,
        h_star_right,
        wave_speed_left,
        wave_speed_right,
    )
    q_star = numerical_flux._calculate_q_star(
        q_HLL,
        0.25,
        source_term,
        wave_speed_left,
        wave_speed_right,
    )
    node_flux_left, node_flux_right = numerical_flux._calculate_node_flux(
        np.array([height_left, discharge_left]).T,
        np.array([height_right, discharge_right]).T,
        flux_left,
        flux_right,
        wave_speed_left,
        wave_speed_right,
        modified_height_left,
        modified_height_right,
        q_star,
    )

    def test_topography_step(self):
        numerical_flux = godunov.GodunovNumericalFlux(
            1.0,
            bottom=np.array([1.0, 0.0, 1.0, 1.0]),
            source_term=shallow_water.NaturalSouceTerm(0.25),
        )
        expected_topography_step = np.array([0.0, -1.0, 1.0, 0.0, 0.0])
        np.testing.assert_equal(
            numerical_flux._topography_step, expected_topography_step
        )

    def test_build_alternative_topography_step(self):
        numerical_flux = godunov.GodunovNumericalFlux(1.0)
        self.assertEqual(numerical_flux._topography_step, 0.0)

    def test_build_alternative_source_term_for_no_bottom(self):
        numerical_flux = godunov.GodunovNumericalFlux(1.0)
        self.assertTrue(
            isinstance(numerical_flux._source_term, shallow_water.VanishingSourceTerm)
        )

    def test_build_alternative_source_term_for_constant_bottom(self):
        numerical_flux = godunov.GodunovNumericalFlux(
            1.0, bottom=np.array([1.0, 1.0, 1.0, 1.0])
        )
        self.assertTrue(
            isinstance(numerical_flux._source_term, shallow_water.VanishingSourceTerm)
        )

    def test_no_specified_source_term_error(self):
        self.assertRaises(
            ValueError,
            godunov.GodunovNumericalFlux,
            1.0,
            bottom=np.array([0.0, 0.0, 1.0, 0.0]),
        )

    def test_calculate_h_star(self):
        expected_h_star_left = np.array([1.12132034, 1.5, 0.5, 1.12132034, 1.12132034])
        expected_h_star_right = np.array([1.12132034, 0.5, 1.5, 1.12132034, 1.12132034])

        assert_almost_equal(self.h_star_left, expected_h_star_left)
        assert_almost_equal(self.h_star_right, expected_h_star_right)

    def test_calculate_modified_heights(self):
        expected_modified_height_left = np.array(
            [-1.58578643, 0.0, -1.0, -2.24264068, -1.58578643]
        )
        expected_modified_height_right = np.array(
            [2.24264068, 1.0, 0.0, 1.58578643, 2.24264068]
        )

        assert_almost_equal(self.modified_height_left, expected_modified_height_left)
        assert_almost_equal(self.modified_height_right, expected_modified_height_right)

    def test_calculate_q_star(self):
        expected_q_star = np.array([0.73223305, 0.5, -0.5, -0.73223305, 0.73223305])

        assert_almost_equal(self.q_star, expected_q_star)

    def test_calculate_node_flux(self):
        expected_node_flux_left = np.array(
            [
                [1.24264069, 0.96446609],
                [1.0, 1.5],
                [-1.0, 1.0],
                [-1.24264069, 0.96446609],
                [1.24264069, 0.96446609],
            ]
        )
        expected_node_flux_right = np.array(
            [
                [1.24264069, 0.96446609],
                [1.0, 1.0],
                [-1.0, 1.5],
                [-1.24264069, 0.96446609],
                [1.24264069, 0.96446609],
            ]
        )

        assert_almost_equal(self.node_flux_left, expected_node_flux_left)
        assert_almost_equal(self.node_flux_right, expected_node_flux_right)

    def test_numerical_flux(self):
        expected_flux_left = np.array(
            [
                [1.24264069, 0.96446609],
                [1.0, 1.0],
                [-1.0, 1.5],
                [-1.24264069, 0.96446609],
            ]
        )
        expected_flux_right = -np.array(
            [
                [1.0, 1.5],
                [-1.0, 1.0],
                [-1.24264069, 0.96446609],
                [1.24264069, 0.96446609],
            ]
        )
        flux_left, flux_right = self.numerical_flux(0.0, self.dof_vector)

        assert_almost_equal(flux_left, expected_flux_left)
        assert_almost_equal(flux_right, expected_flux_right)
