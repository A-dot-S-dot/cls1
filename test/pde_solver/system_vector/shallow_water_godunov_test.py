from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
import pde_solver.system_vector.shallow_water_godunov as shallow_water_godunov
import problem.shallow_water as shallow_water


class TestSWEGodunovFlux(TestCase):
    dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
    numerical_flux = shallow_water_godunov.GodunovNumericalFlux(
        VOLUME_SPACE,
        1,
        np.array([0.0, 1.0, 0.0, 0.0]),
        shallow_water.NaturalSouceTerm(),
    )
    height_left = np.array([2, 1, 0, 1])
    height_right = np.array([1, 0, 1, 2])
    discharge_left = np.array([0, 1, 0, -1])
    discharge_right = np.array([1, 0, -1, 0])
    flux_left = np.array([[0, 2.0], [1, 1.5], [0, 0], [-1, 1.5]])
    flux_right = np.array([[1, 1.5], [0, 0], [-1, 1.5], [0, 2.0]])
    wave_speed_left = np.array([-np.sqrt(2), 0, -2, -2])
    wave_speed_right = np.array([2.0, 2.0, 0.0, np.sqrt(2)])
    h_HLL = numerical_flux._calculate_HLL_value(
        height_left,
        height_right,
        discharge_left,
        discharge_right,
        wave_speed_left,
        wave_speed_right,
    )
    h_star_left = numerical_flux._calculate_h_star_left(
        h_HLL, wave_speed_left, wave_speed_right
    )
    h_star_right = numerical_flux._calculate_h_star_right(
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
    q_HLL = numerical_flux._calculate_HLL_value(
        discharge_left,
        discharge_right,
        flux_left[:, 1],
        flux_right[:, 1],
        wave_speed_left,
        wave_speed_right,
    )
    source_term = shallow_water.NaturalSouceTerm()(
        height_left,
        height_right,
        numerical_flux._topography_step,
        VOLUME_SPACE.mesh.step_length,
    )
    q_star = numerical_flux._calculate_q_star(
        q_HLL,
        0.25,
        source_term,
        wave_speed_left,
        wave_speed_right,
    )
    node_flux_left = numerical_flux._calculate_node_flux_left(
        modified_height_left,
        q_star,
        wave_speed_left,
        np.array([height_left, discharge_left]).T,
        flux_left,
    )
    node_flux_right = numerical_flux._calculate_node_flux_right(
        modified_height_right,
        q_star,
        wave_speed_right,
        np.array([height_right, discharge_right]).T,
        flux_right,
    )
    cell_flux_left = numerical_flux._calculate_cell_flux_left(node_flux_right)
    cell_flux_right = numerical_flux._calculate_cell_flux_right(node_flux_left)

    def test_topography_step(self):
        numerical_flux = shallow_water_godunov.GodunovNumericalFlux(
            VOLUME_SPACE,
            1,
            np.array([1.0, 0.0, 1.0, 2.0]),
            source_term=shallow_water.NaturalSouceTerm(),
        )
        self.assertListEqual(list(numerical_flux._topography_step), [-1, -1, 1, 1])

    def test_build_alternative_topography_step(self):
        numerical_flux = shallow_water_godunov.GodunovNumericalFlux(VOLUME_SPACE, 1)
        self.assertListEqual(list(numerical_flux._topography_step), [0, 0, 0, 0])

    def test_build_alternative_source_term_for_no_bottom(self):
        numerical_flux = shallow_water_godunov.GodunovNumericalFlux(VOLUME_SPACE, 1)
        self.assertTrue(
            isinstance(numerical_flux._source_term, shallow_water.VanishingSourceTerm)
        )

    def test_build_alternative_source_term_for_constant_bottom(self):
        numerical_flux = shallow_water_godunov.GodunovNumericalFlux(
            VOLUME_SPACE, 1, np.array([1.0, 1.0, 1.0, 1.0])
        )
        self.assertTrue(
            isinstance(numerical_flux._source_term, shallow_water.VanishingSourceTerm)
        )

    def test_no_specified_source_term_error(self):
        self.assertRaises(
            ValueError,
            shallow_water_godunov.GodunovNumericalFlux,
            VOLUME_SPACE,
            1,
            np.array([0.0, 0.0, 1.0, 0.0]),
        )

    def test_calculate_h_HLL(self):
        expected_h_HLL = np.array([1.12132034, 0.5, 0.5, 1.12132034])

        for i in range(4):
            self.assertAlmostEqual(self.h_HLL[i], expected_h_HLL[i])

    def test_calculate_h_star_left(self):
        expected_h_star_left = np.array([1.12132034, 1.5, 0.5, 1.12132034])

        for i in range(4):
            self.assertAlmostEqual(self.h_star_left[i], expected_h_star_left[i])

    def test_calucalte_h_star_right(self):
        expected_h_star_right = np.array([1.12132034, 0.5, 1.5, 1.12132034])

        for i in range(4):
            self.assertAlmostEqual(self.h_star_right[i], expected_h_star_right[i])

    def test_calculate_modified_heights(self):
        expected_modified_height_left = np.array([-1.58578643, 0.0, -1.0, -2.24264068])
        expected_modified_height_right = np.array([2.24264068, 1.0, 0.0, 1.58578643])

        for i in range(4):
            self.assertAlmostEqual(
                self.modified_height_left[i], expected_modified_height_left[i]
            )
            self.assertAlmostEqual(
                self.modified_height_right[i], expected_modified_height_right[i]
            )

    def test_calculate_q_star(self):
        expected_q_star = np.array([0.73223305, 0.5, -0.5, -0.73223305])

        for i in range(4):
            self.assertAlmostEqual(self.q_star[i], expected_q_star[i])

    def test_calculate_node_flux_left(self):
        expected_node_flux_left = np.array(
            [
                [1.24264069, 0.96446609],
                [1.0, 1.5],
                [-1.0, 1.0],
                [-1.24264069, 0.96446609],
            ]
        )

        for i in range(4):
            for j in range(2):
                self.assertAlmostEqual(
                    self.node_flux_left[i, j], expected_node_flux_left[i, j]
                )

    def test_calculate_node_flux_right(self):
        expected_node_flux_right = np.array(
            [
                [1.24264069, 0.96446609],
                [1.0, 1.0],
                [-1.0, 1.5],
                [-1.24264069, 0.96446609],
            ]
        )

        for i in range(4):
            for j in range(2):
                self.assertAlmostEqual(
                    self.node_flux_right[i, j], expected_node_flux_right[i, j]
                )

    def test_calculate_cell_flux_left(self):
        expected_cell_flux_left = np.array(
            [
                [1.24264069, 0.96446609],
                [1.0, 1.0],
                [-1.0, 1.5],
                [-1.24264069, 0.96446609],
            ]
        )

        for i in range(4):
            for j in range(2):
                self.assertAlmostEqual(
                    self.cell_flux_left[i, j], expected_cell_flux_left[i, j]
                )

    def test_calucalte_cell_flux_right(self):
        expected_cell_flux_right = np.array(
            [
                [1.0, 1.5],
                [-1.0, 1.0],
                [-1.24264069, 0.96446609],
                [1.24264069, 0.96446609],
            ]
        )

        for i in range(4):
            for j in range(2):
                self.assertAlmostEqual(
                    self.cell_flux_right[i, j], expected_cell_flux_right[i, j]
                )

    def test_numerical_flux(self):
        expected_flux_left = np.array(
            [
                [1.24264069, 0.96446609],
                [1.0, 1.0],
                [-1.0, 1.5],
                [-1.24264069, 0.96446609],
            ]
        )
        expected_flux_right = np.array(
            [
                [1.0, 1.5],
                [-1.0, 1.0],
                [-1.24264069, 0.96446609],
                [1.24264069, 0.96446609],
            ]
        )
        flux_left, flux_right = self.numerical_flux(self.dof_vector)

        for i in range(VOLUME_SPACE.dimension):
            for j in range(2):
                self.assertAlmostEqual(
                    flux_left[i, j],
                    expected_flux_left[i, j],
                    msg=f"left numerical flux, (i,j)=({i}, {j})",
                )
                self.assertAlmostEqual(
                    flux_right[i, j],
                    expected_flux_right[i, j],
                    msg=f"right numerical flux, (i,j)=({i}, {j})",
                )
