from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
from shallow_water.solver import lax_friedrichs
import shallow_water


class TestLocalLaxFriedrichsIntermediateState(TestCase):
    flux = shallow_water.Flux(1)
    wave_speed = shallow_water.MaximumWaveSpeed(VOLUME_SPACE, 1)
    intermediate_state = lax_friedrichs.LocalLaxFriedrichsIntermediateState(
        VOLUME_SPACE, flux, wave_speed
    )

    def test_intermediate_state(self):
        dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
        intermediate_state = self.intermediate_state(dof_vector)

        expected_intermediate_state = np.array(
            [[1.25, 0.625], [0.75, 0.875], [0.75, -0.875], [1.25, -0.625]]
        )

        for i in range(VOLUME_SPACE.node_number):
            for j in range(2):
                self.assertAlmostEqual(
                    intermediate_state[i, j],
                    expected_intermediate_state[i, j],
                    msg=f"intermediate_state, index=({i}, {j})",
                )


class TestLocalLaxFriedrichNumericalFlux(TestCase):
    dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
    flux = shallow_water.Flux(1)
    wave_speed = shallow_water.MaximumWaveSpeed(VOLUME_SPACE, 1)
    intermediate_state = lax_friedrichs.LocalLaxFriedrichsIntermediateState(
        VOLUME_SPACE, flux, wave_speed
    )
    numerical_flux = lax_friedrichs.LocalLaxFriedrichsFlux(
        VOLUME_SPACE, flux, wave_speed, intermediate_state
    )
    value_left = dof_vector[VOLUME_SPACE.left_cell_indices]
    flux_left = shallow_water.Flux(1)(value_left)
    wave_speed = np.array([2.0, 2.0, 2.0, 2.0])
    intermediate_state = np.array(
        [[1.25, 0.625], [0.75, 0.875], [0.75, -0.875], [1.25, -0.625]]
    )
    node_flux = numerical_flux._calculate_node_flux(
        wave_speed, intermediate_state, value_left, flux_left
    )

    def test_calculate_node_flux(self):
        expected_node_flux = np.array(
            [[1.5, 0.75], [1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75]]
        )

        for i in range(4):
            for j in range(2):
                self.assertAlmostEqual(self.node_flux[i, j], expected_node_flux[i, j])

    def test_numerical_flux(self):
        expected_flux_left = np.array(
            [[1.5, 0.75], [1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75]]
        )
        expected_flux_right = np.array(
            [[1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75], [1.5, 0.75]]
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
