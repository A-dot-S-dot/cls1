from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
import shallow_water
from shallow_water.solver import gmc, godunov, lax_friedrichs


class TestLocalAntidiffusiveBounds(TestCase):
    dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
    flux = shallow_water.Flux(1)
    wave_speed = shallow_water.MaximumWaveSpeed(VOLUME_SPACE, 1)
    intermediate_state = lax_friedrichs.IntermediateState(
        VOLUME_SPACE, flux, wave_speed
    )
    local_bounds = gmc.LocalAntidiffusiveFluxBounds(
        VOLUME_SPACE, wave_speed, intermediate_state
    )


class TestGMCLimiter(TestCase):
    dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])

    _flux = shallow_water.Flux(1)
    _wave_speed = shallow_water.MaximumWaveSpeed(VOLUME_SPACE, 1)
    _intermediate_state = lax_friedrichs.IntermediateState(
        VOLUME_SPACE, _flux, _wave_speed
    )
    _local_bounds = gmc.LocalAntidiffusiveFluxBounds(
        VOLUME_SPACE, _wave_speed, _intermediate_state, gamma=1.0
    )
    _low_order_flux = lax_friedrichs.LLFNumericalFLux(
        VOLUME_SPACE, _flux, _wave_speed, _intermediate_state
    )
    _high_order_flux = godunov.GodunovNumericalFlux(
        VOLUME_SPACE, 1, shallow_water.Flux(1), shallow_water.WaveSpeed(VOLUME_SPACE, 1)
    )
    gmc_flux = gmc.GMCNumericalFlux(
        VOLUME_SPACE, _low_order_flux, _high_order_flux, _local_bounds
    )

    wave_speed = np.array([2.0, 2.0, 2.0, 2.0])
    intermediate_state = np.array(
        [[1.25, 0.625], [0.75, 0.875], [0.75, -0.875], [1.25, -0.625]]
    )
    u_min = np.array([[0.0, 0.0], [0.0, -1.0], [0.0, -1.0], [1.0, -1.0]])
    u_max = np.array([[2.0, 1.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
    antidiffusive_flux_left = np.array(
        [
            [-0.25735931, 0.21446609],
            [-0.5, -0.25],
            [0.5, -0.25],
            [0.25735931, 0.21446609],
        ]
    )
    antidiffusive_flux_right = np.array(
        [
            [0.5, 0.25],
            [-0.5, 0.25],
            [-0.25735931, -0.21446609],
            [0.25735931, -0.21446609],
        ]
    )
    di = _local_bounds._calculate_di(wave_speed)
    u_bar = _local_bounds._calculate_u_bar(wave_speed, intermediate_state, di)
    Q_minus, Q_plus = _local_bounds(dof_vector)
    P_minus, P_plus = gmc_flux._calculate_signed_antidiffusive_fluxes(
        antidiffusive_flux_left, antidiffusive_flux_right
    )
    R_minus, R_plus = gmc_flux._calculate_R(P_minus, P_plus, Q_minus, Q_plus)
    alpha = gmc_flux._calculate_alpha(R_minus, R_plus, antidiffusive_flux_left)
    flux_left, flux_right = gmc_flux(dof_vector)

    def test_calculate_di(self):
        expected_di = [4.0, 4.0, 4.0, 4.0]

        self.assertListEqual(list(self.di), expected_di)

    def test_calculate_u_bar(self):
        expected_u_bar = [[1.0, 0.75], [0.75, 0.0], [1.0, -0.75], [1.25, 0.0]]

        for i in range(4):
            self.assertListEqual(list(self.u_bar[i]), expected_u_bar[i])

    def test_Q_minus(self):
        expected_Q_minus = [[-8.0, -7.0], [-3.0, -8.0], [-8.0, -1.0], [-5.0, -8.0]]

        for i in range(4):
            self.assertListEqual(list(self.Q_minus[i]), expected_Q_minus[i])

    def test_Q_plus(self):
        expected_Q_plus = [[8.0, 1.0], [5.0, 8.0], [8.0, 7.0], [3.0, 8.0]]

        for i in range(4):
            self.assertListEqual(list(self.Q_plus[i]), expected_Q_plus[i])

    def test_P_minus(self):
        expected_P_minus = [
            [-0.25735931, 0.0],
            [-1.0, -0.25],
            [-0.25735931, -0.46446609],
            [0.0, -0.21446609],
        ]
        for i in range(4):
            self.assertListEqual(list(self.P_minus[i]), expected_P_minus[i])

    def test_P_plus(self):
        expected_P_plus = [
            [0.5, 0.46446609],
            [0.0, 0.25],
            [0.5, 0.0],
            [0.51471862, 0.21446609],
        ]
        for i in range(4):
            self.assertListEqual(list(self.P_plus[i]), expected_P_plus[i])

    def test_R_minus(self):
        expected_R_minus = np.ones((4, 2))

        for i in range(4):
            self.assertListEqual(list(self.R_minus[i]), list(expected_R_minus[i]))

    def test_R_plus(self):
        expected_R_plus = np.ones((4, 2))

        for i in range(4):
            self.assertListEqual(list(self.R_plus[i]), list(expected_R_plus[i]))

    def test_alpha(self):
        alpha = np.ones((4, 2))

        for i in range(4):
            self.assertListEqual(list(self.alpha[i]), list(alpha[i]))
