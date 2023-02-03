from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
import shallow_water
from shallow_water.solver import gmc, godunov, lax_friedrichs
from core import finite_volume
from numpy.testing import assert_equal


class TestGMCLimiter(TestCase):
    dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
    dof_vector_star = np.array(
        [[2.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0], [1.0, 1.0]]
    )
    riemann_solver = shallow_water.RiemannSolver(
        finite_volume.PeriodicBoundaryConditionsApplier((1, 1)),
        gravitational_acceleration=1.0,
    )
    riemann_solver.solve(0.0, dof_vector)

    _low_order_flux = lax_friedrichs.LLFNumericalFLux(riemann_solver)
    _high_order_flux = godunov.GodunovNumericalFlux(1.0)
    gmc_flux = gmc.GMCNumericalFlux(
        VOLUME_SPACE, riemann_solver, _high_order_flux, gamma=1.0
    )

    wave_speed = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    intermediate_state = np.array(
        [[1.25, 0.625], [0.75, 0.875], [0.75, -0.875], [1.25, -0.625], [1.25, 0.625]]
    )
    u_min = np.array(
        [[1.0, -1], [0.0, 0.0], [0.0, -1.0], [0.0, -1.0], [1.0, -1.0], [0.0, 0.0]]
    )
    u_max = np.array(
        [[2.0, 1.0], [2.0, 1.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0], [2.0, 1.0]]
    )
    antidiffusive_flux = np.array(
        [
            [-0.25735931, 0.21446609],
            [-0.5, -0.25],
            [0.5, -0.25],
            [0.25735931, 0.21446609],
            [-0.25735931, 0.21446609],
        ]
    )
    di = gmc_flux._calculate_di()
    u_bar = gmc_flux._calculate_u_bar(di)
    Q_minus, Q_plus = gmc_flux._calculate_local_antidiffusive_flux_bounds(
        dof_vector_star
    )
    P_minus, P_plus = gmc_flux._calculate_signed_antidiffusive_fluxes(
        antidiffusive_flux
    )
    R_minus, R_plus = gmc_flux._calculate_R(P_minus, P_plus, Q_minus, Q_plus)
    alpha = gmc_flux._calculate_alpha(R_minus, R_plus, antidiffusive_flux)

    def test_calculate_di(self):
        expected_di = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
        assert_equal(self.di, expected_di)

    def test_calculate_u_bar(self):
        expected_u_bar = np.array(
            [
                [1.25, 0.0],
                [1.0, 0.75],
                [0.75, 0.0],
                [1.0, -0.75],
                [1.25, 0.0],
                [1.0, 0.75],
            ]
        )
        assert_equal(self.u_bar, expected_u_bar)

    def test_P_minus(self):
        expected_P_minus = np.array(
            [
                [0.0, -0.21446609],
                [-0.25735931, 0.0],
                [-1.0, -0.25],
                [-0.25735931, -0.46446609],
                [0.0, -0.21446609],
                [-0.25735931, 0.0],
            ]
        )
        assert_equal(self.P_minus, expected_P_minus)

    def test_P_plus(self):
        expected_P_plus = np.array(
            [
                [0.51471862, 0.21446609],
                [0.5, 0.46446609],
                [0.0, 0.25],
                [0.5, 0.0],
                [0.51471862, 0.21446609],
                [0.5, 0.46446609],
            ]
        )
        assert_equal(self.P_plus, expected_P_plus)

    def test_Q_minus(self):
        expected_Q_minus = np.array(
            [
                [-5.0, -8.0],
                [-8.0, -7.0],
                [-3.0, -8.0],
                [-8.0, -1.0],
                [-5.0, -8.0],
                [-8.0, -7.0],
            ]
        )
        assert_equal(self.Q_minus, expected_Q_minus)

    def test_Q_plus(self):
        expected_Q_plus = np.array(
            [[3.0, 8.0], [8.0, 1.0], [5.0, 8.0], [8.0, 7.0], [3.0, 8.0], [8.0, 1.0]]
        )
        assert_equal(self.Q_plus, expected_Q_plus)

    def test_R_minus(self):
        expected_R_minus = np.ones((6, 2))
        assert_equal(self.R_minus, expected_R_minus)

    def test_R_plus(self):
        expected_R_plus = np.ones((6, 2))
        assert_equal(self.R_plus, expected_R_plus)

    def test_alpha(self):
        alpha = np.ones((5, 2))
        assert_equal(self.alpha, alpha)
