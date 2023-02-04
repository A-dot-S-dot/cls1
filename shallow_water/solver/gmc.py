from typing import Callable, Tuple

import core
import numpy as np
from core import finite_volume
from lib import NUMERICAL_FLUX, LocalMaximum, LocalMinimum


class GMCNumericalFlux:
    """Calculates flux by adding to a diffusive flux a limited antidiffusive
    flux, which can be specified independently.

    Note, the bottom must be constant. For more information see 'Bound-preserving
    flux limiting for high-order explicit Runge-Kutta time discretizations of
    hyperbolic conservation laws' by D.Kuzmin et al.

    """

    _local_maximum: Callable[[np.ndarray], np.ndarray]
    _local_minimum: Callable[[np.ndarray], np.ndarray]
    _riemann_solver: core.RiemannSolver
    _high_order_flux: NUMERICAL_FLUX
    _periodic: bool
    _eps: float
    _gamma: float

    def __init__(
        self,
        volume_space: finite_volume.FiniteVolumeSpace,
        riemann_solver: core.RiemannSolver,
        high_order_flux: NUMERICAL_FLUX,
        gamma=0.1,
        eps=1e-12,
    ):
        self._local_minimum = LocalMinimum(volume_space)
        self._local_maximum = LocalMaximum(volume_space)
        self._riemann_solver = riemann_solver
        self._high_order_flux = high_order_flux
        self._periodic = self._riemann_solver.periodic_boundary_condition
        self._gamma = gamma
        self._eps = eps

    @property
    def riemann_solver(self) -> core.RiemannSolver:
        return self._riemann_solver

    def __call__(
        self, time: float, dof_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._riemann_solver.solve(time, dof_vector)

        low_order_flux = self._riemann_solver.intermediate_flux
        high_order_flux_left, high_order_flux_right = self._high_order_flux(
            time, dof_vector
        )
        high_order_flux = np.array([*high_order_flux_left, -high_order_flux_right[-1]])

        antidiffusive_flux = high_order_flux + -low_order_flux
        limited_flux = self._limit_fluxes(antidiffusive_flux)

        return (
            low_order_flux[:-1] + limited_flux[:-1],
            -low_order_flux[1:] + -limited_flux[1:],
        )

    def _limit_fluxes(
        self,
        antidiffusive_flux: np.ndarray,
    ) -> np.ndarray:
        P_minus, P_plus = self._calculate_signed_antidiffusive_fluxes(
            antidiffusive_flux
        )
        Q_minus, Q_plus = self._calculate_local_antidiffusive_flux_bounds(
            self._riemann_solver.dof_vector_with_applied_boundary_conditions
        )
        R_minus, R_plus = self._calculate_R(P_minus, P_plus, Q_minus, Q_plus)
        alpha = self._calculate_alpha(R_minus, R_plus, antidiffusive_flux)

        limited_flux = alpha * antidiffusive_flux

        return limited_flux

    def _calculate_signed_antidiffusive_fluxes(
        self, antidiffusive_flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        P_minus = np.minimum(antidiffusive_flux[:-1], 0.0) + np.minimum(
            -antidiffusive_flux[1:], 0.0
        )
        P_plus = np.maximum(antidiffusive_flux[:-1], 0.0) + np.maximum(
            -antidiffusive_flux[1:], 0.0
        )

        return (
            self._apply_boundary_conditions(
                P_minus,
                np.minimum(-antidiffusive_flux[0], 0.0),
                np.minimum(antidiffusive_flux[-1], 0.0),
            ),
            self._apply_boundary_conditions(
                P_plus,
                np.maximum(-antidiffusive_flux[0], 0.0),
                np.maximum(antidiffusive_flux[-1], 0.0),
            ),
        )

    def _apply_boundary_conditions(
        self, values: np.ndarray, left_value, right_value
    ) -> np.ndarray:
        if self._periodic:
            return np.array([values[-1], *values, values[0]])
        else:
            return np.array([left_value, *values, right_value])

    def _calculate_local_antidiffusive_flux_bounds(
        self, dof_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        u_min, u_max = self._local_minimum(dof_vector[1:-1]), self._local_maximum(
            dof_vector[1:-1]
        )
        u_min = self._apply_boundary_conditions(
            u_min, np.amin(dof_vector[0:2], axis=0), np.amin(dof_vector[-2:], axis=0)
        )
        u_max = self._apply_boundary_conditions(
            u_max, np.amax(dof_vector[0:2], axis=0), np.amax(dof_vector[-2:], axis=0)
        )
        di = self._calculate_di()
        u_bar = self._calculate_u_bar(di)

        u_min = np.minimum(u_bar, u_min)
        u_max = np.maximum(u_bar, u_max)

        Q_minus = di[:, None] * (u_min + -u_bar + self._gamma * (u_min + -dof_vector))
        Q_plus = di[:, None] * (u_max + -u_bar + self._gamma * (u_max + -dof_vector))

        return Q_minus, Q_plus

    def _calculate_di(self) -> np.ndarray:
        wave_speed = self._riemann_solver.wave_speed_right
        di = wave_speed[:-1] + wave_speed[1:]

        return self._apply_boundary_conditions(di, wave_speed[0], wave_speed[-1])

    def _calculate_u_bar(self, di: np.ndarray) -> np.ndarray:
        wave_speed = self._riemann_solver.wave_speed_right
        product = wave_speed[:, None] * self._riemann_solver.intermediate_state
        summand = product[:-1] + product[1:]

        return (
            self._apply_boundary_conditions(summand, product[0], product[-1])
            / di[:, None]
        )

    def _calculate_R(
        self,
        P_minus: np.ndarray,
        P_plus: np.ndarray,
        Q_minus: np.ndarray,
        Q_plus: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        P_minus = P_minus.copy()
        P_plus = P_plus.copy()

        P_minus[np.abs(P_minus) < self._eps] = np.nan
        P_plus[np.abs(P_plus) < self._eps] = np.nan

        R_minus = np.minimum(Q_minus / P_minus, 1.0)
        R_plus = np.minimum(Q_plus / P_plus, 1.0)

        R_minus[np.isnan(R_minus)] = 1.0
        R_plus[np.isnan(R_plus)] = 1.0

        return R_minus, R_plus

    def _calculate_alpha(
        self,
        R_minus: np.ndarray,
        R_plus: np.ndarray,
        antidiffusive_flux: np.ndarray,
    ) -> np.ndarray:
        R_minus_left = R_minus[:-1]
        R_minus_right = R_minus[1:]
        R_plus_left = R_plus[:-1]
        R_plus_right = R_plus[1:]

        alpha = np.ones(antidiffusive_flux.shape)
        case_1 = antidiffusive_flux > 0
        case_2 = antidiffusive_flux < 0
        alpha[case_1] = np.minimum(R_plus_left, R_minus_right)[case_1]
        alpha[case_2] = np.minimum(R_minus_left, R_plus_right)[case_2]

        return alpha
