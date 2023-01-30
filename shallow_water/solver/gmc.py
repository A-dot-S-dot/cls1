from typing import Callable, Tuple

import core
import numpy as np
from core import finite_volume
from lib import LocalMaximum, LocalMinimum, NumericalFlux


class LocalAntidiffusiveFluxBounds:
    _local_minimum: core.SystemVector

    def __init__(
        self,
        volume_space: finite_volume.FiniteVolumeSpace,
        wave_speed: Callable[[np.ndarray], np.ndarray],
        intermediate_state: Callable[[np.ndarray], np.ndarray],
        gamma=0.1,
    ):
        self._volume_space = volume_space
        self._local_minimum = LocalMinimum(volume_space)
        self._local_maximum = LocalMaximum(volume_space)
        self._intermediate_state = intermediate_state
        self._wave_speed = wave_speed
        self._gamma = gamma

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_min = self._local_minimum(dof_vector)
        u_max = self._local_maximum(dof_vector)
        intermediate_state = self._intermediate_state(dof_vector)
        wave_speed = self._wave_speed(dof_vector)

        di = self._calculate_di(wave_speed)
        u_bar = self._calculate_u_bar(wave_speed, intermediate_state, di)

        u_min = np.minimum(u_bar, u_min)
        u_max = np.maximum(u_bar, u_max)

        Q_minus = di[:, None] * (u_min + -u_bar + self._gamma * (u_min + -dof_vector))
        Q_plus = di[:, None] * (u_max + -u_bar + self._gamma * (u_max + -dof_vector))

        return Q_minus, Q_plus

    def _calculate_di(self, wave_speed: np.ndarray) -> np.ndarray:
        return (
            wave_speed[self._volume_space.left_node_indices]
            + wave_speed[self._volume_space.right_node_indices]
        )

    def _calculate_u_bar(
        self, wave_speed: np.ndarray, intermediate_state: np.ndarray, di: np.ndarray
    ) -> np.ndarray:
        product = wave_speed[:, None] * intermediate_state
        return (
            product[self._volume_space.left_node_indices]
            + product[self._volume_space.right_node_indices]
        ) / di[:, None]


class GMCNumericalFlux(NumericalFlux):
    """Calculates flux by adding to a diffusive flux a limited antidiffusive
    flux, which can be specified independently.

    Note, the bottom must be constant. For more information see 'Bound-preserving
    flux limiting for high-order explicit Runge-Kutta time discretizations of
    hyperbolic conservation laws' by D.Kuzmin et al.

    """

    _volume_space: finite_volume.FiniteVolumeSpace
    _low_order_flux: NumericalFlux
    _high_order_flux: NumericalFlux
    _local_antidiffusive_flux_bounds: core.SystemTuple
    _eps: float

    def __init__(
        self,
        volume_space: finite_volume.FiniteVolumeSpace,
        low_order_flux: NumericalFlux,
        high_order_flux: NumericalFlux,
        local_antidiffusive_flux_bounds: core.SystemTuple,
        eps=1e-12,
    ):
        self._volume_space = volume_space
        self._low_order_flux = low_order_flux
        self._high_order_flux = high_order_flux
        self._local_antidiffusive_flux_bounds = local_antidiffusive_flux_bounds
        self._eps = eps

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        low_order_flux_left, low_order_flux_right = self._low_order_flux(dof_vector)
        high_order_flux_left, high_order_flux_right = self._high_order_flux(dof_vector)
        antidiffusive_flux_left, antidiffusive_flux_right = (
            high_order_flux_left + -low_order_flux_left,
            high_order_flux_right + -low_order_flux_right,
        )
        limited_flux_left, limited_flux_right = self._limit_fluxes(
            dof_vector, antidiffusive_flux_left, antidiffusive_flux_right
        )

        return (
            low_order_flux_left + limited_flux_left,
            low_order_flux_right + limited_flux_right,
        )

    def _limit_fluxes(
        self,
        dof_vector: np.ndarray,
        antidiffusive_flux_left: np.ndarray,
        antidiffusive_flux_right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        P_minus, P_plus = self._calculate_signed_antidiffusive_fluxes(
            antidiffusive_flux_left, antidiffusive_flux_right
        )
        Q_minus, Q_plus = self._local_antidiffusive_flux_bounds(dof_vector)
        R_minus, R_plus = self._calculate_R(P_minus, P_plus, Q_minus, Q_plus)
        alpha = self._calculate_alpha(R_minus, R_plus, antidiffusive_flux_left)

        limited_flux_left = (
            alpha[self._volume_space.left_node_indices] * antidiffusive_flux_left
        )
        limited_flux_right = (
            alpha[self._volume_space.right_node_indices] * antidiffusive_flux_right
        )

        return limited_flux_left, limited_flux_right

    def _calculate_signed_antidiffusive_fluxes(
        self, antidiffusive_flux_left: np.ndarray, antidiffusive_flux_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        P_minus = np.minimum(antidiffusive_flux_left, 0.0) + np.minimum(
            antidiffusive_flux_right, 0.0
        )
        P_plus = np.maximum(antidiffusive_flux_left, 0.0) + np.maximum(
            antidiffusive_flux_right, 0.0
        )

        return P_minus, P_plus

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
        antidiffusive_flux_left: np.ndarray,
    ) -> np.ndarray:
        R_minus_left = R_minus[self._volume_space.left_cell_indices]
        R_minus_right = R_minus[self._volume_space.right_cell_indices]
        R_plus_left = R_plus[self._volume_space.left_cell_indices]
        R_plus_right = R_plus[self._volume_space.right_cell_indices]
        node_flux = antidiffusive_flux_left[self._volume_space.right_cell_indices]

        alpha = np.ones(node_flux.shape)
        case_1 = node_flux > 0
        case_2 = node_flux < 0
        alpha[case_1] = np.minimum(R_plus_left, R_minus_right)[case_1]
        alpha[case_2] = np.minimum(R_minus_left, R_plus_right)[case_2]

        return alpha
