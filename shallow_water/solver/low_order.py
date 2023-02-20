from typing import Tuple, TypeVar

import core
import numpy as np
import shallow_water as swe
from shallow_water.core import get_height_positivity_fix

from .solver import ShallowWaterNumericalFlux, ShallowWaterSolver

T = TypeVar("T", bound=core.DiscreteSolution)


class LowOrderFlux(ShallowWaterNumericalFlux):
    """See 'Bound-preserving and entropy-stable algebraic flux correction
    schemes for the shallow water equations with topography' by Hajduk and
    Kuzmin.

    """

    input_dimension = 2
    _riemann_solver: swe.RiemannSolver

    _value_left: np.ndarray
    _value_right: np.ndarray
    _height_HLL: np.ndarray
    _discharge_HLL: np.ndarray
    _height_positivity_fix: np.ndarray
    _modified_height_left: np.ndarray
    _modified_height_right: np.ndarray
    _modified_discharge_left: np.ndarray
    _modified_discharge_right: np.ndarray

    @property
    def _wave_speed(self) -> np.ndarray:
        return self._riemann_solver.wave_speed_right

    @property
    def _flux_left(self) -> np.ndarray:
        return self._riemann_solver.flux_left

    @property
    def _flux_right(self) -> np.ndarray:
        return self._riemann_solver.flux_right

    def __init__(self, gravitational_acceleration: float, bathymetry=None):
        super().__init__(gravitational_acceleration, bathymetry)
        self._riemann_solver = swe.RiemannSolver(
            gravitational_acceleration=gravitational_acceleration,
        )

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, bar_state = self._riemann_solver.solve(value_left, value_right)
        self._value_left, self._value_right = value_left, value_right

        self._height_HLL, self._discharge_HLL = swe.get_height_and_discharge(bar_state)

        self._build_modified_heights()
        self._modify_discharge()

        left_state = np.array(
            [self._modified_height_left, self._modified_discharge_left]
        ).T
        right_state = np.array(
            [self._modified_height_right, self._modified_discharge_right]
        ).T

        return (
            self._wave_speed[:, None] * (left_state + -value_left) + -self._flux_left,
            self._wave_speed[:, None] * (right_state + -value_right) + self._flux_right,
        )

    def _build_modified_heights(self):
        self._height_positivity_fix = get_height_positivity_fix(
            self._height_HLL, self.bathymetry_step
        )
        self._modified_height_left = self._height_HLL + self._height_positivity_fix
        self._modified_height_right = self._height_HLL + -self._height_positivity_fix

    def _modify_discharge(self):
        source_term = self.get_source_term(
            swe.get_average(*swe.get_heights(self._value_left, self._value_right))
        )
        velocity_average = swe.get_average(
            *swe.get_velocities(self._value_left, self._value_right)
        )

        self._modified_discharge_left = (
            self._discharge_HLL
            + velocity_average * self._height_positivity_fix
            + source_term
        )
        self._modified_discharge_right = (
            self._discharge_HLL
            + -velocity_average * self._height_positivity_fix
            + source_term
        )

    def get_source_term(self, height_average: np.ndarray) -> np.ndarray:
        if self.bathymetry_step is None:
            return np.array([0.0])
        else:
            return -(
                self.gravitational_acceleration
                * height_average
                * self._height_positivity_fix
                / self._wave_speed
            )


def get_low_order_flux(
    benchmark: swe.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> LowOrderFlux:
    bathymetry = swe.build_bathymetry_discretization(benchmark, len(mesh))

    return LowOrderFlux(
        benchmark.gravitational_acceleration,
        bathymetry=bathymetry,
    )


class LowOrderSolver(ShallowWaterSolver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = get_low_order_flux
        super().__init__(benchmark, **kwargs)
