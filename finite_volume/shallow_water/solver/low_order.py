from typing import Tuple, TypeVar

import core
import defaults
import finite_volume
import finite_volume.shallow_water as swe
import numpy as np

T = TypeVar("T", bound=core.DiscreteSolution)


class LowOrderFlux(swe.NumericalFlux):
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

    def __init__(self, gravitational_acceleration=None, bathymetry=None):
        self._gravitational_acceleration = (
            gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
        )

        self._build_bathymetry(bathymetry)
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
        self._build_modified_discharge()

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
        self._height_positivity_fix = swe.get_height_positivity_fix(
            self._height_HLL, self._bathymetry_step
        )
        self._modified_height_left = self._height_HLL + self._height_positivity_fix
        self._modified_height_right = self._height_HLL + -self._height_positivity_fix

    def _build_modified_discharge(self):
        self._build_source_term(
            swe.get_average(*swe.get_heights(self._value_left, self._value_right))
        )
        velocity_average = swe.get_average(
            *swe.get_velocities(self._value_left, self._value_right)
        )

        self._modified_discharge_left = (
            self._discharge_HLL
            + velocity_average * self._height_positivity_fix
            + self._source_term
        )
        self._modified_discharge_right = (
            self._discharge_HLL
            + -velocity_average * self._height_positivity_fix
            + self._source_term
        )

    def _build_source_term(self, height_average: np.ndarray):
        if self._bathymetry_step is None:
            self._source_term = np.array([0.0])
        else:
            self._source_term = -(
                self._gravitational_acceleration
                * height_average
                * self._height_positivity_fix
                / self._wave_speed
            )


class LowOrderFluxGetter(swe.FluxGetter):
    def __call__(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        space: finite_volume.FiniteVolumeSpace,
        bathymetry=None,
    ) -> finite_volume.NumericalFlux:
        bathymetry = bathymetry or swe.build_bathymetry_discretization(
            benchmark, len(space.mesh)
        )

        return LowOrderFlux(
            benchmark.gravitational_acceleration,
            bathymetry=bathymetry,
        )


class LowOrderSolver(swe.Solver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self.flux_getter = LowOrderFluxGetter()
        super().__init__(benchmark, **kwargs)


class LowOrderParser(finite_volume.SolverParser):
    prog = "low-order"
    name = "Low order finite volume scheme"
    solver = LowOrderSolver
