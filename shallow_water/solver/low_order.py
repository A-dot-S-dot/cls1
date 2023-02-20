from typing import Dict, Optional, Tuple, TypeVar

import core
import defaults
import lib
import numpy as np
import shallow_water
from shallow_water.core import get_height_positivity_fix

from .solver import ShallowWaterSolver

T = TypeVar("T", bound=core.DiscreteSolution)


class LowOrderFlux(lib.NumericalFlux):
    """See 'Bound-preserving and entropy-stable algebraic flux correction
    schemes for the shallow water equations with topography' by Hajduk and
    Kuzmin.

    """

    input_dimension = 2
    _bathymetry_step: np.ndarray
    _value_left: np.ndarray
    _value_right: np.ndarray
    _height_HLL: np.ndarray
    _discharge_HLL: np.ndarray
    _height_positivity_fix: np.ndarray
    _modified_height_left: np.ndarray
    _modified_height_right: np.ndarray
    _modified_discharge_left: np.ndarray
    _modified_discharge_right: np.ndarray

    _riemann_solver: shallow_water.RiemannSolver

    @property
    def _gravitational_acceleration(self) -> float:
        return self._riemann_solver.gravitational_acceleration

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
        self._riemann_solver = shallow_water.RiemannSolver(
            gravitational_acceleration=gravitational_acceleration,
        )

        self._build_topography_step(bathymetry)

    def _build_topography_step(self, bathymetry: Optional[np.ndarray]):
        if bathymetry is None or shallow_water.is_constant(bathymetry):
            self._bathymetry_step = np.array([0])
        else:
            bathymetry = np.array([bathymetry[0], *bathymetry, bathymetry[-1]])
            self._bathymetry_step = np.diff(bathymetry)

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        bar_state, _ = self._riemann_solver.solve(value_left, value_right)
        self._value_left, self._value_right = value_left, value_right

        self._height_HLL, self._discharge_HLL = shallow_water.get_height_and_discharge(
            bar_state
        )

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
            self._height_HLL, self._bathymetry_step
        )
        self._modified_height_left = self._height_HLL + self._height_positivity_fix
        self._modified_height_right = self._height_HLL - self._height_positivity_fix

    def _modify_discharge(self):
        source_term = self.get_source_term(
            np.average(
                shallow_water.get_heights(self._value_left, self._value_right), axis=0
            )
        )
        velocity_average = np.average(
            shallow_water.get_velocities(self._value_left, self._value_right), axis=0
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
        if self._bathymetry_step is None:
            return np.array([0.0])
        else:
            return -(
                self._gravitational_acceleration
                * height_average
                * self._height_positivity_fix
                / self._wave_speed
            )


def get_low_order_flux(
    benchmark: shallow_water.ShallowWaterBenchmark,
    mesh: core.Mesh,
) -> LowOrderFlux:
    bathymetry = shallow_water.build_bathymetry_discretization(benchmark, len(mesh))

    return LowOrderFlux(
        benchmark.gravitational_acceleration,
        bathymetry=bathymetry,
    )


class LowOrderSolver(ShallowWaterSolver):
    def _get_flux(
        self, benchmark: shallow_water.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        return get_low_order_flux(benchmark, mesh)


class CoarseLowOrderSolver(LowOrderSolver, core.CoarseSolver):
    def __init__(self, *args, coarsening_degree=None, **kwargs):
        self._coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        LowOrderSolver.__init__(self, *args, **kwargs)


class AntidiffusiveLowOrderSolver(ShallowWaterSolver):
    def _build_args(
        self, benchmark: shallow_water.ShallowWaterBenchmark, gamma=None, **kwargs
    ) -> Dict:
        self._gamma = gamma or defaults.ANTIDIFFUSION_GAMMA

        return super()._build_args(benchmark, **kwargs)

    def _get_flux(
        self, benchmark: shallow_water.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        shallow_water.assert_constant_bathymetry(benchmark, len(mesh))

        numerical_flux = get_low_order_flux(benchmark, mesh)
        antidiffusive_flux = lib.LinearAntidiffusiveFlux(self._gamma, mesh.step_length)

        return lib.CorrectedNumericalFlux(numerical_flux, antidiffusive_flux)
