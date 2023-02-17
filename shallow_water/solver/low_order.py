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
    bathymetry_step: np.ndarray
    height_positivity_fix: np.ndarray

    _riemann_solver: shallow_water.RiemannSolver

    @property
    def g(self) -> float:
        return self._riemann_solver.gravitational_acceleration

    @property
    def wave_speed(self) -> np.ndarray:
        return self._riemann_solver.wave_speed_right

    def __init__(self, gravitational_acceleration: float, bathymetry=None):
        self._riemann_solver = shallow_water.RiemannSolver(
            gravitational_acceleration=gravitational_acceleration,
        )

        self._build_topography_step(bathymetry)

    def _build_topography_step(self, bathymetry: Optional[np.ndarray]):
        if bathymetry is None or shallow_water.is_constant(bathymetry):
            self.bathymetry_step = np.array([0])
        else:
            bathymetry = np.array([bathymetry[0], *bathymetry, bathymetry[-1]])
            self.bathymetry_step = np.diff(bathymetry)

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        bar_state, _ = self._riemann_solver.solve(value_left, value_right)

        h_HLL, q_HLL = shallow_water.get_height_and_discharge(bar_state)

        modified_height_left, modified_height_right = self._modify_height(h_HLL)
        modified_discharge_left, modified_discharge_right = self._modify_discharge(
            q_HLL,
            np.average(shallow_water.get_velocities(value_left, value_right), axis=0),
        )

        source_term = self.get_source_term(
            np.average(shallow_water.get_heights(value_left, value_right), axis=0)
        )

        left_state = np.array([modified_height_left, q_HLL + source_term]).T
        right_state = np.array([modified_height_right, q_HLL + source_term]).T

        flux_left = self._riemann_solver.flux_left
        flux_right = self._riemann_solver.flux_right

        return (
            self.wave_speed[:, None] * (left_state + -value_left) + -flux_left,
            self.wave_speed[:, None] * (right_state + -value_right) + flux_right,
        )

    def _modify_height(self, h_HLL: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.height_positivity_fix = get_height_positivity_fix(
            h_HLL, self.bathymetry_step
        )

        return h_HLL + self.height_positivity_fix, h_HLL - self.height_positivity_fix

    def _modify_discharge(
        self, q_HLL: np.ndarray, velocity_average: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            q_HLL + velocity_average * self.height_positivity_fix,
            q_HLL + -velocity_average * self.height_positivity_fix,
        )

    def get_source_term(self, height_average: np.ndarray) -> np.ndarray:
        if self.bathymetry_step is None:
            return np.array([0.0])
        else:
            return -(
                self.g * height_average * self.height_positivity_fix / self.wave_speed
            )


class LowOrderFluxBuilder(lib.NumericalFluxBuilder):
    @staticmethod
    def build_flux(
        benchmark: shallow_water.ShallowWaterBenchmark,
        mesh: core.Mesh,
    ) -> LowOrderFlux:
        topography = shallow_water.build_bathymetry_discretization(benchmark, len(mesh))

        return LowOrderFlux(
            benchmark.gravitational_acceleration,
            bathymetry=topography,
        )


class LowOrderSolver(ShallowWaterSolver):
    def _build_flux(
        self, benchmark: shallow_water.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        return LowOrderFluxBuilder.build_flux(benchmark, mesh)


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

    def _build_flux(
        self, benchmark: shallow_water.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        shallow_water.assert_constant_bathymetry(benchmark, len(mesh))

        numerical_flux = LowOrderFluxBuilder.build_flux(benchmark, mesh)
        antidiffusive_flux = lib.LinearAntidiffusiveFlux(self._gamma, mesh.step_length)

        return lib.CorrectedNumericalFlux(numerical_flux, antidiffusive_flux)
