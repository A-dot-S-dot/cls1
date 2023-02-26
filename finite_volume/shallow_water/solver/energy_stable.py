"""The fluxes implemented here, are discussed in 'Weel-balanced and energy
stable schemes for the shallow water equations with discontinuous topography' by
U. S. Fjordholm et al."""
from typing import Tuple

import defaults
import finite_volume
import finite_volume.shallow_water as swe
import numpy as np


class EnergyStableFlux(finite_volume.NumericalFlux):
    input_dimension = 2
    _flux: swe.Flux
    _average: np.ndarray

    def __init__(self, gravitational_acceleration=None):
        self._flux = swe.Flux(gravitational_acceleration)

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._average = swe.get_average(value_left, value_right)
        flux = self._flux(self._average)
        return -flux, flux


class FirstOrderDiffusiveEnergyStableFlux(swe.NumericalFlux):
    input_dimension = 2
    _es_flux: EnergyStableFlux

    def __init__(
        self,
        gravitational_acceleration=None,
        bathymetry=None,
    ):
        self.gravitational_acceleration = (
            gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
        )

        self._build_bathymetry_step(bathymetry)
        self._es_flux = EnergyStableFlux(gravitational_acceleration)

    def __call__(self, value_left: np.ndarray, value_right: np.ndarray):
        _, flux = self._es_flux(value_left, value_right)
        average = swe.get_average(value_left, value_right)
        self._build_wave_speeds(swe.get_height(average), swe.get_velocity(average))
        self._build_diffusion_matrix()
        self._build_entropy_step(value_left, value_right)

        diffusion = np.array([D.dot(V) for D, V in zip(self._D, self._entropy_step)])

        new_flux = flux + -diffusion
        source = self._get_source_term(value_left, value_right)

        return -new_flux - source, new_flux - source

    def _build_diffusion_matrix(self):
        self._build_R()
        self._build_Lambda()

        self._D = (
            1
            / 2
            * np.array(
                [np.dot(R, np.dot(L, R.T)) for R, L in zip(self._R, self._Lambda)]
            )
        )

    def _build_wave_speeds(
        self, height_average: np.ndarray, velocity_average: np.ndarray
    ):
        x = np.sqrt(self.gravitational_acceleration * height_average)
        self._l_minus = velocity_average + -x
        self._l_plus = velocity_average + x

    def _build_R(self):
        self._R = (
            1
            / np.sqrt(2 * self.gravitational_acceleration)
            * np.array(
                [[[1.0, 1.0], [lm, lp]] for lm, lp in zip(self._l_minus, self._l_plus)]
            )
        )

    def _build_Lambda(self):
        self._Lambda = (
            1
            / np.sqrt(2 * self.gravitational_acceleration)
            * np.array(
                [
                    np.diag([np.abs(lm), np.abs(lp)])
                    for lm, lp in zip(self._l_minus, self._l_plus)
                ]
            )
        )

    def _build_entropy_step(self, value_left, value_right):
        hl, hr = swe.get_heights(value_left, value_right)
        ul, ur = swe.get_velocities(value_left, value_right)

        self._entropy_step = np.array(
            [
                self.gravitational_acceleration * (hr - hl + self.bathymetry_step)
                - ur**2 / 2
                + ul**2 / 2,
                ur + -ul,
            ]
        ).T


class EnergyStableFluxGetter(swe.FluxGetter):
    def _get_flux(
        self, benchmark: swe.ShallowWaterBenchmark
    ) -> finite_volume.NumericalFlux:
        return EnergyStableFlux(benchmark.gravitational_acceleration)


class FirstOrderDiffusiveEnergyStableFluxGetter(swe.FluxGetter):
    def __call__(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        space: finite_volume.FiniteVolumeSpace,
        bathymetry=None,
    ) -> finite_volume.NumericalFlux:
        bathymetry = bathymetry or swe.build_bathymetry_discretization(
            benchmark, len(space.mesh)
        )
        return FirstOrderDiffusiveEnergyStableFlux(
            benchmark.gravitational_acceleration, bathymetry
        )


class EnergyStableSolver(swe.Solver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = EnergyStableFluxGetter()
        super().__init__(benchmark, **kwargs)


class FirstOrderDiffusiveEnergyStableSolver(swe.Solver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self._get_flux = FirstOrderDiffusiveEnergyStableFluxGetter()
        super().__init__(benchmark, **kwargs)