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
    _energy_stable_flux: EnergyStableFlux
    _entropy: swe.Entropy

    def __init__(
        self,
        gravitational_acceleration=None,
        bathymetry=None,
    ):
        self._gravitational_acceleration = (
            gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
        )

        self._build_bathymetry(bathymetry)
        self._energy_stable_flux = EnergyStableFlux(gravitational_acceleration)
        self._entropy = swe.Entropy(gravitational_acceleration)

    def __call__(self, value_left: np.ndarray, value_right: np.ndarray):
        flux = self._get_raw_flux(value_left, value_right)
        diffusion = self._get_diffusion(value_left, value_right)

        new_flux = flux - diffusion
        source = self._get_source_term(value_left, value_right)

        return -new_flux - source, new_flux - source

    def _get_raw_flux(self, value_left: np.ndarray, value_right: np.ndarray):
        _, flux = self._energy_stable_flux(value_left, value_right)

        return flux

    def _get_diffusion(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> np.ndarray:
        entropy = self._entropy(
            finite_volume.get_dof_vector(value_left, value_right),
            bathymetry=self._bathymetry,
        )
        entropy_step = self._get_entropy_step(entropy)

        diffusion_matrix = self._get_diffusion_matrix(value_left, value_right)

        return (diffusion_matrix @ entropy_step)[..., 0]

    def _get_diffusion_matrix(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> np.ndarray:
        average = swe.get_average(value_left, value_right)
        wave_speed_minus, wave_speed_plus = self._get_wave_speeds(
            swe.get_height(average), swe.get_velocity(average)
        )
        R = self._get_R(wave_speed_minus, wave_speed_plus)
        L = self._get_Lambda(wave_speed_minus, wave_speed_plus)
        R_transpose = np.moveaxis(R, -1, -2)

        return 1 / 2 * R @ L @ R_transpose

    def _get_wave_speeds(
        self, value_height: np.ndarray, value_velocity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        sqrt_gh = np.sqrt(self._gravitational_acceleration * value_height)
        return value_velocity - sqrt_gh, value_velocity + sqrt_gh

    def _get_R(
        self, wave_speed_minus: np.ndarray, wave_speed_plus: np.ndarray
    ) -> np.ndarray:
        R = (
            1
            / np.sqrt(2 * self._gravitational_acceleration)
            * np.array(
                [
                    [np.ones(wave_speed_plus.shape), np.ones(wave_speed_plus.shape)],
                    [wave_speed_minus, wave_speed_plus],
                ]
            )
        )
        return np.moveaxis(R, -1, 0)

    def _get_Lambda(
        self, wave_speed_minus: np.ndarray, wave_speed_plus: np.ndarray
    ) -> np.ndarray:
        Lambda = (
            1
            / np.sqrt(2 * self._gravitational_acceleration)
            * np.array(
                [
                    [np.abs(wave_speed_minus), np.zeros(wave_speed_minus.shape)],
                    [np.zeros(wave_speed_plus.shape), np.abs(wave_speed_plus)],
                ]
            )
        )
        return np.moveaxis(Lambda, -1, 0)

    def _get_entropy_step(self, entropy: np.ndarray) -> np.ndarray:
        entropy_step = np.diff(entropy, axis=0)
        return entropy_step[..., None]


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
        self.flux_getter = EnergyStableFluxGetter()
        super().__init__(benchmark, **kwargs)


class FirstOrderDiffusiveEnergyStableSolver(swe.Solver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self.flux_getter = FirstOrderDiffusiveEnergyStableFluxGetter()
        super().__init__(benchmark, **kwargs)
