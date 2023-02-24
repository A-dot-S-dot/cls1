from typing import Optional, Tuple

import defaults
import finite_volume
import finite_volume.shallow_water as swe
import numpy as np


class NumericalFlux(finite_volume.NumericalFlux):
    bathymetry_step: np.ndarray
    gravitational_acceleration: float
    _numerical_flux: finite_volume.NumericalFlux

    def __init__(
        self,
        numerical_flux: finite_volume.NumericalFlux,
        gravitational_acceleration=None,
        bathymetry=None,
    ):
        self._numerical_flux = numerical_flux
        self.input_dimension = numerical_flux.input_dimension
        self.gravitational_acceleration = (
            gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
        )

        self._build_bathymetry_step(bathymetry)

    def _build_bathymetry_step(self, bathymetry: Optional[np.ndarray | float]):
        if (
            bathymetry is None
            or isinstance(bathymetry, float)
            or swe.is_constant(bathymetry)
        ):
            self.bathymetry_step = np.array([0])
        else:
            bathymetry = np.array([bathymetry[0], *bathymetry, bathymetry[-1]])
            self.bathymetry_step = np.diff(bathymetry)

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flux_left, flux_right = self._numerical_flux(*values)
        source = self._get_source_term(*values)

        return flux_left + -source, flux_right + -source

    def _get_source_term(self, *values: np.ndarray) -> np.ndarray:
        height_average = self._get_height_average(*values)
        height_source = np.zeros(len(height_average))
        discharge_source = (
            self.gravitational_acceleration / 2 * height_average * self.bathymetry_step
        )

        return np.array([height_source, discharge_source]).T

    def _get_height_average(self, *values) -> np.ndarray:
        value_left, value_right = finite_volume.get_required_values(2, *values)

        return swe.get_average(*swe.get_heights(value_left, value_right))


class FluxGetter(finite_volume.FluxGetter):
    def __call__(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        space: finite_volume.FiniteVolumeSpace,
        bathymetry=None,
    ) -> finite_volume.NumericalFlux:
        bathymetry = bathymetry or swe.build_bathymetry_discretization(
            benchmark, len(space.mesh)
        )
        numerical_flux = self._get_flux(benchmark)

        return NumericalFlux(
            numerical_flux,
            benchmark.gravitational_acceleration,
            bathymetry=bathymetry,
        )

    def _get_flux(
        self, benchmark: swe.ShallowWaterBenchmark
    ) -> finite_volume.NumericalFlux:
        raise NotImplementedError


class Solver(finite_volume.Solver):
    _get_boundary_conditions = swe.get_boundary_conditions
