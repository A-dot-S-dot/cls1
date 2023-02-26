"""A finite element version of the implemented solvers are discussed in
'Bound-preserving and entropy-stable algebraic flux correction schemes for the
shallow water equations with topography' by H. Hajduk and D. Kuzmin.

"""
from typing import Dict, Tuple

import core
import finite_volume
import finite_volume.shallow_water as swe
import numpy as np
from finite_volume.scalar.solver import mcl

from .central import CentralFluxGetter
from .low_order import LowOrderFlux


class MCLFlux(LowOrderFlux):
    """Calculates flux by adding to a diffusive flux a limited antidiffusive
    flux, which can be specified independently.

    """

    _high_order_flux: finite_volume.NumericalFlux
    _boundary_conditions: core.BoundaryConditions

    _limited_height: np.ndarray
    _velocity_bar_state: np.ndarray

    def __init__(
        self,
        gravitational_acceleration: float,
        boundary_conditions: core.BoundaryConditions,
        high_order_flux=None,
        bathymetry=None,
    ):
        LowOrderFlux.__init__(self, gravitational_acceleration, bathymetry)

        self._boundary_conditions = boundary_conditions

        self._high_order_flux = high_order_flux or finite_volume.CentralFlux(
            swe.Flux(gravitational_acceleration)
        )

        self.input_dimension = self._high_order_flux.input_dimension

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, low_order_flux = LowOrderFlux.__call__(
            self, *finite_volume.get_required_values(2, *values)
        )
        _, high_order_flux = self._high_order_flux(*values)
        antidiffusive_flux = high_order_flux + -low_order_flux

        _, limited_fh = self._get_limited_height_fluxes(
            swe.get_height(antidiffusive_flux)
        )
        self._build_limited_heights(limited_fh)

        _, limited_fq = self._get_limited_discharge_fluxes(
            swe.get_discharge(antidiffusive_flux)
        )

        flux_left = -low_order_flux + np.array([-limited_fh, -limited_fq]).T
        flux_right = low_order_flux + np.array([limited_fh, limited_fq]).T

        return flux_left, flux_right

    def _get_limited_height_fluxes(
        self, antidiffusive_flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        height_max, height_min = self._get_height_bounds()
        return mcl.limit(
            antidiffusive_flux,
            self._wave_speed,
            self._modified_height_left,
            self._modified_height_right,
            height_max,
            height_min,
        )

    def _get_height_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        hl, hr = self._boundary_conditions.get_cell_neighbours(
            self._modified_height_left, self._modified_height_right
        )

        self._replace_nans(hl, hr)

        return np.maximum(hl, hr), np.minimum(hl, hr)

    def _replace_nans(self, *values: np.ndarray):
        for x in values:
            if np.isnan(x[0]):
                x[0] = x[1].copy()

            if np.isnan(x[-1]):
                x[-1] = x[-2].copy()

    def _build_limited_heights(
        self,
        limited_height_flux: np.ndarray,
    ):
        self._limited_height = self._height_HLL + limited_height_flux / self._wave_speed

    def _get_limited_discharge_fluxes(
        self, antidiffusive_discharge_flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._build_velocity_bar_state()

        antidiffusive_velocity_flux = self._get_antidiffusive_velocity_fluxes(
            antidiffusive_discharge_flux
        )
        _, limited_fv = self._get_limited_velocity_fluxes(antidiffusive_velocity_flux)

        limited_fq = limited_fv + -self._wave_speed * (
            self._modified_discharge_right
            + -self._limited_height * self._velocity_bar_state
        )

        return -limited_fq, limited_fq

    def _build_velocity_bar_state(self):
        self._velocity_bar_state = (
            self._modified_discharge_left + self._modified_discharge_right
        ) / (self._modified_height_left + self._modified_height_right)

    def _get_antidiffusive_velocity_fluxes(
        self, antidiffusive_discharge_flux: np.ndarray
    ) -> np.ndarray:
        return antidiffusive_discharge_flux + self._wave_speed * (
            self._modified_discharge_right
            + -self._limited_height * self._velocity_bar_state
        )

    def _get_limited_velocity_fluxes(
        self, antidiffusive_flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        velocity_max, velocity_min = self._get_velocity_bounds()
        limited_flux = np.zeros(len(antidiffusive_flux))

        velocity_max_left, velocity_max_right = velocity_max[:-1], velocity_max[1:]
        velocity_min_left, velocity_min_right = velocity_min[:-1], velocity_min[1:]

        flux_maximum = self._wave_speed * np.minimum(
            self._limited_height * (velocity_max_right - self._velocity_bar_state),
            self._limited_height * (self._velocity_bar_state - velocity_min_left),
        )
        flux_minimum = self._wave_speed * np.maximum(
            self._limited_height * (velocity_min_right - self._velocity_bar_state),
            self._limited_height * (self._velocity_bar_state - velocity_max_left),
        )

        limited_flux = np.maximum(
            flux_minimum, np.minimum(flux_maximum, antidiffusive_flux)
        )

        return -limited_flux, limited_flux

    def _get_velocity_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        hl, hr = self._boundary_conditions.get_cell_neighbours(self._height_HLL)
        vl, vr = self._boundary_conditions.get_cell_neighbours(self._velocity_bar_state)
        ql, qr = self._boundary_conditions.get_cell_neighbours(
            self._modified_discharge_left, self._modified_discharge_right
        )

        self._replace_nans(hl, hr, vl, vr, ql, qr)

        velocity_maximum = core.maximum(vl, vr, ql / hl, qr / hr)
        velocity_minimum = core.minimum(vl, vr, ql / hl, qr / hr)

        return velocity_maximum, velocity_minimum


class MCLFluxGetter(swe.FluxGetter):
    _high_order_flux_getter: swe.FluxGetter

    def __init__(self, high_order_flux_getter: swe.FluxGetter):
        self._high_order_flux_getter = high_order_flux_getter

    def __call__(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        space: finite_volume.FiniteVolumeSpace,
        bathymetry=None,
    ) -> finite_volume.NumericalFlux:
        high_order_flux = self._high_order_flux_getter(benchmark, space, bathymetry=0.0)
        boundary_conditions = swe.get_boundary_conditions(
            *benchmark.boundary_conditions,
            inflow_left=benchmark.inflow_left,
            inflow_right=benchmark.inflow_right,
        )
        bathymetry = swe.build_bathymetry_discretization(benchmark, len(space.mesh))

        return MCLFlux(
            benchmark.gravitational_acceleration,
            boundary_conditions,
            high_order_flux=high_order_flux,
            bathymetry=bathymetry,
        )


class MCLSolver(swe.Solver):
    def _build_args(
        self, benchmark: swe.ShallowWaterBenchmark, flux_getter=None, **kwargs
    ) -> Dict:
        self._get_flux = MCLFluxGetter(flux_getter or CentralFluxGetter())
        return super()._build_args(benchmark, **kwargs)
