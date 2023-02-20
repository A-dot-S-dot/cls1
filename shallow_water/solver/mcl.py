"""A finite element version of the implemented solvers are discussed in
'Bound-preserving and entropy-stable algebraic flux correction schemes for the
shallow water equations with topography' by H. Hajduk and D. Kuzmin.

"""
from typing import Dict, Tuple

import core
import lib
import numpy as np
import shallow_water

from .central import get_central_flux
from .low_order import LowOrderFlux
from .solver import ShallowWaterSolver


class MCLFlux(LowOrderFlux, lib.NumericalFluxWithArbitraryInput):
    """Calculates flux by adding to a diffusive flux a limited antidiffusive
    flux, which can be specified independently.

    """

    _high_order_flux: lib.NumericalFlux
    _boundary_conditions: core.BoundaryConditions

    _low_order_flux_left: np.ndarray
    _low_order_flux_right: np.ndarray
    _antidiffusive_height_flux_left: np.ndarray
    _antidiffusive_height_flux_right: np.ndarray
    _antidiffusive_discharge_flux_left: np.ndarray
    _antidiffusive_discharge_flux_right: np.ndarray
    _height_minimum_left: np.ndarray
    _height_minimum_right: np.ndarray
    _height_maximum_left: np.ndarray
    _height_maximum_right: np.ndarray
    _limited_height_flux_left: np.ndarray
    _limited_height_flux_right: np.ndarray
    _limited_discharge_flux_left: np.ndarray
    _limited_discharge_flux_right: np.ndarray
    _limited_height_left: np.ndarray
    _limited_height_right: np.ndarray
    _velocity_bar_state: np.ndarray
    _velocity_minimum_left: np.ndarray
    _velocity_minimum_right: np.ndarray
    _velocity_maximum_left: np.ndarray
    _velocity_maximum_right: np.ndarray

    def __init__(
        self,
        gravitational_acceleration: float,
        high_order_flux: lib.NumericalFlux,
        boundary_conditions: core.BoundaryConditions,
        bathymetry=None,
    ):
        LowOrderFlux.__init__(self, gravitational_acceleration, bathymetry)

        self.input_dimension = high_order_flux.input_dimension

        self._high_order_flux = high_order_flux
        self._boundary_conditions = boundary_conditions

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._low_order_flux_left, self._low_order_flux_right = LowOrderFlux.__call__(
            self, *self._get_required_values(2, *values)
        )
        self._build_antidiffusive_fluxes(*values)
        self._build_limited_height_fluxes()
        self._build_limited_discharge_fluxes()

        flux_left = (
            self._low_order_flux_left
            + np.array(
                [
                    self._limited_height_flux_left,
                    self._limited_discharge_flux_left,
                ]
            ).T
        )
        flux_right = (
            self._low_order_flux_right
            + np.array(
                [
                    self._limited_height_flux_right,
                    self._limited_discharge_flux_right,
                ]
            ).T
        )
        return flux_left, flux_right

    def _build_antidiffusive_fluxes(self, *values: np.ndarray):
        high_order_flux_left, high_order_flux_right = self._high_order_flux(*values)

        antidiffusive_flux_left = high_order_flux_left + -self._low_order_flux_left
        antidiffusive_flux_right = high_order_flux_right + -self._low_order_flux_right

        self._antidiffusive_height_flux_left = antidiffusive_flux_left[:, 0]
        self._antidiffusive_height_flux_right = antidiffusive_flux_right[:, 0]
        self._antidiffusive_discharge_flux_left = antidiffusive_flux_left[:, 1]
        self._antidiffusive_discharge_flux_right = antidiffusive_flux_right[:, 1]

    def _build_limited_height_fluxes(self):
        self._build_height_bounds()

        self._limited_height_flux_left = self._limit_antidiffusive_flux(
            self._antidiffusive_height_flux_left,
            inflow_value=self._modified_height_right,
            outflow_value=self._modified_height_left,
            inflow_min=self._height_minimum_right,
            inflow_max=self._height_maximum_right,
            outflow_min=self._height_minimum_left,
            outflow_max=self._height_maximum_left,
        )
        self._limited_height_flux_right = self._limit_antidiffusive_flux(
            self._antidiffusive_height_flux_right,
            inflow_value=self._modified_height_left,
            outflow_value=self._modified_height_right,
            inflow_min=self._height_minimum_left,
            inflow_max=self._height_maximum_left,
            outflow_min=self._height_minimum_right,
            outflow_max=self._height_maximum_right,
        )

    def _build_height_bounds(self):
        hl, hr = self._boundary_conditions.get_cell_neighbours(
            self._modified_height_left, self._modified_height_right
        )

        self._replace_nans(hl, hr)

        height_minimum = np.minimum(hl, hr)
        height_maximum = np.maximum(hl, hr)

        self._height_minimum_left = height_minimum[:-1]
        self._height_minimum_right = height_minimum[1:]
        self._height_maximum_left = height_maximum[:-1]
        self._height_maximum_right = height_maximum[1:]

    def _replace_nans(self, *values: np.ndarray):
        for x in values:
            if np.isnan(x[0]):
                x[0] = x[1].copy()

            if np.isnan(x[-1]):
                x[-1] = x[-2].copy()

    def _limit_antidiffusive_flux(
        self,
        antidiffusive_flux: np.ndarray,
        inflow_value: np.ndarray,
        inflow_min: np.ndarray,
        inflow_max: np.ndarray,
        outflow_value: np.ndarray,
        outflow_max: np.ndarray,
        outflow_min: np.ndarray,
    ) -> np.ndarray:
        limited_flux = np.zeros(len(antidiffusive_flux))

        positive_flux_case = antidiffusive_flux >= 0
        negative_flux_case = antidiffusive_flux < 0

        limited_flux[positive_flux_case] = np.minimum(
            antidiffusive_flux[positive_flux_case],
            self._wave_speed[positive_flux_case]
            * np.maximum(
                inflow_max[positive_flux_case] - inflow_value[positive_flux_case],
                outflow_value[positive_flux_case] - outflow_min[positive_flux_case],
            ),
        )

        limited_flux[negative_flux_case] = np.maximum(
            antidiffusive_flux[negative_flux_case],
            self._wave_speed[negative_flux_case]
            * np.maximum(
                inflow_min[negative_flux_case] - inflow_value[negative_flux_case],
                outflow_value[negative_flux_case] - outflow_max[negative_flux_case],
            ),
        )

        return limited_flux

    def _build_limited_discharge_fluxes(self):
        self._build_limited_heights()
        self._build_velocity_bounds()

        self._limited_discharge_flux_left = self._limit_antidiffusive_flux(
            self._antidiffusive_discharge_flux_left,
            inflow_value=self._modified_discharge_right,
            outflow_value=self._modified_discharge_left,
            inflow_min=self._velocity_minimum_right * self._limited_height_right,
            inflow_max=self._velocity_maximum_right * self._limited_height_right,
            outflow_min=self._velocity_minimum_left * self._limited_height_left,
            outflow_max=self._velocity_maximum_left * self._limited_height_left,
        )
        self._limited_discharge_flux_right = self._limit_antidiffusive_flux(
            self._antidiffusive_discharge_flux_right,
            inflow_value=self._modified_discharge_left,
            outflow_value=self._modified_discharge_right,
            inflow_min=self._velocity_minimum_left * self._limited_height_left,
            inflow_max=self._velocity_maximum_left * self._limited_height_left,
            outflow_min=self._velocity_minimum_right * self._limited_height_right,
            outflow_max=self._velocity_maximum_right * self._limited_height_right,
        )

    def _build_limited_heights(self):
        self._limited_height_left = (
            self._height_HLL + self._limited_height_flux_left / self._wave_speed
        )
        self._limited_height_right = (
            self._height_HLL + self._limited_height_flux_right / self._wave_speed
        )

    def _build_velocity_bounds(self):
        self._build_velocity_bar_state()

        hl, hr = self._boundary_conditions.get_cell_neighbours(self._height_HLL)
        vl, vr = self._boundary_conditions.get_cell_neighbours(self._velocity_bar_state)
        ql, qr = self._boundary_conditions.get_cell_neighbours(
            self._modified_discharge_left, self._modified_discharge_right
        )

        velocity_minimum = np.minimum(np.minimum(vl, ql / hl), np.minimum(vr, qr / hr))
        velocity_maximum = np.maximum(np.maximum(vl, ql / hl), np.maximum(vr, qr / hr))

        self._velocity_minimum_left = velocity_minimum[:-1]
        self._velocity_minimum_right = velocity_minimum[1:]
        self._velocity_maximum_left = velocity_maximum[:-1]
        self._velocity_maximum_right = velocity_maximum[1:]

    def _build_velocity_bar_state(self):
        self._velocity_bar_state = (
            self._modified_discharge_left + self._modified_discharge_right
        ) / (self._modified_height_left + self._modified_height_right)


class EntropyStableMCLFlux(MCLFlux):
    ...


def get_mcl_flux(
    benchmark: shallow_water.ShallowWaterBenchmark,
    mesh: core.Mesh,
    high_order_flux_getter: lib.FLUX_GETTER[shallow_water.ShallowWaterBenchmark],
) -> MCLFlux:
    high_order_flux = high_order_flux_getter(benchmark, mesh)
    boundary_conditions = shallow_water.get_boundary_conditions(
        benchmark.boundary_conditions,
        inflow_left=benchmark.inflow_left,
        inflow_right=benchmark.inflow_right,
    )
    bathymetry = shallow_water.build_bathymetry_discretization(benchmark, len(mesh))

    return MCLFlux(
        benchmark.gravitational_acceleration,
        high_order_flux,
        boundary_conditions,
        bathymetry=bathymetry,
    )


class MCLSolver(ShallowWaterSolver):
    _high_order_flux_getter: lib.FLUX_GETTER[shallow_water.ShallowWaterBenchmark]

    def _build_args(
        self, benchmark: shallow_water.ShallowWaterBenchmark, flux_getter=None, **kwargs
    ) -> Dict:
        self._high_order_flux_getter = flux_getter or get_central_flux
        return super()._build_args(benchmark, **kwargs)

    def _get_flux(
        self, benchmark: shallow_water.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        return get_mcl_flux(benchmark, mesh, self._high_order_flux_getter)
