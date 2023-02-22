from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

import core
import defaults
import lib
import numpy as np
import shallow_water as swe
from core import finite_volume


class ShallowWaterNumericalFlux(lib.NumericalFlux):
    bathymetry_step: np.ndarray
    gravitational_acceleration: float
    _numerical_flux: lib.NumericalFlux

    def __init__(
        self,
        numerical_flux: lib.NumericalFlux,
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
        value_left, value_right = lib.get_required_values(2, *values)

        return swe.get_average(*swe.get_heights(value_left, value_right))


class FluxGetter:
    def __call__(
        self, benchmark: swe.ShallowWaterBenchmark, mesh: core.Mesh, bathymetry=None
    ) -> lib.NumericalFlux:
        bathymetry = bathymetry or swe.build_bathymetry_discretization(
            benchmark, len(mesh)
        )
        numerical_flux = self._get_flux(benchmark)

        return ShallowWaterNumericalFlux(
            numerical_flux,
            benchmark.gravitational_acceleration,
            bathymetry=bathymetry,
        )

    def _get_flux(self, benchmark: swe.ShallowWaterBenchmark) -> lib.NumericalFlux:
        raise NotImplementedError


class ShallowWaterSolver(core.Solver):
    _get_flux: FluxGetter

    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        args = self._build_args(benchmark, **kwargs)

        core.Solver.__init__(self, **args)

    def _build_args(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        name=None,
        short=None,
        mesh_size=None,
        cfl_number=None,
        ode_solver_type=None,
        save_history=False,
    ) -> Dict:
        solution = finite_volume.get_finite_volume_solution(
            benchmark, mesh_size or defaults.CALCULATE_MESH_SIZE, save_history
        )
        step_length = solution.space.mesh.step_length

        numerical_flux = self._get_flux(benchmark, solution.space.mesh)
        boundary_conditions = swe.get_boundary_conditions(
            *benchmark.boundary_conditions,
            inflow_left=benchmark.inflow_left,
            inflow_right=benchmark.inflow_right,
        )
        right_hand_side = lib.NumericalFluxDependentRightHandSide(
            numerical_flux,
            step_length,
            boundary_conditions,
            swe.RiemannSolver(benchmark.gravitational_acceleration),
        )

        time_stepping = core.get_mesh_dependent_time_stepping(
            benchmark,
            solution.space.mesh,
            cfl_number or defaults.FINITE_VOLUME_CFL_NUMBER,
        )

        return {
            "solution": solution,
            "right_hand_side": right_hand_side,
            "ode_solver_type": ode_solver_type or core.Heun,
            "time_stepping": time_stepping,
            "name": name,
            "short": short,
        }
