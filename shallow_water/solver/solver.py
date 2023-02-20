from typing import Dict, Optional, Tuple

import core
import defaults
import lib
import numpy as np
import shallow_water as swe
from core import finite_volume


class ShallowWaterNumericalFlux(lib.NumericalFlux):
    bathymetry_step: np.ndarray
    gravitational_acceleration: float

    def __init__(self, gravitational_acceleration: float, bathymetry=None):
        self.gravitational_acceleration = gravitational_acceleration

        self._build_bathymetry_step(bathymetry)

    def _build_bathymetry_step(self, bathymetry: Optional[np.ndarray]):
        if bathymetry is None or swe.is_constant(bathymetry):
            self.bathymetry_step = np.array([0])
        else:
            bathymetry = np.array([bathymetry[0], *bathymetry, bathymetry[-1]])
            self.bathymetry_step = np.diff(bathymetry)

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flux = self._get_flux(*values)
        source = self._get_source_term(*values)

        return -flux + -source, flux + -source

    def _get_flux(self, *values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

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


class ShallowWaterNumericalFluxDependentRightHandSide(
    lib.NumericalFluxDependentRightHandSide
):
    _numerical_flux: ShallowWaterNumericalFlux

    def _adjust_boundary_conditions(self):
        super()._adjust_boundary_conditions()

        if not self._boundary_conditions.periodic:
            bathymetry_step = self._numerical_flux.bathymetry_step.copy()

            if len(bathymetry_step) > 1:
                self._numerical_flux.bathymetry_step = bathymetry_step[1:-1]


class ShallowWaterSolver(core.Solver):
    _get_flux: lib.FLUX_GETTER[swe.ShallowWaterBenchmark]

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
        right_hand_side = ShallowWaterNumericalFluxDependentRightHandSide(
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
