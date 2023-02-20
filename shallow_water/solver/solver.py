from typing import Dict, Optional

import core
import defaults
import lib
import numpy as np
import shallow_water as swe
from core import finite_volume


class ShallowWaterNumericalFlux(lib.NumericalFlux):
    _gravitational_acceleration: float
    _bathymetry_step: np.ndarray

    def __init__(self, gravitational_acceleration: float, bathymetry=None):
        self._gravitational_acceleration = gravitational_acceleration

        self._build_bathymetry_step(bathymetry)

    def _build_bathymetry_step(self, bathymetry: Optional[np.ndarray]):
        if bathymetry is None or swe.is_constant(bathymetry):
            self._bathymetry_step = np.array([0])
        else:
            bathymetry = np.array([bathymetry[0], *bathymetry, bathymetry[-1]])
            self._bathymetry_step = np.diff(bathymetry)


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
        adaptive=False,
        save_history=False,
    ) -> Dict:
        solution = finite_volume.get_finite_volume_solution(
            benchmark, mesh_size or defaults.CALCULATE_MESH_SIZE, save_history
        )
        step_length = solution.space.mesh.step_length

        numerical_flux = self._get_flux(benchmark, solution.space.mesh)
        boundary_conditions = swe.get_boundary_conditions(
            benchmark.boundary_conditions,
            inflow_left=benchmark.inflow_left,
            inflow_right=benchmark.inflow_right,
        )
        right_hand_side = lib.NumericalFluxDependentRightHandSide(
            numerical_flux, step_length, boundary_conditions
        )

        optimal_time_step = finite_volume.OptimalTimeStep(
            swe.RiemannSolver(benchmark.gravitational_acceleration),
            boundary_conditions,
            step_length,
        )
        time_stepping = core.get_adaptive_time_stepping(
            benchmark,
            solution,
            optimal_time_step,
            cfl_number or defaults.FINITE_VOLUME_CFL_NUMBER,
            adaptive,
        )
        cfl_checker = core.CFLChecker(optimal_time_step)

        return {
            "solution": solution,
            "right_hand_side": right_hand_side,
            "ode_solver_type": ode_solver_type or core.ForwardEuler,
            "time_stepping": time_stepping,
            "name": name,
            "short": short,
            "cfl_checker": cfl_checker,
        }
