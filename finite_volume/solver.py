from abc import ABC, abstractmethod
from typing import Dict, Callable

import core
import defaults
import finite_volume
from finite_volume import scalar


class FluxGetter(ABC):
    @abstractmethod
    def __call__(
        self, benchmark: core.Benchmark, mesh: core.Mesh
    ) -> finite_volume.NumericalFlux:
        ...


class Solver(core.Solver):
    _get_boundary_conditions = core.get_boundary_conditions
    _get_flux: FluxGetter

    def __init__(self, benchmark: core.Benchmark, **kwargs):
        args = self._build_args(benchmark, **kwargs)

        core.Solver.__init__(self, **args)

    def _build_args(
        self,
        benchmark: core.Benchmark,
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
        boundary_conditions = self._get_boundary_conditions(
            *benchmark.boundary_conditions,
            inflow_left=benchmark.inflow_left,
            inflow_right=benchmark.inflow_right,
        )
        right_hand_side = finite_volume.NumericalFluxDependentRightHandSide(
            numerical_flux, step_length, boundary_conditions
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
