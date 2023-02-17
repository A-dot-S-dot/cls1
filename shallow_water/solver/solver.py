from typing import Dict

import core
import defaults
import lib
import shallow_water
from core import finite_volume


class ShallowWaterSolver(core.Solver):
    def __init__(self, benchmark: shallow_water.ShallowWaterBenchmark, **kwargs):
        args = self._build_args(benchmark, **kwargs)

        core.Solver.__init__(self, **args)

    def _build_args(
        self,
        benchmark: shallow_water.ShallowWaterBenchmark,
        name=None,
        short=None,
        mesh_size=None,
        cfl_number=None,
        ode_solver_type=None,
        adaptive=False,
        save_history=False,
    ) -> Dict:
        solution = finite_volume.build_finite_volume_solution(
            benchmark,
            mesh_size or defaults.CALCULATE_MESH_SIZE,
            save_history,
            benchmark.boundary_conditions == "periodic",
        )
        step_length = solution.space.mesh.step_length

        numerical_flux = self._build_flux(benchmark, solution.space.mesh)
        neighbours = shallow_water.build_node_neighbours(
            benchmark.boundary_conditions,
            radius=numerical_flux.input_dimension // 2,
            inflow_left=benchmark.inflow_left,
            inflow_right=benchmark.inflow_right,
        )
        right_hand_side = lib.NumericalFluxDependentRightHandSide(
            numerical_flux, step_length, neighbours
        )

        optimal_time_step = finite_volume.OptimalTimeStep(
            shallow_water.RiemannSolver(benchmark.gravitational_acceleration),
            neighbours,
            step_length,
        )
        time_stepping = core.build_adaptive_time_stepping(
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

    def _build_flux(
        self, benchmark: shallow_water.ShallowWaterBenchmark, mesh: core.Mesh
    ) -> lib.NumericalFlux:
        raise NotImplementedError
