from typing import Tuple

import core
import finite_volume
import finite_volume.shallow_water as swe

import numpy as np


class LaxWendroffFlux(finite_volume.NumericalFlux):
    input_dimension = 2
    _time_step = 0
    _step_length: float
    _llf_flux: finite_volume.NumericalFlux
    _wave_speed: core.WAVE_SPEED

    def __init__(self, step_length: float, gravitational_acceleration=None):
        self._step_length = step_length
        self._llf_flux = finite_volume.LaxFriedrichsFlux(
            swe.RiemannSolver(gravitational_acceleration)
        )
        self._wave_speed = swe.MaximumWaveSpeed(gravitational_acceleration)

    def set_time_step(self, time_step: float):
        self._time_step = time_step

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, llf_flux = self._llf_flux(value_left, value_right)
        _, wave_speed = self._wave_speed(value_left, value_right)

        flux = (
            llf_flux
            + self._time_step
            * wave_speed[:, None]
            * (value_right - value_left)
            / self._step_length
        )
        return -flux, flux


class LaxWendroffFluxGetter(finite_volume.FluxGetter):
    def __call__(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        space: finite_volume.FiniteVolumeSpace,
        bathymetry=None,
    ) -> finite_volume.NumericalFlux:
        bathymetry = bathymetry or swe.build_bathymetry_discretization(
            benchmark, len(space.mesh)
        )
        numerical_flux = LaxWendroffFlux(
            space.mesh.step_length, benchmark.gravitational_acceleration
        )

        return swe.NumericalFlux(
            numerical_flux,
            benchmark.gravitational_acceleration,
            bathymetry=bathymetry,
        )


class LaxWendroffRightHandSide(finite_volume.NumericalFluxDependentRightHandSide):
    _numerical_flux: LaxWendroffFlux

    def set_time_step(self, time_step: float):
        self._numerical_flux.set_time_step(time_step)


class LaxWendroffSolver(swe.Solver):
    flux_getter = None
    _right_hand_side: LaxWendroffRightHandSide

    def _get_solver_args(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        name=None,
        short=None,
        mesh_size=None,
        cfl_number=None,
        save_history=False,
    ):
        solver_args = super()._get_solver_args(
            benchmark,
            name,
            short,
            mesh_size,
            cfl_number,
            core.ForwardEuler,
            save_history,
        )
        step_length = solver_args["solution"].space.mesh.step_length
        boundary_conditions = self._get_boundary_conditions(
            *benchmark.boundary_conditions,
            radius=1,
            inflow_left=benchmark.inflow_left,
            inflow_right=benchmark.inflow_right,
        )
        solver_args["right_hand_side"] = LaxWendroffRightHandSide(
            LaxWendroffFlux(step_length, benchmark.gravitational_acceleration),
            step_length,
            boundary_conditions,
        )

        return solver_args

    def _update(self, time_step: float):
        self._right_hand_side.set_time_step(time_step)
        super()._update(time_step)


class LaxWendroffParser(finite_volume.SolverParser):
    prog = "lw"
    name = "Lax-Wendroff finite volume scheme"
    solver = LaxWendroffSolver
