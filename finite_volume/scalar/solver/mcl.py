from typing import Tuple

import core
import finite_volume
import numpy as np
from finite_volume import scalar


def limit(
    antidiffusive_flux: np.ndarray,
    wave_speed: np.ndarray,
    bar_state_left: np.ndarray,
    bar_state_right: np.ndarray,
    local_maximum: np.ndarray,
    local_minimum: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    limited_flux = np.zeros(len(antidiffusive_flux))

    local_maximum_left, local_maximum_right = local_maximum[:-1], local_maximum[1:]
    local_minimum_left, local_minimum_right = local_minimum[:-1], local_minimum[1:]

    flux_maximum = wave_speed * np.minimum(
        local_maximum_right - bar_state_right, bar_state_left - local_minimum_left
    )
    flux_minimum = wave_speed * np.maximum(
        local_minimum_right - bar_state_right, bar_state_left - local_maximum_left
    )

    limited_flux = np.maximum(
        flux_minimum, np.minimum(flux_maximum, antidiffusive_flux)
    )

    return -limited_flux, limited_flux


class MCLFlux(finite_volume.LaxFriedrichsFlux):
    input_dimension = 2
    _high_order_flux: finite_volume.NumericalFlux
    _local_maximum: core.LocalMaximum
    _local_minimum: core.LocalMinimum

    def __init__(
        self,
        riemann_solver: core.RiemannSolver,
        neighbours_indices: core.NeighbourIndicesMapping,
    ):
        super().__init__(riemann_solver)
        self._high_order_flux = finite_volume.CentralFlux(riemann_solver.flux)
        self._local_maximum = core.LocalMaximum(neighbours_indices)
        self._local_minimum = core.LocalMinimum(neighbours_indices)

    @property
    def _wave_speed(self) -> np.ndarray:
        return self._riemann_solver.wave_speed_right

    @property
    def _bar_state(self) -> np.ndarray:
        return self._riemann_solver.intermediate_state

    def __call__(
        self, value_left: np.ndarray, value_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, low_order_flux = super().__call__(value_left, value_right)
        _, high_order_flux = self._high_order_flux(value_left, value_right)

        antidiffusive_flux = high_order_flux - low_order_flux
        value_max, value_min = self._get_bounds(
            finite_volume.get_dof_vector(value_left, value_right)
        )

        limited_flux_left, limited_flux_right = limit(
            antidiffusive_flux,
            self._wave_speed,
            self._bar_state,
            self._bar_state,
            value_max,
            value_min,
        )

        return -low_order_flux + limited_flux_left, low_order_flux + limited_flux_right

    def _get_bounds(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._local_maximum(dof_vector), self._local_minimum(dof_vector)


class MCLFluxGetter(finite_volume.FluxGetter):
    def __call__(
        self, benchmark: core.Benchmark, space: finite_volume.FiniteVolumeSpace
    ) -> finite_volume.NumericalFlux:
        riemann_solver = scalar.get_riemann_solver(benchmark.problem)
        neighbours_mapping = finite_volume.NeighbourIndicesMapping(
            len(space.mesh) + 2, benchmark.boundary_conditions == ("periodic",)
        )
        return MCLFlux(riemann_solver, neighbours_mapping)


class MCLSolver(finite_volume.Solver):
    def __init__(self, benchmark: core.Benchmark, **kwargs):
        self.flux_getter = MCLFluxGetter()
        super().__init__(benchmark, **kwargs)


class MCLParser(finite_volume.SolverParser):
    prog = "mcl-fv"
    name = "Finite Volume MCL Solver"
    solver = MCLSolver
