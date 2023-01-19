from typing import Type

import numpy as np

from . import interpolate
from . import ode_solver as os
from .benchmark import Benchmark
from .discretization import DiscreteSolution
from .discretization.finite_element import LagrangeSpace
from .discretization.finite_volume import FiniteVolumeSpace
from .mesh import Mesh, UniformMesh
from .time_stepping import TimeStepping


def build_finite_element_solution(
    benchmark: Benchmark, mesh_size: int, polynomial_degree: int, save_history=False
) -> DiscreteSolution[LagrangeSpace]:
    mesh = UniformMesh(benchmark.domain, mesh_size)
    space = LagrangeSpace(mesh, polynomial_degree)
    interpolator = interpolate.NodeValuesInterpolator(*space.basis_nodes)
    solution = DiscreteSolution(
        interpolator.interpolate(benchmark.initial_data),
        start_time=benchmark.start_time,
        grid=space.grid,
        solver_space=space,
        save_history=save_history,
    )

    return solution


def build_finite_volume_solution(
    benchmark: Benchmark, mesh_size: int, save_history=False
) -> DiscreteSolution[FiniteVolumeSpace]:
    mesh = UniformMesh(benchmark.domain, mesh_size)
    space = FiniteVolumeSpace(mesh)
    interpolator = interpolate.CellAverageInterpolator(mesh, 2)
    solution = DiscreteSolution(
        interpolator.interpolate(benchmark.initial_data),
        start_time=benchmark.start_time,
        grid=space.grid,
        solver_space=space,
        save_history=save_history,
    )

    return solution


def build_mesh_dependent_time_stepping(
    benchmark: Benchmark, mesh: Mesh, cfl_number: float
) -> TimeStepping:
    return TimeStepping(
        benchmark.end_time,
        cfl_number,
        lambda: mesh.step_length,
        start_time=benchmark.start_time,
    )


def build_optimal_ode_solver(
    element_space: LagrangeSpace,
) -> Type[os.ExplicitRungeKuttaMethod[np.ndarray]]:
    degree = element_space.polynomial_degree
    optimal_solver = {
        1: os.Heun,
        2: os.StrongStabilityPreservingRungeKutta3,
        3: os.StrongStabilityPreservingRungeKutta4,
        4: os.RungeKutta8,
        5: os.RungeKutta8,
        6: os.RungeKutta8,
        7: os.RungeKutta8,
    }

    return optimal_solver[degree]


# def build_coarse_solution(solver):
#     solver.solution = CoarseSolution(
#         solver.fine_solver.solution, solver.coarsening_degree
#     )

# ################################################################################
# # EXACT RESOLVED SOLVER (BASED ON GODUNOV)
# ################################################################################
# def build_fine_numerical_flux_container(solver):
#     observed_numerical_flux = vector.ObservedNumericalFlux(
#         solver.fine_solver.numerical_flux
#     )
#     solver.fine_numerical_fluxes = vector.NumericalFluxContainer(
#         vector.NumericalFluxDependentRightHandSide(
#             solver.fine_solver.space, observed_numerical_flux
#         ),
#         observed_numerical_flux,
#     )
#     solver.fine_solver.right_hand_side = solver.fine_numerical_fluxes
#     solver.fine_solver.solve()


# def build_reduced_exact_right_hand_side(solver):
#     build_fine_numerical_flux_container(solver)
#     build_godunov_numerical_flux(solver)
#     solver.numerical_flux = vector.ObservedNumericalFlux(solver.numerical_flux)

#     solver.subgrid_flux = vector.ExactSubgridFlux(
#         solver.fine_numerical_fluxes,
#         solver.solution,
#         solver.numerical_flux,
#         solver.coarsening_degree,
#     )

#     solver.right_hand_side = vector.NumericalFluxDependentRightHandSide(
#         solver.space,
#         vector.CorrectedNumericalFlux(solver.numerical_flux, solver.subgrid_flux),
#     )


# ################################################################################
# # RESOLVED SOLVED WHICH SUBGRID FLUXES ARE CALCULATED BY A NETWORK
# ################################################################################
# def build_reduced_network_time_stepping(solver):
#     solver.cfl_number = solver.cfl_number / solver.coarsening_degree
#     build_shallow_water_godunov_time_stepping(solver)


# def setup_network(solver):
#     solver.network.load_state_dict(torch.load(solver.network_path))
#     solver.network.eval()


# def build_reduced_network_right_hand_side(solver):
#     build_godunov_numerical_flux(solver)
#     curvature = network.Curvature(solver.mesh.step_length)

#     solver.subgrid_flux = vector.NetworkSubgridFlux(
#         solver.network, curvature, solver.local_degree
#     )
#     solver.right_hand_side = vector.NumericalFluxDependentRightHandSide(
#         solver.space,
#         vector.CorrectedNumericalFlux(solver.numerical_flux, solver.subgrid_flux),
#     )
