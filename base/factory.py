from typing import Type

from . import interpolate
from . import ode_solver as os
from .benchmark import Benchmark
from .discretization import DiscreteSolution, finite_element, finite_volume
from .mesh import Mesh, UniformMesh
from .time_stepping import TimeStepping


def build_finite_element_solution(
    benchmark: Benchmark, mesh_size: int, polynomial_degree: int, save_history=False
) -> DiscreteSolution:
    mesh = UniformMesh(benchmark.domain, mesh_size)
    space = finite_element.LagrangeSpace(mesh, polynomial_degree)
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
) -> DiscreteSolution:
    mesh = UniformMesh(benchmark.domain, mesh_size)
    space = finite_volume.FiniteVolumeSpace(mesh)
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
    element_space: finite_element.LagrangeSpace,
) -> Type[os.ExplicitRungeKuttaMethod]:
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
