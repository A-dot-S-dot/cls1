from typing import Type

import numpy as np

from . import interpolate
from . import ode_solver as os
from .benchmark import Benchmark
from .discretization import DiscreteSolution, DiscreteSolutionWithHistory
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
    solution_type = DiscreteSolutionWithHistory if save_history else DiscreteSolution

    return solution_type(
        interpolator.interpolate(benchmark.initial_data),
        start_time=benchmark.start_time,
        space=space,
    )


def build_finite_volume_solution(
    benchmark: Benchmark, mesh_size: int, save_history=False
) -> DiscreteSolution[FiniteVolumeSpace]:
    mesh = UniformMesh(benchmark.domain, mesh_size)
    space = FiniteVolumeSpace(mesh)
    interpolator = interpolate.CellAverageInterpolator(mesh, 2)
    solution_type = DiscreteSolutionWithHistory if save_history else DiscreteSolution

    return solution_type(
        interpolator.interpolate(benchmark.initial_data),
        start_time=benchmark.start_time,
        space=space,
    )


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
