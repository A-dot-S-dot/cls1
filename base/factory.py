from typing import Tuple, Type

from . import interpolate
from . import ode_solver as os
from .benchmark import Benchmark
from .discretization import DiscreteSolution, finite_element, finite_volume
from .mesh import Mesh, UniformMesh
from .system import SystemVector
from .time_stepping import TimeStepping


class FiniteElementSolutionFactory:
    def __call__(
        self, benchmark: Benchmark, mesh_size: int, polynomial_degree: int
    ) -> Tuple[DiscreteSolution, finite_element.LagrangeSpace]:
        mesh = UniformMesh(benchmark.domain, mesh_size)
        space = finite_element.LagrangeSpace(mesh, polynomial_degree)
        interpolator = interpolate.NodeValuesInterpolator(*space.basis_nodes)
        solution = DiscreteSolution(
            interpolator.interpolate(benchmark.initial_data),
            start_time=benchmark.start_time,
            grid=space.grid,
            space=space,
        )

        return solution, space


class FiniteVolumeSolutionFactory:
    def __call__(
        self, benchmark: Benchmark, mesh_size: int
    ) -> Tuple[DiscreteSolution, finite_volume.FiniteVolumeSpace]:
        mesh = UniformMesh(benchmark.domain, mesh_size)
        space = finite_volume.FiniteVolumeSpace(mesh)
        interpolator = interpolate.CellAverageInterpolator(mesh, 2)
        solution = DiscreteSolution(
            interpolator.interpolate(benchmark.initial_data),
            start_time=benchmark.start_time,
            grid=space.grid,
            space=space,
        )

        return solution, space


# def build_coarse_solution(solver):
#     solver.solution = CoarseSolution(
#         solver.fine_solver.solution, solver.coarsening_degree
#     )


class MeshDependentTimeSteppingFactory:
    def __call__(
        self, benchmark: Benchmark, mesh: Mesh, cfl_number: float
    ) -> TimeStepping:
        return TimeStepping(
            benchmark.end_time,
            cfl_number,
            lambda: mesh.step_length,
            start_time=benchmark.start_time,
        )


class OptimalODESolverFactory:
    def __call__(
        self, element_space: finite_element.LagrangeSpace
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


FINITE_ELEMENT_SOLUTION_FACTORY = FiniteElementSolutionFactory()
FINITE_VOLUME_SOLUTION_FACTORY = FiniteVolumeSolutionFactory()
OPTIMAL_ODE_SOLVER_FACTORY = OptimalODESolverFactory()
MESH_DEPENDENT_TIME_STEPPING = MeshDependentTimeSteppingFactory()
