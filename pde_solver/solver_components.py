"""This modules build solver components from arguments."""
from argparse import Namespace
from typing import Sequence

from benchmark import Benchmark
from mesh import Mesh

from factory import *


class SolverComponents:
    _args: Namespace
    _solver_factories: Sequence[PDESolverFactory]
    _benchmark: Benchmark
    _mesh: UniformMesh
    _ode_solver_factory = ODESolverFactory()
    _flux_factory = FluxFactory()
    _artificial_diffusion_factory = ArtificialDiffusionFactory()

    def __init__(self, args: Namespace):
        self._args = args
        self._build_benchmark_factory()
        self._build_solver_factories()

    def _build_benchmark_factory(self):
        if self._args.problem == "advection":
            self._benchmark_factory = AdvectionBenchmarkFactory()
        elif self._args.problem == "burgers":
            self._benchmark_factory = BurgersBenchmarkFactory()
        else:
            raise NotImplementedError

        self._benchmark_factory.benchmark_name = self._args.benchmark
        self._benchmark_factory.end_time = self._args.end_time

    def _build_solver_factories(self):
        self._solver_factories = []
        for solver_args in self._args.solver:
            self._solver_factories.append(self._build_solver_factory(solver_args))

    def _build_solver_factory(self, solver_args: Namespace) -> PDESolverFactory:
        solver_factory = self._get_solver_factory(solver_args.solver)

        solver_factory.attributes = solver_args
        solver_factory.problem_name = self._args.problem
        solver_factory.mesh = self.mesh
        solver_factory.initial_data = self.benchmark.initial_data
        solver_factory.start_time = self.benchmark.start_time
        solver_factory.end_time = self.benchmark.end_time

        return solver_factory

    def _get_solver_factory(self, solver_name: str) -> PDESolverFactory:
        if solver_name == "cg":
            solver_factory = ContinuousGalerkinSolverFactory()
            solver_factory.flux_factory = self._flux_factory
            solver_factory.ode_solver_factory = self._ode_solver_factory
        elif solver_name == "cg_low":
            solver_factory = LowOrderCGFactory()
            solver_factory.flux_factory = self._flux_factory
            solver_factory.ode_solver_factory = self._ode_solver_factory
            solver_factory.artificial_diffusion_factory = (
                self._artificial_diffusion_factory
            )
        else:
            raise NotImplementedError

        return solver_factory

    @property
    def benchmark(self) -> Benchmark:
        return self._benchmark_factory.benchmark

    @property
    def mesh(self) -> Mesh:
        return UniformMesh(self.benchmark.domain, self._args.elements_number)

    @property
    def solver_factories(self) -> Sequence[PDESolverFactory]:
        return self._solver_factories
