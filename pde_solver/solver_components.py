"""This modules build solver components from arguments."""
from argparse import Namespace
from typing import Sequence

from benchmark import Benchmark
from factory import (
    BENCHMARK_FACTORY,
    PDESolverFactory,
    ContinuousGalerkinSolverFactory,
    MCLSolverFactory,
    LowOrderCGFactory,
)
from mesh import Mesh
from mesh.uniform import UniformMesh


class SolverComponents:
    _args: Namespace
    _solver_factories: Sequence[PDESolverFactory]
    _benchmark: Benchmark
    _mesh: UniformMesh

    def __init__(self, args: Namespace):
        self._args = args
        self._setup_benchmark_factory()
        self._build_solver_factories()

    def _setup_benchmark_factory(self):
        BENCHMARK_FACTORY.problem_name = self._args.problem
        BENCHMARK_FACTORY.benchmark_name = self._args.benchmark
        BENCHMARK_FACTORY.end_time = self._args.end_time

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
        elif solver_name == "cg_low":
            solver_factory = LowOrderCGFactory()
        elif solver_name == "mcl":
            solver_factory = MCLSolverFactory()
        else:
            raise NotImplementedError

        return solver_factory

    @property
    def benchmark(self) -> Benchmark:
        return BENCHMARK_FACTORY.benchmark

    @property
    def mesh(self) -> Mesh:
        return UniformMesh(self.benchmark.domain, self._args.elements_number)

    @property
    def solver_factories(self) -> Sequence[PDESolverFactory]:
        return self._solver_factories
