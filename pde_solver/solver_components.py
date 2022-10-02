"""This modules build solver components from arguments."""
from argparse import Namespace
from typing import Sequence

from benchmark import Benchmark
from defaults import EOC_MESH_SIZE, PLOT_MESH_SIZE
from factory import (
    BENCHMARK_FACTORY,
    ContinuousGalerkinSolverFactory,
    LowOrderCGFactory,
    MCLSolverFactory,
    FiniteElementSolverFactory,
)
from mesh import Mesh
from mesh.uniform import UniformMesh


class SolverComponents:
    _args: Namespace
    _solver_factories: Sequence[FiniteElementSolverFactory]
    _benchmark: Benchmark
    _mesh: UniformMesh

    def __init__(self, args: Namespace):
        self._args = args
        self._setup_benchmark_factory()
        self._build_solver_factories()

    def _setup_benchmark_factory(self):
        BENCHMARK_FACTORY.problem_name = self._args.program
        BENCHMARK_FACTORY.benchmark_number = self._args.benchmark
        BENCHMARK_FACTORY.end_time = self._args.end_time
        BENCHMARK_FACTORY.command = self._get_command()

    def _get_command(self) -> str:
        if self._args.plot:
            return "plot"
        elif self._args.eoc:
            return "eoc"
        else:
            raise NotImplementedError

    def _build_solver_factories(self):
        self._solver_factories = []

        if self._args.solver:
            for solver_args in self._args.solver:
                self._solver_factories.append(self._build_solver_factory(solver_args))

    def _build_solver_factory(
        self, solver_args: Namespace
    ) -> FiniteElementSolverFactory:
        solver_factory = self._get_solver_factory(solver_args.solver)

        solver_factory.attributes = solver_args
        solver_factory.problem_name = self._args.program
        solver_factory.mesh = self.mesh
        solver_factory.initial_data = self.benchmark.initial_data
        solver_factory.start_time = self.benchmark.start_time
        solver_factory.end_time = self.benchmark.end_time

        return solver_factory

    def _get_solver_factory(self, solver_name: str) -> FiniteElementSolverFactory:
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
        return UniformMesh(self.benchmark.domain, self.mesh_size)

    @property
    def mesh_size(self) -> int:
        if self._args.mesh_size:
            return self._args.mesh_size
        elif self._args.plot:
            return PLOT_MESH_SIZE
        elif self._args.eoc:
            return EOC_MESH_SIZE
        else:
            raise NotImplementedError

    @property
    def solver_factories(self) -> Sequence[FiniteElementSolverFactory]:
        return self._solver_factories
