"""This modules build solver components from arguments."""
from argparse import Namespace
from typing import Sequence

from benchmark import Benchmark
from defaults import EOC_MESH_SIZE, PLOT_MESH_SIZE
from factory import BENCHMARK_FACTORY
from factory.pde_solver_factory import (
    ContinuousGalerkinSolverFactory,
    LowOrderCGFactory,
    MCLSolverFactory,
    PDESolverFactory,
    SWEGodunovSolverFactory,
)

from pde_solver.mesh import Mesh, UniformMesh


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

    def _build_solver_factory(self, solver_args: Namespace) -> PDESolverFactory:
        solver_factory = self._get_solver_factory(solver_args.solver)

        solver_factory.attributes = solver_args
        solver_factory.problem_name = self._args.program
        solver_factory.mesh = self.mesh
        solver_factory.benchmark = self.benchmark

        return solver_factory

    def _get_solver_factory(self, solver_name: str) -> PDESolverFactory:
        if solver_name == "cg":
            solver_factory = ContinuousGalerkinSolverFactory()
        elif solver_name == "cg_low":
            solver_factory = LowOrderCGFactory()
        elif solver_name == "mcl":
            solver_factory = MCLSolverFactory()
        elif solver_name == "godunov":
            solver_factory = SWEGodunovSolverFactory()
        else:
            raise NotImplementedError(f"no solver '{solver_name}' implemented.")

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
    def solver_factories(self) -> Sequence[PDESolverFactory]:
        return self._solver_factories
