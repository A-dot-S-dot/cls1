"""This modules build solver components from arguments."""
from argparse import Namespace
from typing import Iterable

from benchmark import Benchmark
from defaults import CALCULATION_MESH_SIZE, EOC_MESH_SIZE, PLOT_MESH_SIZE
from factory import BENCHMARK_FACTORY
from factory.pde_solver_factory import *

from pde_solver.mesh import Mesh, UniformMesh


class SolverComponents:
    _args: Namespace
    _solver_factories: Iterable[PDESolverFactory]
    _benchmark: Benchmark
    _solver_factory_classes = {
        "cg": ContinuousGalerkinSolverFactory,
        "cg_low": LowOrderCGFactory,
        "mcl": MCLSolverFactory,
        "godunov": SWEGodunovSolverFactory,
    }
    _mesh_sizes = {
        "plot": PLOT_MESH_SIZE,
        "animate": PLOT_MESH_SIZE,
        "eoc": EOC_MESH_SIZE,
        "calculation": CALCULATION_MESH_SIZE,
    }

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
        if hasattr(self._args, "plot") and self._args.plot:
            return "plot"
        elif hasattr(self._args, "animate") and self._args.animate:
            return "animate"
        elif hasattr(self._args, "eoc") and self._args.eoc:
            return "eoc"
        elif hasattr(self._args, "calculation") and self._args.calculation:
            return "calculation"
        else:
            raise NotImplementedError("No known command is available.")

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
        try:
            return self._solver_factory_classes[solver_name]()
        except KeyError:
            raise NotImplementedError(f"no solver '{solver_name}' implemented.")

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
        else:
            command = self._get_command()

            try:
                return self._mesh_sizes[command]
            except KeyError:
                raise NotImplementedError(f"No mesh size for '{command}' implemented.")

    @property
    def solver_factories(self) -> Iterable[PDESolverFactory]:
        return self._solver_factories
