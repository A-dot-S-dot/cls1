import time
from argparse import Namespace

from benchmark import Benchmark, NoExactSolutionError, SWEBenchmark
from command import Command
from defaults import PLOT_MESH_SIZE
from factory.pde_solver_factory import PDESolverFactory
from pde_solver.discrete_solution import DiscreteSolution
from pde_solver.solver_components import SolverComponents
from tqdm import tqdm

from .plotter import NothingToPlotError, SolutionPlotter
from .scalar import ScalarFunctionPlotter
from .swe import SWEFunctionPlotter


class PlotCommand(Command):
    _args: Namespace
    _components: SolverComponents
    _plotter: SolutionPlotter

    def __init__(self, args: Namespace):
        self._args = args
        self._components = SolverComponents(args)
        benchmark = self._components.benchmark

        self._build_plotter(benchmark)
        self._build_grid(benchmark)

    def _build_plotter(self, benchmark: Benchmark):
        if isinstance(benchmark, SWEBenchmark):
            self._plotter = SWEFunctionPlotter(benchmark)
        else:
            self._plotter = ScalarFunctionPlotter(benchmark)

    def _build_grid(self, benchmark: Benchmark):
        self._plotter.set_grid(benchmark.domain, PLOT_MESH_SIZE)

    def _get_cell_resolution(self) -> int:
        cell_resolution = 1
        for solver_factory in self._components.solver_factories:
            cell_resolution = max(
                cell_resolution, solver_factory.cell_quadrature_degree
            )

        return cell_resolution

    def execute(self):
        self._add_plots()
        self._plotter.set_suptitle(f"{len(self._components.mesh)} cells")

        if not self._args.plot.quite:
            try:
                self._plotter.show()
            except NothingToPlotError:
                tqdm.write("WARNING: Nothing to plot...")

    def _add_plots(self):
        self._add_exact_solution()
        self._add_discrete_solutions()

        if self._args.plot.initial:
            self._plotter.add_initial_data()

    def _add_exact_solution(self):
        try:
            self._plotter.add_exact_solution()
        except NoExactSolutionError as error:
            tqdm.write("WARNING: " + str(error))

    def _add_discrete_solutions(self):
        for solver_factory in tqdm(
            self._components.solver_factories,
            desc="Calculate solutions",
            unit="solver",
            leave=False,
        ):
            self._add_discrete_solution(solver_factory)

    def _add_discrete_solution(self, solver_factory: PDESolverFactory):
        solution = self._solve_pde(solver_factory)
        self._plot_discrete_solution(solution, solver_factory)

    def _solve_pde(self, solver_factory: PDESolverFactory) -> DiscreteSolution:
        solver = solver_factory.solver

        start_time = time.time()
        solver.solve()

        tqdm.write(
            f"Solved {solver_factory.info} with {solver_factory.dimension} DOFs and {solver.time_stepping.time_steps} time steps in {time.time()-start_time:.2f}s."
        )

        return solver.solution

    def _plot_discrete_solution(
        self, solution: DiscreteSolution, solver_factory: PDESolverFactory
    ):
        self._plotter.add_function_values(
            solver_factory.grid, solution.end_solution, *solver_factory.plot_label
        )
