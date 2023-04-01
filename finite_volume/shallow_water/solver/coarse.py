import core
import defaults
import finite_volume
import finite_volume.shallow_water as swe

from .lax_friedrichs import LaxFriedrichsFluxGetter


class CoarseSolver(swe.Solver):
    _coarsening_degree: int

    def __init__(
        self,
        benchmark: core.Benchmark,
        save_history=False,
        coarsening_degree=None,
        flux_getter=None,
        **kwargs
    ):
        self.flux_getter = flux_getter or LaxFriedrichsFluxGetter()
        self.coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        super().__init__(benchmark, save_history=False, **kwargs)

        self.solution = (
            core.CoarseSolutionWithHistory(self.solution, self.coarsening_degree)
            if save_history
            else core.CoarseSolution(self.solution, self.coarsening_degree)
        )

    def reinitialize(self, benchmark: core.Benchmark):
        mesh = core.UniformMesh(
            benchmark.domain, len(self.solution.space.mesh) * self.coarsening_degree
        )
        interpolator = core.CellAverageInterpolator(mesh, 2)
        value = interpolator.interpolate(benchmark.initial_data)
        self.solution.set_value(value, benchmark.start_time)
        self._ode_solver.reinitialize(value, benchmark.start_time)


class CoarseParser(finite_volume.SolverParser):
    prog = "coarse"
    name = "Coarsened Solver."
    solver = CoarseSolver

    def _add_arguments(self):
        self._add_flux()
        self._add_coarsening_degree()

    def _add_coarsening_degree(self):
        self.add_argument(
            "+c",
            "++coarsening-degree",
            help="Specify the coarsening degree.",
            type=core.positive_int,
            metavar="<degree>",
            default=defaults.COARSENING_DEGREE,
        )
