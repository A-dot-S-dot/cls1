import core
import defaults
from .space import get_finite_element_solution


class Solver(core.Solver):
    def reinitialize(self, benchmark: core.Benchmark):
        initial_data = get_finite_element_solution(
            benchmark,
            len(self.solution.space.mesh),
            self.solution.space.polynomial_degree,
        )
        self.solution.set_value(initial_data.value, initial_data.time)
        self._ode_solver.reinitialize(initial_data.value, initial_data.time)


class SolverParser(core.SolverParser):
    _cfl_default = defaults.FINITE_ELEMENT_CFL_NUMBER

    def _add_arguments(self):
        self.add_argument(
            "+p",
            "++polynomial-degree",
            help="Set polynomial degree used for finite elements.",
            metavar="<degree>",
            type=core.positive_int,
            default=defaults.POLYNOMIAL_DEGREE,
        )
