import core
import defaults


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
