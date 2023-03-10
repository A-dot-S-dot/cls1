import core
import defaults


class SolverParser(core.SolverParser):
    def _add_arguments(self, mesh_size_default=None, cfl_default=None):
        super()._add_arguments(
            mesh_size_default, cfl_default or defaults.FINITE_ELEMENT_CFL_NUMBER
        )

        self.add_argument(
            "+p",
            "++polynomial-degree",
            help="Set polynomial degree used for finite elements.",
            metavar="<degree>",
            type=core.positive_int,
            default=defaults.POLYNOMIAL_DEGREE,
        )
