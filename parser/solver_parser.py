from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from defaults import *

from . import types as parser_type


class CGParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog="cg",
            description="Continuous Galerkin Solver",
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )
        self._add_polynomial_degree()
        self._add_exact_flux()
        self._add_cfl_number()
        self._add_label()

    def _add_polynomial_degree(self):
        self.add_argument(
            "+p",
            "++polynomial-degree",
            help="polynomial degree used for finite elements",
            metavar="degree",
            type=parser_type.positive_int,
            default=POLYNOMIAL_DEGREE,
        )

    def _add_exact_flux(self):
        self.add_argument(
            "++exact-flux", action="store_true", help="calculate flux matrices exactly"
        )

    def _add_cfl_number(self):
        self.add_argument(
            "++cfl",
            help="specify the cfl number for time stepping",
            type=parser_type.positive_float,
            metavar="number",
            dest="cfl_number",
            default=CFL_NUMBER,
        )

    def _add_label(self):
        self.add_argument("++label", type=str, help="label for ploting")


class LowCGParser(CGParser):
    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog="cg_low",
            description="Low order Continuous Galerkin Solver",
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )
        self._add_polynomial_degree()
        self._add_cfl_number()
        self._add_label()


class MCLParser(CGParser):
    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog="mcl",
            description="MCL Limiter",
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )
        self._add_polynomial_degree()
        self._add_cfl_number()
        self._add_label()


SOLVER_PARSER = {"cg": CGParser(), "cg_low": LowCGParser(), "mcl": MCLParser()}
