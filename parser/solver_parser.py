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
        self.add_argument(
            "+p",
            "++polynomial-degree",
            help="polynomial degree used for finite elements",
            metavar="degree",
            type=parser_type.positive_int,
            default=POLYNOMIAL_DEGREE,
        )
        self.add_argument(
            "++exact-flux", action="store_true", help="calculate flux matrices exactly"
        )
        self.add_argument("++label", type=str, help="label for ploting")


SOLVER_PARSER = {"cg": CGParser()}
