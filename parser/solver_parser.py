from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from abc import ABC, abstractmethod

from defaults import *

from . import types as parser_type


class SolverParser(ArgumentParser, ABC):
    prog: str
    description: str

    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog=self.prog,
            description=self.description,
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )

        self._add_arguments()

    @abstractmethod
    def _add_arguments(self):
        ...


class CGParser(SolverParser):
    prog = "cg"
    description = "Continuous Galerkin Solver"

    def _add_arguments(self):
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
    prog = "cg_low"
    description = "Low order Continuous Galerkin Solver"

    def _add_arguments(self):
        self._add_polynomial_degree()
        self._add_cfl_number()
        self._add_label()
        self._add_ode_solver()

    def _add_cfl_number(self):
        self.add_argument(
            "++cfl",
            help="specify the cfl number for time stepping",
            type=parser_type.positive_float,
            metavar="number",
            dest="cfl_number",
            default=MCL_CFL_NUMBER,
        )

    def _add_ode_solver(self):
        self.add_argument(
            "++ode",
            help="specify ode solver",
            choices={"euler", "heun", "ssp3", "ssp4"},
            metavar="solver",
            dest="ode_solver",
            default=ODE_SOLVER,
        )


class MCLParser(LowCGParser):
    prog = "mcl"
    description = "MCL Limiter"


SOLVER_PARSER = {"cg": CGParser(), "cg_low": LowCGParser(), "mcl": MCLParser()}
