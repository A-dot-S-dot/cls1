import argparse
from abc import ABC, abstractmethod
from typing import Type

import defaults
import pde_solver.solver as solver

from . import argument


class SolverParser(argparse.ArgumentParser, ABC):
    prog: str
    name: str
    solver: Type[solver.Solver]

    def __init__(self):
        argparse.ArgumentParser.__init__(
            self,
            prog=self.prog,
            description=self.name,
            prefix_chars="+",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
        )

        self._add_arguments()

    @abstractmethod
    def _add_arguments(self):
        ...

    def parse_arguments(self, *arguments) -> argparse.Namespace:
        arguments = self.parse_args(*arguments)
        arguments.solver = self.solver

        return arguments


class CGParser(SolverParser):
    prog = "cg"
    name = "Continuous Galerkin"
    solver = solver.ContinuousGalerkinSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_polynomial_degree(self)
        argument.add_exact_flux(self)
        argument.add_cfl_number(self, defaults.CFL_NUMBER)


class LowCGParser(SolverParser):
    prog = "cg_low"
    name = "Low order Continuous Galerkin"
    solver = solver.LowOrderContinuousGalerkinSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_polynomial_degree(self)
        argument.add_cfl_number(self, defaults.MCL_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_ode_solver(self)


class MCLParser(SolverParser):
    prog = "mcl"
    name = "MCL Solver"
    solver = solver.MCLSolver

    def _add_arguments(self):
        argument.add_name(self, "MCL Solver")
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_polynomial_degree(self)
        argument.add_cfl_number(self, defaults.MCL_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_ode_solver(self)


class GodunovParser(SolverParser):
    prog = "godunov"
    name = "Godunov's finite volume scheme"
    solver = solver.GodunovSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.GODUNOV_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)


class ReducedExactSolverParser(SolverParser):
    prog = "reduced-exact"
    name = "Reduced Exact Solved (Godunov)"
    solver = solver.ReducedExactSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(
            self, defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
        )
        argument.add_cfl_number(self, defaults.GODUNOV_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_coarsening_degree(self)


class ReducedNetworkParser(SolverParser):
    prog = "reduced-network"
    name = """Reduced Solver with Neural Network (Godunov)"""
    solver = solver.ReducedNetworkSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(
            self, defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
        )
        argument.add_coarsening_degree(self)
        argument.add_cfl_number(self, defaults.GODUNOV_CFL_NUMBER)
        argument.add_network_load_path(self)


SCALAR_SOLVER_PARSERS = {
    "cg": CGParser,
    "cg_low": LowCGParser,
    "mcl": MCLParser,
}
SHALLOW_WATER_SOLVER_PARSERS = {
    "godunov": GodunovParser,
    "reduced-exact": ReducedExactSolverParser,
    "reduced-network": ReducedNetworkParser,
}

SOLVER_PARSERS = {}
SOLVER_PARSERS.update(SCALAR_SOLVER_PARSERS)
SOLVER_PARSERS.update(SHALLOW_WATER_SOLVER_PARSERS)
