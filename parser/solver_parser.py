import argparse
from abc import ABC, abstractmethod
from typing import Type

import defaults
from core.solver import Solver
from scalar.solver import *
from shallow_water.solver import *

from . import argument


class SolverParser(argparse.ArgumentParser, ABC):
    prog: str
    name: str
    solver: Type[Solver]

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
    solver = ContinuousGalerkinSolver

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
    solver = LowOrderContinuousGalerkinSolver

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
    solver = MCLSolver

    def _add_arguments(self):
        argument.add_name(self, "MCL Solver")
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_polynomial_degree(self)
        argument.add_cfl_number(self, defaults.MCL_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_ode_solver(self)


class LocalLaxFriedrichsParser(SolverParser):
    prog = "llf"
    name = "Local Lax-Friedrichs finite volume scheme"
    solver = LocalLaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.GODUNOV_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)


class AntidiffusiveLocalLaxFriedrichsParser(SolverParser):
    prog = "antidiffusive-llf"
    name = "Local Lax-Friedrichs finite volume scheme with Antidiffusion"
    solver = AntidiffusiveLocalLaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.GODUNOV_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_antidiffusion_gamma(self)


class CoarseLocalLaxFriedrichsParser(SolverParser):
    prog = "coarse-llf"
    name = "Coarse Local Lax-Friedrichs finite volume scheme"
    solver = CoarseLocalLaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.GODUNOV_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_coarsening_degree(self)


class GodunovParser(SolverParser):
    prog = "godunov"
    name = "Godunov's finite volume scheme"
    solver = GodunovSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.GODUNOV_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)


class CoarseGodunovParser(SolverParser):
    prog = "coarse-godunov"
    name = "Coarse Godunov's finite volume scheme"
    solver = CoarseGodunovSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.GODUNOV_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_coarsening_degree(self)


class AntidiffusiveGodunovParser(SolverParser):
    prog = "antidiffusive-godunov"
    name = "Godunov's finite volume scheme with Antidiffusion"
    solver = AntidiffusiveGodunovSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.GODUNOV_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_antidiffusion_gamma(self)


class SubgridNetworkParser(SolverParser):
    prog = "subgrid-network"
    name = """Solver with NN corrected flux"""
    solver = SubgridNetworkSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(
            self, defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
        )
        argument.add_coarsening_degree(self)
        argument.add_cfl_number(
            self, defaults.GODUNOV_CFL_NUMBER / defaults.COARSENING_DEGREE
        )
        argument.add_network_load_path(self)


class LimitedSubgridNetworkParser(SolverParser):
    prog = "limited-subgrid-network"
    name = """Limited Solver with NN corrected flux"""
    solver = LimitedSubgridNetworkSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(
            self, defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
        )
        argument.add_coarsening_degree(self)
        argument.add_cfl_number(
            self, defaults.GODUNOV_CFL_NUMBER / defaults.COARSENING_DEGREE
        )
        argument.add_network_load_path(self)
        argument.add_limiting_gamma(self)


SCALAR_SOLVER_PARSERS = {
    "cg": CGParser,
    "cg_low": LowCGParser,
    "mcl": MCLParser,
}
SHALLOW_WATER_SOLVER_PARSERS = {
    "llf": LocalLaxFriedrichsParser,
    "coarse-llf": CoarseLocalLaxFriedrichsParser,
    "antidiffusive-llf": AntidiffusiveLocalLaxFriedrichsParser,
    "godunov": GodunovParser,
    "coarse-godunov": CoarseGodunovParser,
    "antidiffusive-godunov": AntidiffusiveGodunovParser,
    "subgrid-network": SubgridNetworkParser,
    "limited-subgrid-network": LimitedSubgridNetworkParser,
}

SOLVER_PARSERS = {}
SOLVER_PARSERS.update(SCALAR_SOLVER_PARSERS)
SOLVER_PARSERS.update(SHALLOW_WATER_SOLVER_PARSERS)
