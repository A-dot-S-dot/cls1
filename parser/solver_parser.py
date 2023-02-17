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
        argument.add_cfl_number(self, defaults.FINITE_ELEMENT_CFL_NUMBER)


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
    solver = LaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)


class AntidiffusiveLocalLaxFriedrichsParser(SolverParser):
    prog = "antidiffusive-llf"
    name = "Local Lax-Friedrichs finite volume scheme with Antidiffusion"
    solver = AntidiffusiveLocalLaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_antidiffusion_gamma(self)


class CoarseLaxFriedrichsParser(SolverParser):
    prog = "coarse-llf"
    name = "Coarse Local Lax-Friedrichs finite volume scheme"
    solver = CoarseLocalLaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_coarsening_degree(self)


class CentralFluxParser(SolverParser):
    prog = "central"
    name = "Simple high order central scheme"
    solver = CentralFluxSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)


class LowOrderParser(SolverParser):
    prog = "low-order"
    name = "Low order finite volume scheme"
    solver = LowOrderSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)


class CoarseLowOrderParser(SolverParser):
    prog = "coarse-low-order"
    name = "Coarse low order finite volume scheme"
    solver = CoarseLowOrderSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_coarsening_degree(self)


class AntidiffusiveLowOrderParser(SolverParser):
    prog = "antidiffusive-low-order"
    name = "Low order finite volume scheme with Antidiffusion"
    solver = AntidiffusiveLowOrderSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
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
            self, defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE
        )
        argument.add_network_load_path(self)


SCALAR_SOLVER_PARSERS = {
    "cg": CGParser,
    "cg_low": LowCGParser,
    "mcl": MCLParser,
}
SHALLOW_WATER_SOLVER_PARSERS = {
    "llf": LocalLaxFriedrichsParser,
    "coarse-llf": CoarseLaxFriedrichsParser,
    "antidiffusive-llf": AntidiffusiveLocalLaxFriedrichsParser,
    "low-order": LowOrderParser,
    "coarse-low-order": CoarseLowOrderParser,
    "antidiffusive-low-order": AntidiffusiveLowOrderParser,
    "central": CentralFluxParser,
    "subgrid-network": SubgridNetworkParser,
}

SOLVER_PARSERS = {}
SOLVER_PARSERS.update(SCALAR_SOLVER_PARSERS)
SOLVER_PARSERS.update(SHALLOW_WATER_SOLVER_PARSERS)
