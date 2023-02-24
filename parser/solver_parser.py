import argparse
from abc import ABC, abstractmethod
from typing import Type

import defaults
from core.solver import Solver
from finite_element.scalar import solver as fem_scalar
from finite_volume.scalar import solver as fv_scalar
from finite_volume.shallow_water import solver as fv_swe

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
    solver = fem_scalar.ContinuousGalerkinSolver

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
    solver = fem_scalar.LowOrderContinuousGalerkinSolver

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
    solver = fem_scalar.MCLSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_polynomial_degree(self)
        argument.add_cfl_number(self, defaults.MCL_CFL_NUMBER)
        argument.add_adaptive_time_stepping(self)
        argument.add_ode_solver(self)


class ScalarLaxFriedrichsParser(SolverParser):
    prog = "llf"
    name = "Lax-Friedrichs finite volume scheme"
    solver = fv_scalar.LaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)


class ShallowWaterLaxFriedrichsParser(SolverParser):
    prog = "llf"
    name = "Lax-Friedrichs finite volume scheme"
    solver = fv_swe.LaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)


class CentralFluxParser(SolverParser):
    prog = "central"
    name = "Central scheme"
    solver = fv_swe.CentralFluxSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)


class LowOrderParser(SolverParser):
    prog = "low-order"
    name = "Low order finite volume scheme"
    solver = fv_swe.LowOrderSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)


class EnergyStableParser(SolverParser):
    prog = "es"
    name = "Energy stable finite volume scheme"
    solver = fv_swe.EnergyStableSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_ode_solver(self)


class FirstOrderDiffusiveEnergyStableParser(SolverParser):
    prog = "es1"
    name = "Energy stable finite volume scheme with first order diffusion"
    solver = fv_swe.FirstOrderDiffusiveEnergyStableSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_ode_solver(self)


# class SubgridNetworkParser(SolverParser):
#     prog = "subgrid-network"
#     name = """Solver with NN corrected flux"""
#     solver = shallow_water.SubgridNetworkSolver

#     def _add_arguments(self):
#         argument.add_name(self, self.name)
#         argument.add_short(self, self.prog)
#         argument.add_mesh_size(
#             self, defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
#         )
#         argument.add_coarsening_degree(self)
#         argument.add_cfl_number(
#             self, defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE
#         )
#         argument.add_network_load_path(self)


class ShallowWaterMCLParser(SolverParser):
    prog = "mcl"
    name = "MCL Solver"
    solver = fv_swe.MCLSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_ode_solver(self)
        argument.add_flux_getter(self)


class AntidiffusionParser(SolverParser):
    prog = "antidiffusion"
    name = "Solver with antidiffusion."
    solver = fv_swe.LinearAntidiffusiveSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_ode_solver(self)
        argument.add_flux_getter(self)


class CoarseParser(SolverParser):
    prog = "coarse"
    name = "Coarsened Solver."
    solver = fv_swe.CoarseSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_ode_solver(self)
        argument.add_flux_getter(self)


SCALAR_SOLVER_PARSERS = {
    "llf": ScalarLaxFriedrichsParser,
    "cg": CGParser,
    "cg_low": LowCGParser,
    "mcl": MCLParser,
}
SHALLOW_WATER_SOLVER_PARSERS = {
    "llf": ShallowWaterLaxFriedrichsParser,
    "low-order": LowOrderParser,
    "central": CentralFluxParser,
    "es": EnergyStableParser,
    "es1": FirstOrderDiffusiveEnergyStableParser,
    # "subgrid-network": SubgridNetworkParser,
    "antidiffusion": AntidiffusionParser,
    "coarse": CoarseParser,
    "mcl": ShallowWaterMCLParser,
}

SOLVER_PARSERS = {}
SOLVER_PARSERS.update(SCALAR_SOLVER_PARSERS)
SOLVER_PARSERS.update(SHALLOW_WATER_SOLVER_PARSERS)
