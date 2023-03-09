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
    prog = "CG"
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
    prog = "CG-Low"
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


class ScalarFiniteElementMCLParser(SolverParser):
    prog = "MCL-FEM"
    name = "Finite Element MCL Solver"
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
    prog = "LLF"
    name = "Lax-Friedrichs finite volume scheme"
    solver = fv_scalar.LaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)


class ShallowWaterLaxFriedrichsParser(SolverParser):
    prog = "LLF"
    name = "Lax-Friedrichs finite volume scheme"
    solver = fv_swe.LaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)


class ScalarCentralFluxParser(SolverParser):
    prog = "Central"
    name = "Central scheme"
    solver = fv_scalar.CentralFluxSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)


class ShallowWaterCentralFluxParser(SolverParser):
    prog = "Central"
    name = "Central scheme"
    solver = fv_swe.CentralFluxSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)


class LowOrderParser(SolverParser):
    prog = "Low-Order"
    name = "Low order finite volume scheme"
    solver = fv_swe.LowOrderSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)


class EnergyStableParser(SolverParser):
    prog = "ES"
    name = "Energy stable finite volume scheme"
    solver = fv_swe.EnergyStableSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_ode_solver(self)


class FirstOrderDiffusiveEnergyStableParser(SolverParser):
    prog = "ES1"
    name = "Energy stable finite volume scheme with first order diffusion"
    solver = fv_swe.FirstOrderDiffusiveEnergyStableSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_ode_solver(self)


class ReducedLaxFriedrichsSolverParser(SolverParser):
    prog = "reduced-llf"
    name = "Reduced Lax Friedrichs Solver"
    solver = fv_swe.ReducedLaxFriedrichsSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(
            self, defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
        )
        argument.add_cfl_number(
            self, defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE
        )
        argument.add_network_name(self)


class ReducedMCLSolverParser(SolverParser):
    prog = "reduced-mcl"
    name = "Reduced MCL Solver"
    solver = fv_swe.ReducedMCLSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(
            self, defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
        )
        argument.add_cfl_number(
            self, defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE
        )
        argument.add_network_name(self)


class ScalarFiniteVolumeMCLParser(SolverParser):
    prog = "MCL-FV"
    name = "Finite Volume MCL Solver"
    solver = fv_scalar.MCLSolver

    def _add_arguments(self):
        argument.add_name(self, self.name)
        argument.add_short(self, self.prog)
        argument.add_mesh_size(self)
        argument.add_cfl_number(self, defaults.FINITE_VOLUME_CFL_NUMBER)
        argument.add_ode_solver(self)


class ShallowWaterMCLParser(SolverParser):
    prog = "MCL"
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
    prog = "Antidiffusion"
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
    prog = "Coarse"
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
    "central": ScalarCentralFluxParser,
    "mcl-fv": ScalarFiniteVolumeMCLParser,
    "cg": CGParser,
    "cg_low": LowCGParser,
    "mcl-fem": ScalarFiniteElementMCLParser,
}
SHALLOW_WATER_SOLVER_PARSERS = {
    "llf": ShallowWaterLaxFriedrichsParser,
    "low-order": LowOrderParser,
    "central": ShallowWaterCentralFluxParser,
    "es": EnergyStableParser,
    "es1": FirstOrderDiffusiveEnergyStableParser,
    "reduced-llf": ReducedLaxFriedrichsSolverParser,
    "reduced-mcl": ReducedMCLSolverParser,
    "antidiffusion": AntidiffusionParser,
    "coarse": CoarseParser,
    "mcl": ShallowWaterMCLParser,
}

SOLVER_PARSERS = {}
SOLVER_PARSERS.update(SCALAR_SOLVER_PARSERS)
SOLVER_PARSERS.update(SHALLOW_WATER_SOLVER_PARSERS)
