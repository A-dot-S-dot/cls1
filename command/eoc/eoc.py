from argparse import Namespace
from typing import Sequence

from benchmark.abstract import NoExactSolutionError
from command.command import Command
from factory.pde_solver_factory import PDESolverFactory
from pde_solver.error import L1Norm, L2Norm, LInfinityNorm
from pde_solver.solver_components import SolverComponents
from tqdm import tqdm

from .eoc_calculator import *
from .eoc_data_frame import *


class EOCCommand(Command):
    _components: SolverComponents
    _args: Namespace
    _solver_factories: Sequence[PDESolverFactory]
    _eoc_caluclator: EOCCalculator

    def __init__(self, args: Namespace):
        self._args = args

        self._components = SolverComponents(args)
        self._solver_factories = self._components.solver_factories
        self._build_eoc_calculator()

    def _build_eoc_calculator(self):
        self._eoc_calculator = ScalarEOCCalculator()
        self._eoc_calculator.benchmark = self._components.benchmark
        self._eoc_calculator.refine_number = self._args.eoc.refine

    def execute(self):
        if len(self._solver_factories) == 0:
            print("WARNING: Nothing to do...")
            return

        try:
            self._print_eocs()
        except NoExactSolutionError as error:
            print("ERROR: " + str(error))

    def _print_eocs(self):
        eocs = []
        titles = []

        for solver_factory in tqdm(
            self._components.solver_factories,
            desc="Calculate solutions",
            unit="solver",
            leave=False,
        ):
            titles.append(solver_factory.eoc_title)
            self._calculate_eoc(solver_factory)
            eocs.append(EOCDataFrame(self._eoc_calculator))

        for eoc, title in zip(eocs, titles):
            print()
            print(title)
            print(eoc)

    def _calculate_eoc(self, solver_factory: PDESolverFactory):
        l2_norm = L2Norm(solver_factory.cell_quadrature_degree)
        l1_norm = L1Norm(solver_factory.cell_quadrature_degree)
        linf_norm = LInfinityNorm(solver_factory.cell_quadrature_degree + 5)

        self._eoc_calculator.add_norms(l2_norm, l1_norm, linf_norm)
        self._eoc_calculator.solver_factory = solver_factory

        self._eoc_calculator.calculate()
