"""This module provides custom actions for ArgumentParser in parser.py."""
from argparse import Action, ArgumentParser, Namespace
from typing import Dict, List, Optional, Sequence

from .command_parser import EOCParser, PlotParser
from .solver_parser import (
    ADVECTION_SOLVER_PARSERS,
    BURGERS_SOLVER_PARSERS,
    SWE_SOLVER_PARSERS,
)


class PlotAction(Action):
    "Create list of plot arguments."

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: List[str],
        option_string: Optional[str] = ...,
    ) -> None:
        plot_parser = PlotParser()
        arguments = plot_parser.parse_args(values)

        setattr(namespace, "plot", arguments)


class EOCAction(Action):
    "Create list of eoc arguments."

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: List[str],
        option_string: Optional[str] = ...,
    ) -> None:
        eoc_parser = EOCParser()
        arguments = eoc_parser.parse_args(values)

        setattr(namespace, "eoc", arguments)


class SolverAction(Action):
    """Create list of solver arguments"""

    _solver_parsers: Dict

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: List[str],
        option_string: Optional[str] = ...,
    ) -> None:
        solver_arguments = list()
        while len(values) > 0:
            raw_solver_arguments = self._pop_solver_arguments(values)
            solver_arguments.append(self._get_solver_namespace(raw_solver_arguments))

        setattr(namespace, "solver", solver_arguments)

    def _pop_solver_arguments(self, values: List[str]) -> List[str]:
        slice_index = None

        for i, value in enumerate(values):
            if i > 0 and value in self._solver_parsers.keys():
                slice_index = i
                break

        solver_arguments = values[:slice_index]
        del values[:slice_index]

        return solver_arguments

    def _get_solver_namespace(self, raw_solver_arguments: Sequence[str]) -> Namespace:
        solver_key = raw_solver_arguments[0]
        namespace = Namespace(solver=solver_key)
        self._solver_parsers[solver_key].parse_args(raw_solver_arguments[1:], namespace)

        return namespace


class AdvectionSolverAction(SolverAction):
    _solver_parsers = ADVECTION_SOLVER_PARSERS


class BurgersSolverAction(SolverAction):
    _solver_parsers = BURGERS_SOLVER_PARSERS


class SWESolverAction(SolverAction):
    _solver_parsers = SWE_SOLVER_PARSERS
