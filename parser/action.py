"""This module provides custom actions for ArgumentParser in parser.py."""
import argparse
from typing import Dict, List, Optional, Sequence

from .solver_parser import SCALAR_SOLVER_PARSERS, SHALLOW_WATER_SOLVER_PARSERS


class SolverAction(argparse.Action):
    """Create list of solver arguments"""

    _solver_parsers: Dict

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
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
            if (
                i > 0
                and value in self._solver_parsers.keys()
                and values[i - 1] not in ["+f"]
            ):
                slice_index = i
                break

        solver_arguments = values[:slice_index]
        del values[:slice_index]

        return solver_arguments

    def _get_solver_namespace(
        self, raw_solver_arguments: Sequence[str]
    ) -> argparse.Namespace:
        solver_key = raw_solver_arguments[0]
        namespace = argparse.Namespace()
        try:
            self._solver_parsers[solver_key]().parse_arguments(
                raw_solver_arguments[1:], namespace
            )
        except KeyError:
            print(
                "ERROR: Only the following solvers are available: "
                + ", ".join(self._solver_parsers.keys())
            )
            quit()

        return namespace


class ScalarSolverAction(SolverAction):
    _solver_parsers = SCALAR_SOLVER_PARSERS


class ShallowWaterSolverAction(SolverAction):
    _solver_parsers = SHALLOW_WATER_SOLVER_PARSERS
