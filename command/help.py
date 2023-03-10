from argparse import ArgumentParser
from typing import Any, Dict, Type

from .command import Command, CommandParser


class Help(Command):
    parser: ArgumentParser

    def __init__(self, parser: Type[ArgumentParser]):
        self.parser = parser()

    def execute(self):
        self.parser.print_help()


class HelpParser(CommandParser):
    _solver_parsers: Dict

    def __init__(self, solver_parsers):
        self._solver_parsers = solver_parsers

    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "help",
            help="Display help messages.",
            description="""Task for displaying help messages for solvers.
            Available solvers are: """
            + ", ".join([*self._solver_parsers.keys()]),
        )

    def _add_arguments(self, parser):
        parser.add_argument(
            "parser",
            help="Specify page which should be displayed in terminal.",
            type=lambda input: self._solver_parsers[input],
            metavar="<page>",
        )

    def postprocess(self, arguments):
        arguments.command = Help
