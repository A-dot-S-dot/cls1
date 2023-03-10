import argparse
import textwrap
from typing import Dict

from command import CommandParser


class CustomArgumentParser:
    _command_parser: Dict[str, CommandParser]

    def __init__(self, command_parser: Dict[str, CommandParser]):
        self._parser = argparse.ArgumentParser(
            prog="cls1",
            description=textwrap.dedent(
                """\
            Explore different PDE-Solver for one-dimension conservation laws.

            """
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._command_parser = command_parser

        parsers = self._parser.add_subparsers(
            title="Commands",
            dest="command",
            metavar="<command>",
            required=True,
        )

        for parser in command_parser.values():
            parser.add_parser(parsers)

    def parse_arguments(self, *arguments) -> argparse.Namespace:
        arguments = self._parser.parse_args(*arguments)
        self._command_parser[arguments.command].postprocess(arguments)

        return arguments
