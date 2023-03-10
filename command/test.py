import subprocess
from typing import Any, List

from .command import Command, CommandParser


class Test(Command):
    _file: List[str]

    def __init__(self, file=None):
        self._file = file or []

    def execute(self):
        subprocess.call(["test/test"] + self._file)


class TestParser(CommandParser):
    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "test",
            help="Run unit test.",
            description="Task for running unit tests. If no argument is given run all tests.",
        )

    def _add_arguments(self, parser):
        parser.add_argument(
            "file",
            nargs="*",
            help="Run unittest contained in FILE_test.py.",
            metavar="<file>",
        )

    def postprocess(self, arguments):
        arguments.command = Test
