from abc import ABC, abstractmethod
from typing import Any


class Command(ABC):
    @abstractmethod
    def execute(self):
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class CommandParser(ABC):
    def add_parser(self, parsers):
        parser = self._get_parser(parsers)
        self._add_arguments(parser)
        self._add_general_arguments(parser)

    @abstractmethod
    def _get_parser(self, parsers) -> Any:
        ...

    @abstractmethod
    def _add_arguments(self, parser):
        ...

    def _add_general_arguments(self, parser):
        parser.add_argument(
            "--profile",
            help="Profile program for optimization purposes. You can specify the number of lines to be printed. Otherwise print %(const)s lines.",
            type=int,
            nargs="?",
            const=30,
            default=0,
            metavar="<lines>",
        )
        parser.add_argument(
            "--args", action="store_true", help="Print given arguments."
        )

    @abstractmethod
    def postprocess(self, arguments):
        ...
