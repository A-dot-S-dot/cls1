import argparse
from typing import Any, List, Tuple

import core
import matplotlib.pyplot as plt

from .command import Command, CommandParser


class AnalyzeData(Command):
    _data_paths: List[str]
    _histogram: bool

    def __init__(self, data_paths: List[str], histogram=False):
        self._data_paths = data_paths
        self._histogram = histogram

    def execute(self):
        for data_path in self._data_paths:
            data = core.load_data(data_path)
            print(data_path.upper())
            print("-" * len(data_path))
            print(data.describe())
            print("")
            if self._histogram:
                data.hist()

        if self._histogram:
            plt.show()


class AnalazyDataParser(CommandParser):
    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "analyze-data",
            help="Analyze data.",
            description="""Analyze data by printing summary statistics and historgrams.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_arguments(self, parser):
        parser.add_argument("data_path", help="Specify data location.", nargs="+")
        parser.add_argument("--histogram", action="store_true", help="Plot historgram.")

    def postprocess(self, arguments):
        arguments.data_paths = arguments.data_path
        arguments.command = AnalyzeData

        del arguments.data_path
