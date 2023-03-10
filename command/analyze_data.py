import argparse
from typing import Any

import core
import matplotlib.pyplot as plt
import pandas as pd

from .command import Command, CommandParser


class AnalyzeData(Command):
    _data: pd.DataFrame
    _histogram: bool

    def __init__(self, data_path: str, histogram=False):
        self._data = core.load_data(data_path)
        self._histogram = histogram

    def execute(self):
        print(self._data.describe())
        if self._histogram:
            self._data.hist()
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
        parser.add_argument("data_path", help="Specify data location.")
        parser.add_argument("--histogram", action="store_true", help="Plot historgram.")

    def postprocess(self, arguments):
        arguments.command = AnalyzeData
