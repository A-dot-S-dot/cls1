from .command import Command
import core
import pandas as pd
import matplotlib.pyplot as plt


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
