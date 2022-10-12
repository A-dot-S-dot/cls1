from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import custom_type
from defaults import *


class PlotParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog="plot",
            description="Plot benchmarks and computed solutions.",
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )

        self._add_arguments()

    def _add_arguments(self):
        self._add_initial_data_argument()

    def _add_initial_data_argument(self):
        self.add_argument("++initial", help="plot initial data", action="store_true")


class EOCParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog="eoc",
            description="Compute experimental order of convergence (EOC).",
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )

        self._add_arguments()

    def _add_arguments(self):
        self._add_refine_argument()

    def _add_refine_argument(self):
        self.add_argument(
            "+r",
            "++refine",
            help="specify number of refinements",
            type=custom_type.positive_int,
            default=REFINE_NUMBER,
        )


class CalculationParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog="calc",
            description="Calculate solutions without doing anything with them.",
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )

        self._add_arguments()

    def _add_arguments(self):
        ...
