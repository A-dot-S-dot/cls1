from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from defaults import *

from . import types as parser_type


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
        self._add_quite_argument()

    def _add_quite_argument(self):
        self.add_argument("+q", "++quite", help="suppress output", action="store_true")


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
            type=parser_type.positive_int,
            default=REFINE_NUMBER,
        )
