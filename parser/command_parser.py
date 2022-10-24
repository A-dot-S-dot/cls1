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
        self._add_initial_data()
        self._add_save()

    def _add_initial_data(self):
        self.add_argument("++initial", help="Plot initial data.", action="store_true")

    def _add_save(self):
        self.add_argument(
            "++save",
            help="Save file in specified direction.",
            nargs="?",
            const=PLOT_TARGET,
            metavar="<file>",
        )


class AnimateParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog="animate",
            description="Animate benchmarks and computed solutions.",
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )

        self._add_arguments()

    def _add_arguments(self):
        self._add_initial_data()
        self._add_interval()
        self._add_start_time()
        self._add_save()
        self._add_frame_factor()

    def _add_initial_data(self):
        self.add_argument("++initial", help="Plot initial data.", action="store_true")

    def _add_interval(self):
        self.add_argument(
            "++interval",
            help="Specify interval between frames in milli seconds.",
            type=custom_type.positive_int,
            metavar="<interval>",
            default=INTERVAL,
        )

    def _add_start_time(self):
        self.add_argument(
            "+t",
            "++start-time",
            help="Set start time for animation.",
            type=custom_type.positive_float,
            metavar="<time>",
            default=0,
        )

    def _add_save(self):
        self.add_argument(
            "++save",
            help="""Save file in specified direction. If no specified show the
            plot without saving it.""",
            nargs="?",
            const=ANIMATION_TARGET,
            metavar="<save>",
        )

    def _add_frame_factor(self):
        self.add_argument(
            "++frame_factor",
            help="""Specifies how many second one time unit should last. Has
            only effect on save files, i.e. it can be only used with combination
            of '++save' option.""",
            type=custom_type.positive_float,
            default=FRAME_FACTOR,
            metavar="<factor>",
        )


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
        self._add_refine()

    def _add_refine(self):
        self.add_argument(
            "+r",
            "++refine",
            help="Specify number of refinements.",
            type=custom_type.positive_int,
            default=REFINE_NUMBER,
            metavar="<number>",
        )


class CalculateParser(ArgumentParser):
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
