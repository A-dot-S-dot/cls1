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
        self._add_save_argument()

    def _add_initial_data_argument(self):
        self.add_argument("++initial", help="plot initial data", action="store_true")

    def _add_save_argument(self):
        self.add_argument(
            "++save",
            help="save file in specified direction",
            nargs="?",
            const=PLOT_TARGET,
            metavar="FILE",
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
        self._add_initial_data_argument()
        self._add_interval_argument()
        self._add_start_time_argument()
        self._add_save_argument()
        self._add_frame_factor_argument()

    def _add_initial_data_argument(self):
        self.add_argument("++initial", help="plot initial data", action="store_true")

    def _add_interval_argument(self):
        self.add_argument(
            "++interval",
            help="interval between frames in milli seconds.",
            type=custom_type.positive_int,
            default=INTERVAL,
        )

    def _add_start_time_argument(self):
        self.add_argument(
            "+t",
            "++start-time",
            help="set start time for animation",
            type=custom_type.positive_float,
            default=0,
        )

    def _add_save_argument(self):
        self.add_argument(
            "++save",
            help="save file in specified direction",
            nargs="?",
            const=ANIMATION_TARGET,
            metavar="FILE",
        )

    def _add_frame_factor_argument(self):
        self.add_argument(
            "++frame_factor",
            help="""Specifies how many second one time unit should last. Has
            only effect on save files, i.e. it can be only used with combination
            of '++save' option.""",
            type=custom_type.positive_float,
            default=FRAME_FACTOR,
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


class SaveCoarseSolutionAndSubgridFluxesParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog="save",
            description="Save coarse solution and subgrid fluxes.",
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )

        self._add_arguments()

    def _add_arguments(self):
        self._add_coarsening_degree_argument()
        self._add_local_degree_argument()

    def _add_coarsening_degree_argument(self):
        self.add_argument(
            "+d",
            "++coarsening-degree",
            help="degree of coarsening",
            metavar="DEGREE",
            type=custom_type.positive_int,
            default=COARSENING_DEGREE,
        )

    def _add_local_degree_argument(self):
        self.add_argument(
            "+l",
            "++local-degree",
            help="Indicates how many neighboured values on each side of one value should be saved.",
            metavar="DEGREE",
            type=custom_type.non_negative_int,
            default=LOCAL_DEGREE,
        )
