"""This module provides a task for displaying help messages.

"""
from parser.benchmark_parser import *
from parser.command_parser import *
from parser.parser import AVAILABLE_HELP_ARGUMENTS
from parser.solver_parser import SOLVER_PARSERS

from .command import Command


class HelpCommand(Command):
    benchmarks_parser = BenchmarkParsers()

    def execute(self):
        page = self._args.page

        if page in SOLVER_PARSERS.keys():
            SOLVER_PARSERS[page].print_help()
        elif page == "benchmark":
            self._print_benchmarks()
        elif page == "plot":
            parser = PlotParser()
            parser.print_help()
        elif page == "animate":
            parser = AnimateParser()
            parser.print_help()
        elif page == "eoc":
            parser = EOCParser()
            parser.print_help()
        else:
            raise NotImplementedError(
                f"No help message for {page} available. Available arguments are: {AVAILABLE_HELP_ARGUMENTS}"
            )

        if self._args.option:
            print()
            print(f"WARNING: Options {self._args.option} couldn't be processed.")

    def _print_benchmarks(self):
        if self._args.option:
            BENCHMARK_PARSERS[self._args.option[0]].parse_args(
                [*self._args.option[1:], "+h"]
            )
            self._args.option = None
        else:
            self.benchmarks_parser.print_help()
