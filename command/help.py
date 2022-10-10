"""This module provides a task for displaying help messages.

"""
from parser.command_parser import EOCParser, PlotParser
from parser.solver_parser import SOLVER_PARSERS

from benchmark import Benchmark
from factory.benchmark_factory import BENCHMARK_FACTORY

from .command import Command


class HelpCommand(Command):
    def execute(self):
        page = self._args.page

        if page in SOLVER_PARSERS.keys():
            SOLVER_PARSERS[page].print_help()
        elif page == "benchmark":
            self._print_benchmarks()
        elif page == "plot":
            parser = PlotParser()
            parser.print_help()
        elif page == "eoc":
            parser = EOCParser()
            parser.print_help()
        else:
            raise NotImplementedError(f"No help message for {page} available.")

    def _print_benchmarks(self):
        problem_titles = ["Linear Advection", "Burgers", "Shallow-Water Equations"]
        for problem_key, problem_title in zip(
            BENCHMARK_FACTORY._benchmark.keys(), problem_titles
        ):
            self._print_problem_benchmarks(problem_key, problem_title)

    def _print_problem_benchmarks(self, problem_key: str, problem_title: str):
        self._print_description(problem_title)
        benchmarks = BENCHMARK_FACTORY._benchmark[problem_key]

        for benchmark_index, benchmark in enumerate(benchmarks):
            self._print_benchmark_information(benchmark_index, benchmark)
            print()

    def _print_description(self, problem_title: str):
        print(problem_title + "\n" + len(problem_title) * "-")

    def _print_benchmark_information(self, benchmark_number: int, benchmark: Benchmark):
        print_message = f"\t{benchmark_number}) {benchmark.name.upper()} ({benchmark.short_facts})\n\t\t{benchmark.description}"

        print(print_message)
