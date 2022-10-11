from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from benchmark import Benchmark
from factory import BENCHMARK_FACTORY


class BenchmarkParser:
    _problem: str
    _description: str
    _prog: str

    def __init__(self):
        self._build_parser()
        self._add_benchmarks()

    def _build_parser(self):
        self._parser = ArgumentParser(
            prog=self._problem,
            description=self._description,
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            # add_help=False,
        )

    def _add_benchmarks(self):
        benchmark_parsers = self._parser.add_subparsers(
            title="Benchmarks",
            dest="benchmark",
            metavar="BENCHMARK",
        )
        for i, benchmark in enumerate(BENCHMARK_FACTORY._benchmark[self._problem]):
            self._add_benchmark(benchmark_parsers, i, benchmark)

    def _add_benchmark(self, parsers, benchmark_index: int, benchmark: Benchmark):
        parsers.add_parser(
            str(benchmark_index),
            help=benchmark.name,
            description=f"{benchmark.description} ({benchmark.short_facts})",
            prefix_chars="+",
        )

    def parse_args(self, *args) -> Namespace:
        return self._parser.parse_args(*args)

    def print_help(self):
        self._parser.print_help()


class AdvectionBenchmarkParser(BenchmarkParser):
    _problem = "advection"
    _description = "Benchmarks for Linear Advection."


class BurgersBenchmarkParser(BenchmarkParser):
    _problem = "burgers"
    _description = "Benchmarks for Burgers."


class SWEBenchmarkParser(BenchmarkParser):
    _problem = "swe"
    _description = "Benchmarks for Linear Advection."


class BenchmarkParsers(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(
            self,
            prog="benchmark",
            description="All available Benchmarks.",
            prefix_chars="+",
            formatter_class=ArgumentDefaultsHelpFormatter,
            add_help=False,
        )
        self.add_argument(
            "problem",
            help="Available problems",
            choices=[*BENCHMARK_FACTORY._benchmark.keys()],
        )


BENCHMARK_PARSERS = {
    "advection": AdvectionBenchmarkParser(),
    "burgers": BurgersBenchmarkParser(),
    "swe": SWEBenchmarkParser(),
}
