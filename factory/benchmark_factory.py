from typing import Optional
from benchmark import Benchmark
from benchmark.advection import *
from benchmark.burgers import *
from benchmark.swe import *

from argparse import Namespace


class BenchmarkFactory:
    problem_name: str
    benchmark_args: Optional[Namespace]
    command: str
    end_time: Optional[float]

    _default_benchmark = {
        "advection": {
            "plot": AdvectionPlot2Benchmark,
            "eoc": AdvectionEOCBenchmark1,
            "calculation": AdvectionPlot2Benchmark,
        },
        "burgers": {
            "plot": BurgersPlotBenchmark,
            "eoc": BurgersEOCBenchmark,
            "calculation": BurgersPlotBenchmark,
        },
        "swe": {
            "plot": SWEBumpSteadyStateBenchmark,
            "calculation": SWEBumpSteadyStateBenchmark,
        },
    }
    _benchmark = {
        "advection": [
            AdvectionPlot1Benchmark,
            AdvectionPlot2Benchmark,
            AdvectionPlot3Benchmark,
            AdvectionEOCBenchmark1,
            AdvectionEOCBenchmark2,
        ],
        "burgers": [BurgersPlotBenchmark, BurgersEOCBenchmark],
        "swe": [SWEBumpSteadyStateBenchmark, SWEWetDryTransitionBenchmark],
    }

    @property
    def benchmark(self) -> Benchmark:
        if self.benchmark_args:
            benchmark_number = int(self.benchmark_args.benchmark)
            benchmark = self._benchmark[self.problem_name][benchmark_number]()
        else:
            benchmark = self._default_benchmark[self.problem_name][self.command]()

        self._set_end_time(benchmark)

        return benchmark

    def _set_end_time(self, benchmark: Benchmark):
        if isinstance(self.end_time, float):
            benchmark.end_time = self.end_time


BENCHMARK_FACTORY = BenchmarkFactory()
