from typing import Optional, Callable
from benchmark import Benchmark
from benchmark.advection import *
from benchmark.burgers import *


class BenchmarkFactory:
    problem_name: str
    command: str
    benchmark_number: Optional[int]
    end_time: Optional[float]

    _default_benchmark = {
        "advection": {"plot": AdvectionPlot2Benchmark, "eoc": AdvectionEOCBenchmark1},
        "burgers": {"plot": BurgersPlotBenchmark, "eoc": BurgersEOCBenchmark},
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
    }

    @property
    def benchmark(self) -> Benchmark:
        if self.benchmark_number:
            benchmark = self._benchmark[self.problem_name][self.benchmark_number]()
        else:
            benchmark = self._default_benchmark[self.problem_name][self.command]()

        self._set_end_time(benchmark)

        return benchmark

    def _set_end_time(self, benchmark: Benchmark):
        if isinstance(self.end_time, float):
            benchmark.end_time = self.end_time


BENCHMARK_FACTORY = BenchmarkFactory()
