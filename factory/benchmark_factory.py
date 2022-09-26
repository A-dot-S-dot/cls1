from benchmark import Benchmark
from benchmark.advection import *
from benchmark.burgers import *


class BenchmarkFactory:
    problem_name: str
    benchmark_name: str
    end_time = None

    @property
    def benchmark(self) -> Benchmark:
        if self.problem_name == "advection":
            benchmark = self._advection_benchmark()
        elif self.problem_name == "burgers":
            benchmark = self._burgers_benchmark()
        else:
            raise NotImplementedError(
                f"no benchmarks for {self.problem_name} availalbe"
            )

        if self.end_time is not None:
            benchmark.end_time = self.end_time

        return benchmark

    def _advection_benchmark(self) -> Benchmark:
        if self.benchmark_name in ["1"]:
            benchmark = AdvectionPlot1Benchmark()
        elif self.benchmark_name in ["2", "plot_default"]:
            benchmark = AdvectionPlot2Benchmark()
        elif self.benchmark_name in ["3"]:
            benchmark = AdvectionPlot3Benchmark()
        elif self.benchmark_name in ["4", "eoc_default"]:
            benchmark = AdvectionEOCBenchmark1()
        elif self.benchmark_name in ["5"]:
            benchmark = AdvectionEOCBenchmark2()
        else:
            raise ValueError(
                f"no advection benchmark for {self.benchmark_name} defined"
            )

        if isinstance(self.end_time, float):
            benchmark.end_time = self.end_time

        return benchmark

    def _burgers_benchmark(self) -> Benchmark:
        if self.benchmark_name in ["1", "plot_default"]:
            benchmark = BurgersPlotBenchmark()
        elif self.benchmark_name in ["2", "eoc_default"]:
            benchmark = BurgersEOCBenchmark()
        else:
            raise ValueError(f"no burgers benchmark for {self.benchmark_name} defined")

        return benchmark


BENCHMARK_FACTORY = BenchmarkFactory()
