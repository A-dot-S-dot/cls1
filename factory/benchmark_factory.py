from benchmark import Benchmark
from abc import abstractmethod, ABC
from benchmark.burgers import *
from benchmark.advection import *


class BenchmarkFactory(ABC):
    benchmark_name: str
    end_time = None

    @property
    @abstractmethod
    def benchmark(self) -> Benchmark:
        ...


class AdvectionBenchmarkFactory(BenchmarkFactory):
    @property
    def benchmark(self) -> Benchmark:
        if self.benchmark_name in ["1", "plot_default"]:
            benchmark = AdvectionPlotBenchmark()
        elif self.benchmark_name in ["2", "eoc_default"]:
            benchmark = AdvectionEOCBenchmark1()
        elif self.benchmark_name in ["3"]:
            benchmark = AdvectionEOCBenchmark2()
        else:
            raise ValueError(
                f"no advection benchmark for {self.benchmark_name} defined"
            )

        if isinstance(self.end_time, float):
            benchmark.end_time = self.end_time

        return benchmark


class BurgersBenchmarkFactory(BenchmarkFactory):
    @property
    def benchmark(self) -> Benchmark:
        if self.benchmark_name in ["1", "plot_default"]:
            benchmark = BurgersPlotBenchmark()
        elif self.benchmark_name in ["2", "eoc_default"]:
            benchmark = BurgersEOCBenchmark()
        else:
            raise ValueError(f"no burgers benchmark for {self.benchmark_name} defined")

        if self.end_time is not None:
            benchmark.end_time = self.end_time

        return benchmark
