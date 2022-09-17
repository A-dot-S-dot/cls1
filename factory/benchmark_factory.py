from benchmark import Benchmark
from abc import abstractmethod, ABC
from benchmark.burgers import *
from benchmark.advection import *


class BenchmarkFactory(ABC):
    @property
    @abstractmethod
    def benchmark(self) -> Benchmark:
        ...


class AdvectionBenchmarkFactory(BenchmarkFactory):
    benchmark_name: str

    @property
    def benchmark(self) -> Benchmark:
        if self.benchmark_name in ["rect", "plot_default"]:
            return AdvectionPlotBenchmark1()
        elif self.benchmark_name in ["cos", "eoc_default"]:
            return AdvectionEOCBenchmark1()
        elif self.benchmark_name in ["gauss"]:
            return AdvectionEOCBenchmark2()
        else:
            raise ValueError(
                f"no advection benchmark for {self.benchmark_name} defined"
            )


class BurgersBenchmarkFactory(BenchmarkFactory):
    benchmark_name: str

    @property
    def benchmark(self) -> Benchmark:
        if self.benchmark_name in ["sin", "plot_default"]:
            return BurgersBenchmark()
        else:
            raise ValueError(f"no burgers benchmark for {self.benchmark_name} defined")
