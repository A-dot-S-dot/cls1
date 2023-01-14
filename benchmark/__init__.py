from .abstract import Benchmark, NoExactSolutionError
from .advection import (
    AdvectionBenchmark,
    AdvectionCosineBenchmark,
    AdvectionGaussianBellBenchmark,
    AdvectionOneHillBenchmark,
    AdvectionThreeHillsBenchmark,
    AdvectionTwoHillsBenchmark,
)
from .burgers import BurgersBenchmark, BurgersSchockBenchmark, BurgersSmoothBenchmark
from .shallow_water import (
    ShallowWaterBenchmark,
    ShallowWaterSteadyStateBenchmark,
    ShallowWaterBumpSteadyStateBenchmark,
    ShallowWaterOscillationNoTopographyBenchmark,
    ShallowWaterRandomOscillationNoTopographyBenchmark,
)
