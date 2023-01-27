"""This module takes arguments from argparse and processes them."""
from .animate import Animate, ScalarAnimator, ShallowWaterAnimator
from .calculate import Calculate
from .command import Command
from .error import (
    CalculateEOC,  # GenerateShallowWaterErrorEvolutionSeries,
    ErrorCalculator,
    ErrorEvolutionCalculator,
    PlotShallowWaterAverageErrorEvolution,
    PlotShallowWaterErrorEvolution,
)
from .generate_data import GenerateData
from .help import Help
from .plot import Plot, ScalarPlotter, ShallowWaterPlotter
from .test import Test

# from .train_network import TrainNetwork
