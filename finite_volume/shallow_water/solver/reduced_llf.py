from finite_volume import shallow_water as swe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from torch import nn

from .lax_friedrichs import LaxFriedrichsFluxGetter
from .reduced_model import ReducedSolver, ReducedSolverParser


class LaxFriedrichsModule(nn.Module):
    input_dimension = 8

    def __init__(self):
        nn.Module.__init__(self)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_dimension, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class LaxFriedrichsEstimator(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(
            [
                ("scale", StandardScaler()),
                (
                    "net",
                    NeuralNetRegressor(
                        LaxFriedrichsModule,
                        callbacks=[EarlyStopping(threshold=1e-8)],
                        callbacks__print_log__floatfmt=".8f",
                        **kwargs,
                    ),
                ),
            ]
        )


class ReducedLaxFriedrichsSolver(ReducedSolver):
    def __init__(
        self, benchmark: swe.ShallowWaterBenchmark, network_file_name="model", **kwargs
    ):
        super().__init__(
            benchmark,
            flux_getter=LaxFriedrichsFluxGetter(),
            network_path="data/reduced-llf/" + network_file_name + ".pkl",
            **kwargs,
        )


class ReducedLaxFriedrichsSolverParser(ReducedSolverParser):
    prog = "reduced-llf"
    name = "Reduced Lax Friedrichs Solver"
    solver = ReducedLaxFriedrichsSolver
