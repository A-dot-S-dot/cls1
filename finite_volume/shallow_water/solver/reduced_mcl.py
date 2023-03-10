import numpy as np
from finite_volume import shallow_water as swe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler
from torch import nn
from torch.optim.lr_scheduler import StepLR

from .mcl import MCLFluxGetter
from .reduced_model import ReducedSolver, ReducedSolverParser


class Curvature:
    step_length = 2.0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        u0 = X[:, :2]
        u1 = X[:, 2:4]
        u2 = X[:, 4:6]
        u3 = X[:, 6:]

        curvature = (
            self._calculate_curvature(u0, u1, u2)
            + self._calculate_curvature(u1, u2, u3)
        ) / 2

        return np.concatenate((X, curvature), axis=1)

    def _calculate_curvature(self, u0, u1, u2):
        return (
            abs(u0 - 2 * u1 + u2)
            * self.step_length
            / (self.step_length**2 + 0.25 * (u0 - u2) ** 2) ** (3 / 2)
        )


class MCLModule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class MCLEstimator(Pipeline):
    def __init__(self, **kwargs):
        callbacks = [
            ("early_stopping", EarlyStopping(threshold=1e-8, patience=10)),
            # ("lr_scheduler", LRScheduler(policy=StepLR, step_size=50)),
        ]
        self._network = NeuralNetRegressor(
            MCLModule,
            # optimizer=Adam,
            lr=0.1,
            callbacks=callbacks,
            callbacks__print_log__floatfmt=".8f",
            **kwargs
        )
        super().__init__([("scale", StandardScaler()), ("net", self._network)])

    @property
    def history(self):
        return self._network.history


class ReducedMCLSolver(ReducedSolver):
    def __init__(
        self, benchmark: swe.ShallowWaterBenchmark, network_file_name="model", **kwargs
    ):
        super().__init__(
            benchmark,
            flux_getter=MCLFluxGetter(),
            network_path="data/reduced-mcl/" + network_file_name + ".pkl",
            **kwargs
        )


class ReducedMCLSolverParser(ReducedSolverParser):
    prog = "reduced-mcl"
    name = "Reduced MCL Solver"
    solver = ReducedMCLSolver
