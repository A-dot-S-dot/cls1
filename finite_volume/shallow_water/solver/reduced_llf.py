import defaults
from finite_volume import shallow_water as swe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from torch import nn

from .lax_friedrichs import LaxFriedrichsFluxGetter
from .reduced_model import ReducedFluxGetter, ReducedSolverParser


class LaxFriedrichsModule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class LaxFriedrichsEstimator(Pipeline):
    def __init__(self, **kwargs):
        self._network = NeuralNetRegressor(
            LaxFriedrichsModule,
            callbacks=[EarlyStopping(threshold=1e-8)],
            callbacks__print_log__floatfmt=".8f",
            **kwargs,
        )

        super().__init__(
            [
                ("scale", StandardScaler()),
                ("net", self._network),
            ]
        )

    @property
    def history(self):
        return self._network.history


class ReducedLaxFriedrichsSolver(swe.Solver):
    def __init__(
        self, benchmark: swe.ShallowWaterBenchmark, network_file_name="model", **kwargs
    ):
        self.flux_getter = ReducedFluxGetter(
            4,
            flux_getter=LaxFriedrichsFluxGetter(),
            network_path="data/reduced-llf/" + network_file_name + ".pkl",
        )
        super().__init__(benchmark, **kwargs)


class ReducedLaxFriedrichsSolverParser(ReducedSolverParser):
    prog = "reduced-llf"
    name = "Reduced Lax Friedrichs Solver"
    solver = ReducedLaxFriedrichsSolver
    _mesh_size_default = 50
    _cfl_default = defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE
