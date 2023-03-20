import defaults
from finite_volume import shallow_water as swe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from .energy_stable import FirstOrderDiffusiveEnergyStableFluxGetter
from .reduced_model import ReducedFluxGetter, ReducedSolverParser


class ES1Module(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class ES1Estimator(Pipeline):
    def __init__(self, **kwargs):
        callbacks = [
            ("early_stopping", EarlyStopping(threshold=1e-8, patience=5)),
            ("lr_scheduler", LRScheduler(MultiStepLR, milestones=[100])),
        ]
        self._network = NeuralNetRegressor(
            ES1Module,
            lr=0.05,
            callbacks=callbacks,
            callbacks__print_log__floatfmt=".8f",
            **kwargs
        )
        super().__init__([("scale", StandardScaler()), ("net", self._network)])

    @property
    def history(self):
        return self._network.history


class ReducedES1Solver(swe.Solver):
    def __init__(
        self, benchmark: swe.ShallowWaterBenchmark, network_file_name="model", **kwargs
    ):
        self.flux_getter = ReducedFluxGetter(
            4,
            flux_getter=FirstOrderDiffusiveEnergyStableFluxGetter(),
            network_path="data/reduced-es1/" + network_file_name + ".pkl",
        )
        super().__init__(benchmark, **kwargs)


class ReducedES1SolverParser(ReducedSolverParser):
    prog = "reduced-es1"
    name = "Reduced Energy Stable Solver"
    solver = ReducedES1Solver
    _mesh_size_default = 50
    _cfl_default = defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE
