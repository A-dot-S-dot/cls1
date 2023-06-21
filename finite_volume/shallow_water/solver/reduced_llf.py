import defaults
from finite_volume import shallow_water as swe
from torch import nn

from .lax_friedrichs import LaxFriedrichsFluxGetter
from .reduced_model import ReducedFluxGetter, ReducedSolverParser, ReducedNetwork


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


class LaxFriedrichsNetwork(ReducedNetwork):
    module_type = LaxFriedrichsModule
    data_path = "data/reduced-llf/data.csv"
    network_path = "data/reduced-llf/model.pkl"
    optimizer_path = "data/reduced-llf/opt.pkl"
    history_path = "data/reduced-llf/history.json"
    input_scaler_path = "data/reduced-llf/input_scaler.pkl"
    output_scaler_path = "data/reduced-llf/output_scaler.pkl"


class ReducedLaxFriedrichsSolver(swe.Solver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self.flux_getter = ReducedFluxGetter(
            4,
            LaxFriedrichsNetwork(),
            flux_getter=LaxFriedrichsFluxGetter(),
        )
        super().__init__(benchmark, **kwargs)


class ReducedLaxFriedrichsSolverParser(ReducedSolverParser):
    prog = "reduced-llf"
    name = "Reduced Lax Friedrichs Solver"
    solver = ReducedLaxFriedrichsSolver
    _mesh_size_default = 50
    _cfl_default = defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE
