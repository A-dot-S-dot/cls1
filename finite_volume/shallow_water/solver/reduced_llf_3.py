import defaults
from finite_volume import shallow_water as swe
from torch import nn

from .lax_friedrichs import LaxFriedrichsFluxGetter
from .reduced_model import ReducedFluxGetter, ReducedNetwork, ReducedSolverParser


class LaxFriedrichs3Module(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class LaxFriedrichs3Network(ReducedNetwork):
    module_type = LaxFriedrichs3Module
    data_path = "data/reduced-llf-3/data.csv"
    network_path = "data/reduced-llf-3/model.pkl"
    optimizer_path = "data/reduced-llf-3/opt.pkl"
    history_path = "data/reduced-llf-3/history.json"
    input_scaler_path = "data/reduced-llf-3/input_scaler.pkl"
    output_scaler_path = "data/reduced-llf-3/output_scaler.pkl"


class ReducedLaxFriedrichs3Solver(swe.Solver):
    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        self.flux_getter = ReducedFluxGetter(
            4,
            LaxFriedrichs3Network(),
            flux_getter=LaxFriedrichsFluxGetter(),
        )
        super().__init__(benchmark, **kwargs)


class ReducedLaxFriedrichs3SolverParser(ReducedSolverParser):
    prog = "reduced-llf-3"
    name = "Third Reduced Lax Friedrichs Solver"
    solver = ReducedLaxFriedrichs3Solver
    _mesh_size_default = 50
    _cfl_default = defaults.FINITE_VOLUME_CFL_NUMBER / 40
