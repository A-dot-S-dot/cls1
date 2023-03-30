import defaults
from finite_volume import shallow_water as swe
from torch import nn

from .lax_friedrichs import LaxFriedrichsFluxGetter
from .reduced_model import ReducedFluxGetter, ReducedNetwork, ReducedSolverParser


class LaxFriedrichs2Module(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class LaxFriedrichs2Network(ReducedNetwork):
    module = LaxFriedrichs2Module


class ReducedLaxFriedrichs2Solver(swe.Solver):
    def __init__(
        self, benchmark: swe.ShallowWaterBenchmark, network_file_name="model", **kwargs
    ):
        self.flux_getter = ReducedFluxGetter(
            4,
            LaxFriedrichs2Network(),
            "data/reduced-llf-2/" + network_file_name + ".pkl",
            flux_getter=LaxFriedrichsFluxGetter(),
        )
        super().__init__(benchmark, **kwargs)


class ReducedLaxFriedrichs2SolverParser(ReducedSolverParser):
    prog = "reduced-llf-2"
    name = "Second Reduced Lax Friedrichs Solver"
    solver = ReducedLaxFriedrichs2Solver
    _mesh_size_default = 50
    _cfl_default = defaults.FINITE_VOLUME_CFL_NUMBER / 20
