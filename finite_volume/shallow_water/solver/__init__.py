from .antidiffusion import *
from .central import *
from .coarse import *
from .energy_stable import *
from .lax_friedrichs import *
from .low_order import *
from .mcl import *
from .reduced_llf import *
from .reduced_mcl import *
from .reduced_model import *

SHALLOW_WATER_FLUX_GETTER = {
    "central": CentralFluxGetter(),
    "es": EnergyStableFluxGetter(),
    "es1": FirstOrderDiffusiveEnergyStableFluxGetter(),
    "llf": LaxFriedrichsFluxGetter(),
    "low-order": LowOrderFluxGetter(),
    "reduced-llf": ReducedFluxGetter(
        LaxFriedrichsFluxGetter(), "data/reduced-llf/model.pkl"
    ),
    "reduced-mcl": ReducedFluxGetter(MCLFluxGetter(), "data/reduced-mcl/model.pkl"),
}

ESTIMATOR_TYPES = {"llf": LaxFriedrichsEstimator, "mcl": MCLEstimator}
SOLVER_PARSER = {
    "llf": LaxFriedrichsParser(),
    "low-order": LowOrderParser(),
    "central": CentralFluxParser(),
    "es": EnergyStableParser(),
    "es1": FirstOrderDiffusiveEnergyStableParser(),
    "mcl": MCLParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
    "reduced-llf": ReducedLaxFriedrichsSolverParser(),
    "reduced-mcl": ReducedMCLSolverParser(),
    "antidiffusion": AntidiffusionParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
    "coarse": CoarseParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
}
