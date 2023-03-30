from .antidiffusion import *
from .central import *
from .coarse import *
from .energy_stable import *
from .lax_friedrichs import *
from .low_order import *
from .mcl import *
from .reduced_es1 import *
from .reduced_llf import *
from .reduced_model import *

SHALLOW_WATER_FLUX_GETTER = {
    "central": CentralFluxGetter(),
    "es": EnergyStableFluxGetter(),
    "es1": FirstOrderDiffusiveEnergyStableFluxGetter(),
    "llf": LaxFriedrichsFluxGetter(),
    "low-order": LowOrderFluxGetter(),
    "reduced-llf": ReducedFluxGetter(
        4,
        LaxFriedrichsNetwork(),
        "data/reduced-llf/model.pkl",
    ),
    ),
    "reduced-es1": ReducedFluxGetter(
        4,
        ES1Network(),
        "data/reduced-es1/model.pkl",
        flux_getter=FirstOrderDiffusiveEnergyStableFluxGetter(),
    ),
}

NETWORK_TYPES = {
    "llf": LaxFriedrichsNetwork,
    "es1": ES1Network,
}
SOLVER_PARSER = {
    "llf": LaxFriedrichsParser(),
    "low-order": LowOrderParser(),
    "central": CentralFluxParser(),
    "es": EnergyStableParser(),
    "es1": FirstOrderDiffusiveEnergyStableParser(),
    "mcl": MCLParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
    "reduced-llf": ReducedLaxFriedrichsSolverParser(),
    "reduced-es1": ReducedES1SolverParser(),
    "antidiffusion": AntidiffusionParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
    "coarse": CoarseParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
}
