from .antidiffusion import *
from .central import *
from .coarse import *
from .energy_stable import *
from .lax_friedrichs import *
from .low_order import *
from .mcl import *
from .reduced_es1 import *
from .reduced_llf import *
from .reduced_llf_2 import *
from .reduced_model import *
from .sde_mcl import *

SHALLOW_WATER_FLUX_GETTER = {
    "central": CentralFluxGetter(),
    "es": EnergyStableFluxGetter(),
    "es1": FirstOrderDiffusiveEnergyStableFluxGetter(),
    "llf": LaxFriedrichsFluxGetter(),
    "low-order": LowOrderFluxGetter(),
    "reduced-llf": ReducedFluxGetter(4, LaxFriedrichsNetwork()),
    "reduced-llf-2": ReducedFluxGetter(4, LaxFriedrichsNetwork()),
    "reduced-es1": ReducedFluxGetter(
        4, ES1Network(), flux_getter=FirstOrderDiffusiveEnergyStableFluxGetter()
    ),
}

NETWORK_TYPES = {
    "llf": LaxFriedrichsNetwork,
    "llf2": LaxFriedrichs2Network,
    "es1": ES1Network,
}
SOLVER_PARSER = {
    "llf": LaxFriedrichsParser(),
    "low-order": LowOrderParser(),
    "central": CentralFluxParser(),
    "es": EnergyStableParser(),
    "es1": FirstOrderDiffusiveEnergyStableParser(),
    "mcl": MCLParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
    "sde-mcl": EntropyFixedMCLParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
    "reduced-llf": ReducedLaxFriedrichsSolverParser(),
    "reduced-llf-2": ReducedLaxFriedrichs2SolverParser(),
    "reduced-es1": ReducedES1SolverParser(),
    "antidiffusion": AntidiffusionParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
    "coarse": CoarseParser(flux_getter=SHALLOW_WATER_FLUX_GETTER),
}
