from .antidiffusion import *
from .central import *
from .coarse import *
from .energy_stable import *
from .lax_friedrichs import *
from .low_order import *
from .mcl import *
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
