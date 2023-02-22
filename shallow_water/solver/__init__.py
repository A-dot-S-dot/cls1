from .antidiffusion import *
from .central import *
from .coarse import *
from .energy_stable import *
from .lax_friedrichs import *
from .low_order import *
from .mcl import *
from .subgrid_network import *

SHALLOW_WATER_FLUX_GETTER = {
    "central": CentralFluxGetter(),
    "es": EnergyStableFluxGetter(),
    "llf": LaxFriedrichsFluxGetter(),
    "low-order": LowOrderFluxGetter(),
}
