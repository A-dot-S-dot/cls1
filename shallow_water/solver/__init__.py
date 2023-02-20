from .antidiffusion import *
from .central import *
from .coarse import *
from .energy_stable import *
from .lax_friedrichs import *
from .low_order import *
from .mcl import *
from .subgrid_network import *

SHALLOW_WATER_FLUX_GETTER = {
    "central": get_central_flux,
    "es": get_energy_stable_flux,
    "llf": get_lax_friedrichs_flux,
    "low-order": get_low_order_flux,
}
