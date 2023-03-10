from .central import *
from .lax_friedrichs import *
from .mcl import *

SOLVER_PARSER = {
    "llf": LaxFriedrichsParser(),
    "central": CentralParser(),
    "mcl-fv": MCLParser(),
}
