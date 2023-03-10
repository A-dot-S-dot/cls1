from .cg import ContinuousGalerkinSolver, CGParser
from .cg_low import LowOrderContinuousGalerkinSolver, CGLowParser
from .mcl import MCLSolver, MCLParser

SOLVER_PARSER = {"cg": CGParser(), "cg-low": CGLowParser(), "mcl-fem": MCLParser()}
