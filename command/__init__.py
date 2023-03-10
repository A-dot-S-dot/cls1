from .analyze_curvature import *
from .analyze_data import *
from .animate import *
from .calculate import *
from .command import *
from .error import *
from .error_evolution import *
from .generate_data import *
from .help import *
from .plot import *
from .test import *
from .train_network import *

PARSER_COMMANDS = {
    "test": TestParser(),
    "help": HelpParser(SCALAR_SOLVER_PARSER | SHALLOW_WATER_SOLVER_PARSER),
    "generate-data": GenerateDataParser(),
    "analyze-data": AnalazyDataParser(),
    "analyze-curvature": AnalyzeCurvatureParser(),
    "train-network": TrainNetworkParser(),
    "calculate": CalculateParser(),
    "plot": PlotParser(),
    "animate": AnimateParser(),
    "eoc": CalculateEOCParser(),
    "error-evolution": PlotErrorEvolutionParser(),
}
