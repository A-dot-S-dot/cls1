from .analyze_curvature import *
from .analyze_data import *
from .animate import *
from .calculate import *
from .command import *
from .delete_curvature import *
from .error import *
from .error_evolution import *
from .generate_data import *
from .help import *
from .parameter_variation_test import *
from .plot import *
from .print_network import *
from .test import *
from .train_network import *

PARSER_COMMANDS = {
    "analyze-curvature": AnalyzeCurvatureParser(),
    "analyze-data": AnalazyDataParser(),
    "animate": AnimateParser(),
    "calculate": CalculateParser(),
    "delete-curvature": DeleteCurvatureParser(),
    "eoc": CalculateEOCParser(),
    "generate-data": GenerateDataParser(),
    "help": HelpParser(SCALAR_SOLVER_PARSER | SHALLOW_WATER_SOLVER_PARSER),
    "parameter-variation-test": ParameterVariationTestParser(),
    "plot": PlotParser(),
    "plot-error-evolution": PlotErrorEvolutionParser(),
    "print-network": PrintNetworkParser(),
    "test": TestParser(),
    "train-network": TrainNetworkParser(),
}
