from .analyze_data import AnalazyDataParser, AnalyzeData
from .animate import Animate, AnimateParser
from .calculate import (
    SCALAR_SOLVER_PARSER,
    SHALLOW_WATER_SOLVER_PARSER,
    Calculate,
    CalculateParser,
)
from .command import Command, CommandParser
from .error import CalculateEOC, CalculateEOCParser
from .help import Help, HelpParser
from .plot import Plot, PlotParser
from .test import Test, TestParser

PARSER_COMMANDS = {
    "test": TestParser(),
    "help": HelpParser(SCALAR_SOLVER_PARSER | SHALLOW_WATER_SOLVER_PARSER),
    "analyze-data": AnalazyDataParser(),
    "calculate": CalculateParser(),
    "plot": PlotParser(),
    "animate": AnimateParser(),
    "eoc": CalculateEOCParser(),
}
