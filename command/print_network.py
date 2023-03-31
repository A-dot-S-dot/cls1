from .command import Command, CommandParser
import argparse
from typing import Any
from finite_volume.shallow_water.solver import NETWORK_TYPES, ReducedNetwork


class PrintNetwork(Command):
    def __init__(self, network: ReducedNetwork):
        self._network = network

    def execute(self):
        self._network.load_params()
        print(list(self._network.module_.parameters()))


class PrintNetworkParser(CommandParser):
    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "print-network",
            help="Print networks of reduced models.",
            description="""Print networks of reduced shallow water models.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_arguments(self, parser):
        self._add_network(parser)

    def _add_network(self, parser):
        parser.add_argument(
            "network",
            help="Specify which network should be trained.",
            choices=NETWORK_TYPES.keys(),
        )

    def postprocess(self, arguments):
        arguments.network = NETWORK_TYPES[arguments.network]()
        arguments.command = PrintNetwork
