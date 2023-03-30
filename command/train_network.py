import argparse
import pickle
import random
from typing import Any, Optional, Type

import core
import defaults
import matplotlib.pyplot as plt
import numpy as np
import torch
from finite_volume.shallow_water.solver import NETWORK_TYPES, ReducedNetwork

from .command import Command, CommandParser
from .generate_data import DIRECTORIES


class TrainNetwork(Command):
    _data_path: str
    _network_path: str
    _skip: int
    _plot_loss: bool
    _network: ...
    _pipeline: ...

    def __init__(
        self,
        data_path: str,
        network_path: str,
        network_type: Type[ReducedNetwork],
        epochs=None,
        skip=None,
        seed=None,
        plot_loss=False,
        resume_training=False,
    ):
        self._network_path = network_path
        self._data_path = data_path
        self._skip = skip or defaults.SKIP
        self._plot_loss = plot_loss
        self._network = network_type(max_epochs=epochs)

        if resume_training:
            self._load_parameters()
        if seed is not None:
            self._set_seed(seed)

    def _load_parameters(self):
        self._network.initialize()
        self._network.load_params(f_params=self._network_path)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _get_device(self) -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"INFO: Using {device} device.")

        return device

    def execute(self):
        df = core.load_data(self._data_path)
        input_dimension = df.shape[1] - 2
        X = df.values[:: self._skip, :input_dimension].astype(np.float32)
        y = df.values[:: self._skip, input_dimension:].astype(np.float32)

        self._network.fit(X, y)
        self._network.save_params(f_params=self._network_path)

        if self._plot_loss:
            self._plot_losses()

    def _plot_losses(self):
        training_loss = self._network.history[:, "train_loss"]
        validation_loss = self._network.history[:, "valid_loss"]
        plt.close()

        plt.plot(
            np.arange(len(self._network.history)),
            np.log10(training_loss),
            label="training loss",
        )
        plt.plot(
            np.arange(len(self._network.history)),
            np.log10(validation_loss),
            label="validation loss",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Log(Loss)")
        plt.legend()
        plt.show()


class TrainNetworkParser(CommandParser):
    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "train-network",
            help="Trains networks for reduced models.",
            description="""Train networks for reduced shallow water models.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_arguments(self, parser):
        self._add_network(parser)
        self._add_network_file_name(parser)
        self._add_epochs(parser)
        self._add_skip(parser)
        self._add_seed(parser)
        self._add_plot_loss(parser)
        self._add_resume_training(parser)

    def _add_network(self, parser):
        parser.add_argument(
            "network",
            help="Specify which network should be trained.",
            choices=NETWORK_TYPES.keys(),
        )

    def _add_network_file_name(self, parser):
        parser.add_argument(
            "-f",
            "--file",
            help="Specify network file name (file-ending is generated automatically).",
            metavar="<name>",
            default="model",
        )

    def _add_epochs(self, parser):
        parser.add_argument(
            "-e",
            "--epochs",
            help="Determines the maximum number of epochs.",
            type=core.positive_int,
            metavar="<number>",
            default=defaults.EPOCHS,
        )

    def _add_skip(self, parser):
        parser.add_argument(
            "--skip",
            help="Use every SKIP-th data point for training.",
            type=core.positive_int,
            metavar="<skip>",
            default=defaults.SKIP,
        )

    def _add_seed(self, parser):
        parser.add_argument(
            "--seed",
            help="Seed for generating random benchmarks",
            type=core.positive_int,
            metavar="<seed>",
        )

    def _add_plot_loss(self, parser):
        parser.add_argument(
            "--plot-loss",
            help="Plot training and validation loss.",
            action="store_true",
        )

    def _add_resume_training(self, parser):
        parser.add_argument(
            "--resume",
            help="Resume training.",
            action="store_true",
            dest="resume_training",
        )

    def postprocess(self, arguments):
        arguments.network_type = NETWORK_TYPES[arguments.network]
        arguments.data_path = DIRECTORIES[arguments.network] + "data.csv"
        arguments.network_path = (
            DIRECTORIES[arguments.network] + arguments.file + ".pkl"
        )
        arguments.command = TrainNetwork

        del arguments.network
        del arguments.file
