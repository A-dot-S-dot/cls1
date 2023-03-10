import argparse
import pickle
from typing import Any

import core
import defaults
import matplotlib.pyplot as plt
import numpy as np
import torch
from finite_volume.shallow_water.solver import ESTIMATOR_TYPES

from .command import Command, CommandParser
from .generate_data import DIRECTORIES


class TrainNetwork(Command):
    _data_path: str
    _network_path: str
    _skip: int
    _plot_loss: bool
    _estimator: ...
    _pipeline: ...

    def __init__(
        self,
        data_path: str,
        network_path: str,
        estimator_type=None,
        epochs=None,
        skip=None,
        plot_loss=False,
    ):
        self._network_path = network_path
        self._data_path = data_path
        self._skip = skip or defaults.SKIP
        self._plot_loss = plot_loss

        estimator_type = estimator_type or ESTIMATOR_TYPES["llf"]
        self._estimator = estimator_type(max_epochs=epochs)

    def _get_device(self) -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"INFO: Using {device} device.")

        return device

    def execute(self):
        df = core.load_data(self._data_path)
        X = df.values[:: self._skip, :8].astype(np.float32)
        y = df.values[:: self._skip, 8:].astype(np.float32)

        self._estimator.fit(X, y)
        self._save_model()

        if self._plot_loss:
            self._plot_losses()

    def _save_model(self):
        with open(self._network_path, "wb") as f:
            pickle.dump(self._estimator, f)

    def _plot_losses(self):
        training_loss = self._estimator.history[:, "train_loss"]
        validation_loss = self._estimator.history[:, "valid_loss"]
        plt.close()

        plt.plot(
            np.arange(len(self._estimator.history)),
            np.log10(training_loss),
            label="training loss",
        )
        plt.plot(
            np.arange(len(self._estimator.history)),
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
        self._add_plot_loss(parser)

    def _add_network(self, parser):
        parser.add_argument(
            "network",
            help="Specify which network should be trained.",
            choices=ESTIMATOR_TYPES.keys(),
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
            "-s",
            "--skip",
            help="Use every SKIP-th data point for training.",
            type=core.positive_int,
            metavar="<skip>",
            default=defaults.SKIP,
        )

    def _add_plot_loss(self, parser):
        parser.add_argument(
            "--plot-loss",
            help="Plot training and validation loss.",
            action="store_true",
        )

    def postprocess(self, arguments):
        arguments.estimator_type = ESTIMATOR_TYPES[arguments.network]
        arguments.data_path = DIRECTORIES[arguments.network] + "data.csv"
        arguments.network_path = (
            DIRECTORIES[arguments.network] + arguments.file + ".pkl"
        )
        arguments.command = TrainNetwork

        del arguments.network
        del arguments.file
