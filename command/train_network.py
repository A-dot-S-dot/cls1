import argparse
import random
from typing import Any, Dict

import core
import defaults
import matplotlib.pyplot as plt
import numpy as np
import torch
from finite_volume.shallow_water.solver import NETWORK_TYPES, ReducedNetwork

from .command import Command, CommandParser


class TrainNetwork(Command):
    _network: ReducedNetwork
    _epochs: int
    _skip: int
    _plot_loss: bool
    _confirm_save: bool

    def __init__(
        self,
        network: ReducedNetwork,
        epochs=None,
        skip=None,
        seed=None,
        plot_loss=False,
        confirm_save=False,
    ):
        self._network = network
        self._epochs = epochs or defaults.EPOCHS
        self._skip = skip or defaults.SKIP
        self._plot_loss = plot_loss
        self._confirm_save = confirm_save

        if seed is not None:
            self._set_seed(seed)

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
        df = core.load_data(self._network.data_path)
        input_dimension = df.shape[1] - 2
        X = df.values[:: self._skip, :input_dimension].astype(np.float32)
        y = df.values[:: self._skip, input_dimension:].astype(np.float32)

        self._network.fit(X, y, epochs=self._epochs)

        if self._save_params:
            self._network.save_params()

        if self._plot_loss:
            self._plot_losses()

    @property
    def _save_params(self) -> bool:
        save = None if self._confirm_save else True

        while save is None:
            answer = input("Save network params?(Y/n) ")
            if answer.lower() in ["y", "yes"]:
                save = True
            elif answer.lower() in ["n", "no"]:
                save = False
            else:
                print(f"Input {answer} is not valid.")

        return save

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
        self._add_network_parameters(parser)
        self._add_epochs(parser)
        self._add_skip(parser)
        self._add_seed(parser)
        self._add_resume_training(parser)
        self._add_confirm_save(parser)
        self._add_plot_loss(parser)

    def _add_network(self, parser):
        parser.add_argument(
            "network",
            help="Specify which network should be trained.",
            choices=NETWORK_TYPES.keys(),
        )

    def _add_network_parameters(self, parser):
        parser.add_argument(
            "-p",
            "--parameter",
            help="Add network parameter which will be parsed directly to the network.",
            nargs=2,
            action="append",
            metavar=("<key>", "<value>"),
            default=list(),
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

    def _add_resume_training(self, parser):
        parser.add_argument(
            "--resume",
            help="Resume training.",
            action="store_true",
            dest="resume_training",
        )

    def _add_confirm_save(self, parser):
        parser.add_argument(
            "--confirm-save",
            help="confirm to save after training. Otherwise training will be aborted.",
            action="store_true",
        )

    def _add_plot_loss(self, parser):
        parser.add_argument(
            "--plot-loss",
            help="Plot training and validation loss.",
            action="store_true",
        )

    def postprocess(self, arguments):
        arguments.network = NETWORK_TYPES[arguments.network](warm_start=True)
        self._load_network(arguments)
        arguments.network.set_params(**self._get_network_parameters(arguments))
        arguments.command = TrainNetwork

    def _load_network(self, arguments):
        if arguments.resume_training:
            arguments.network.load_params()

        del arguments.resume_training

    def _get_network_parameters(self, arguments) -> Dict:
        kwargs = dict()
        for key, value in arguments.parameter:
            kwargs[key] = float(value)

        del arguments.parameter

        return kwargs
