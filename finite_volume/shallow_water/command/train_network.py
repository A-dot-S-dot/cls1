import pickle

import core
import defaults
import matplotlib.pyplot as plt
import numpy as np
import torch

from command import Command

from ..solver.reduced_llf import LaxFriedrichsEstimator


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

        estimator_type = estimator_type or LaxFriedrichsEstimator
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
