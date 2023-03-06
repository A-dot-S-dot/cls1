import pickle

import core
import defaults
import matplotlib.pyplot as plt
import numpy as np
import torch
from skorch import NeuralNetRegressor
from torch import nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from command import Command


class NeuralNetwork(nn.Module):
    input_dimension = 8

    def __init__(self):
        nn.Module.__init__(self)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_dimension, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class Curvature:
    step_length: float

    def __init__(self, step_length: float):
        self.step_length = step_length

    def __call__(self, u0, u1, u2, u3):
        return (
            self._calculate_curvature(u0, u1, u2)
            + self._calculate_curvature(u1, u2, u3)
        ) / 2

    def _calculate_curvature(self, u0, u1, u2):
        return (
            abs(u0 - 2 * u1 + u2)
            * self.step_length
            / (self.step_length**2 + 0.25 * (u0 - u2) ** 2) ** (3 / 2)
        )


class TrainNetwork(Command):
    _skip: int
    _data_path: str
    _network_path: str
    _network: ...
    _pipeline: ...

    def __init__(self, data_path: str, network_path: str, epochs=None, skip=None):
        self._network_path = network_path
        self._data_path = data_path
        self._skip = skip or defaults.SKIP

        self._network = NeuralNetRegressor(
            NeuralNetwork,
            max_epochs=epochs or defaults.EPOCHS,
            lr=0.1,
            device=self._get_device(),
        )
        self._pipeline = Pipeline([("scale", StandardScaler()), ("net", self._network)])

    def _get_device(self) -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"INFO: Using {device} device.")

        return device

    def execute(self):
        df = core.load_data(self._data_path)
        X = df.values[:: self._skip, :8].astype(np.float32)
        y = df.values[:: self._skip, 8:].astype(np.float32)

        self._pipeline.fit(X, y)
        self._save_model()
        self._plot_losses()

    def _save_model(self):
        with open(self._network_path, "wb") as f:
            pickle.dump(self._pipeline, f)

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
