from typing import Sequence

import defaults
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pde_solver.discretization import DiscreteSolution, TemporalInterpolation
from pde_solver.network import NeuralNetwork
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange

from command import Command


class CustomSubgridDataset(Dataset):
    data: pd.DataFrame
    mean: Sequence[float]
    std: Sequence[float]
    input_dimension: int

    def __init__(self, data_file: str):
        self.data = pd.read_csv(
            data_file, header=[0, 1], skipinitialspace=True, index_col=0
        )
        self.mean = list(self.data.mean())
        self.std = list(self.data.std())
        self.input_dimension = len(self.data.columns) - 2

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        coarse_solutions = self.data.iloc[index, : self.input_dimension]
        subgrid_fluxes = self.data.iloc[index, self.input_dimension :]

        return torch.tensor(coarse_solutions, dtype=torch.float32), torch.tensor(
            subgrid_fluxes, dtype=torch.float32
        )


class CustomScheduler(ReduceLROnPlateau):
    optimizer: Optimizer
    factor: float
    min_lrs: ...
    eps: float
    verbose: bool

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    tqdm.write(
                        "Epoch {:.0f}: reducing learning rate to {:.4e}.".format(
                            epoch, new_lr
                        )
                    )


class TrainNetwork(Command):
    _training_dataloader: DataLoader
    _validation_dataloader: DataLoader
    _device: str
    _network: nn.Module
    _loss_function: ...
    _optimizer: Optimizer
    _scheduler: CustomScheduler
    _epochs: int
    _network_path: str

    def __init__(
        self,
        epochs,
        batch_size=None,
        training_data_path=None,
        validation_data_path=None,
        network=None,
        learning_rate=None,
        scheduler=None,
        network_path=None,
    ):
        self._epochs = epochs

        batch_size = batch_size or defaults.BATCH_SIZE
        training_data_path = training_data_path or defaults.TRAINING_DATA_PATH
        validation_data_path = validation_data_path or defaults.VALIDATION_DATA_PATH
        self._build_data_loader(training_data_path, validation_data_path, batch_size)

        self._network = network or self._build_network()
        tqdm.write(str(self._network))

        self._setup_device()

        self._loss_function = nn.MSELoss()

        learning_rate = learning_rate or defaults.LEARNING_RATE
        self._optimizer = AdamW(self._network.parameters(), lr=learning_rate)
        self._scheduler = scheduler or CustomScheduler(self._optimizer, verbose=True)

        self._network_path = network_path or defaults.NETWORK_PATH

    def _build_data_loader(
        self, training_data_path: str, validation_data_path: str, batch_size: int
    ):
        training_data = CustomSubgridDataset(training_data_path)
        validation_data = CustomSubgridDataset(validation_data_path)

        self._training_dataloader = DataLoader(
            training_data, batch_size=batch_size, shuffle=True
        )
        self._validation_dataloader = DataLoader(
            validation_data, batch_size=batch_size, shuffle=True
        )

    def _build_network(self) -> nn.Module:
        input_dimension = len(self._training_dataloader.dataset.data.columns) - 2
        return NeuralNetwork(
            self._training_dataloader.dataset.mean[:input_dimension],
            self._training_dataloader.dataset.std[:input_dimension],
        )

    def _setup_device(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        tqdm.write(f"INFO: Using {self._device} device.")

    def execute(self):
        training_loss = np.empty(self._epochs)
        validate_loss = np.empty(self._epochs)

        with trange(self._epochs, desc="Training", unit="epoch") as t:
            for epoch in t:
                training_loss[epoch] = self._train()
                validate_loss[epoch] = self._validate()
                self._scheduler.step(training_loss[epoch])
                t.set_postfix(loss=training_loss[epoch])

        self._save_model()
        self._plot_losses(training_loss, validate_loss)

    def _train(self):
        num_batches = len(self._training_dataloader)
        self._network.train()
        training_loss = 0

        for X, y in tqdm(
            self._training_dataloader, leave=False, unit="batch", desc="Train"
        ):
            X, y = X.to(self._device), y.to(self._device)

            # Compute prediction error
            pred = self._network(X)
            loss = self._loss_function(pred, y)

            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            training_loss += loss.item()

        return training_loss / num_batches

    def _validate(self):
        num_batches = len(self._validation_dataloader)
        self._network.eval()
        validation_loss = 0

        with torch.no_grad():
            for X, y in tqdm(
                self._validation_dataloader, leave=False, unit="batch", desc="Validate"
            ):
                X, y = X.to(self._device), y.to(self._device)
                pred = self._network(X)
                validation_loss += self._loss_function(pred, y).item()
        validation_loss /= num_batches

        return validation_loss

    def _save_model(self):
        torch.save(self._network.state_dict(), self._network_path)

    def _plot_losses(self, training_loss, validation_loss):
        tqdm.write(f"Reached loss: {min(training_loss)}")

        epochs = np.arange(self._epochs)
        plt.close()

        plt.plot(epochs, np.log10(training_loss), label="training loss")
        plt.plot(epochs, np.log10(validation_loss), label="validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Log(Loss)")
        plt.legend()
        plt.savefig("data/loss.png")
        plt.show()
