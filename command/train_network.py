from argparse import Namespace
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from network import NeuralNetwork
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from .command import Command


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

    def __init__(self, optimizer, patience: int, factor: float):
        ReduceLROnPlateau.__init__(
            self, optimizer, verbose=True, patience=patience, factor=factor
        )

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    tqdm.write(
                        "Epoch {:.0f}: reducing learning rate"
                        " of group to {:.4e}.".format(epoch, new_lr)
                    )


class TrainNetwork(Command):
    _training_dataloader: DataLoader
    _validation_dataloader: DataLoader
    _device: str
    _model: NeuralNetwork
    _loss_function: ...
    _optimizer: Optimizer
    _learning_rate_scheduler: CustomScheduler

    def __init__(self, args: Namespace):
        self._args = args

        self._build_data_loader()
        self._setup_device()
        self._build_model()

        self._loss_function = nn.MSELoss()
        self._optimizer = AdamW(self._model.parameters(), lr=self._args.learning_rate)
        self._learning_rate_scheduler = CustomScheduler(
            self._optimizer, self._args.patience, self._args.factor
        )

    def _build_data_loader(self):
        training_data = CustomSubgridDataset(self._args.train_path)
        validation_data = CustomSubgridDataset(self._args.validate_path)

        self._training_dataloader = DataLoader(
            training_data, batch_size=self._args.batch_size, shuffle=True
        )
        self._validation_dataloader = DataLoader(
            validation_data, batch_size=self._args.batch_size, shuffle=True
        )

    def _setup_device(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        tqdm.write(f"INFO: Using {self._device} device.")

    def _build_model(self):
        self._model = NeuralNetwork()
        input_dimension = self._get_input_dimension()
        self._model.build_network([input_dimension, *self._args.hidden_neurons, 2])
        self._model.setup_normalization(
            self._training_dataloader.dataset.mean[:input_dimension],
            self._training_dataloader.dataset.std[:input_dimension],
        )

        tqdm.write(str(self._model))

    def _get_input_dimension(self) -> int:
        return len(self._training_dataloader.dataset.data.columns) - 2

    def execute(self):
        training_loss = np.empty(self._args.epochs)
        validate_loss = np.empty(self._args.epochs)

        with trange(self._args.epochs, desc="Training", unit="epoch", leave=False) as t:
            for epoch in t:
                training_loss[epoch] = self._train()
                validate_loss[epoch] = self._validate()
                self._learning_rate_scheduler.step(training_loss[epoch])
                t.set_postfix(loss=training_loss[epoch])

        self._save_model()
        self._plot_losses(training_loss, validate_loss)

    def _train(self):
        num_batches = len(self._training_dataloader)
        self._model.train()
        training_loss = 0

        for X, y in tqdm(
            self._training_dataloader, leave=False, unit="batch", desc="Train"
        ):
            X, y = X.to(self._device), y.to(self._device)

            # Compute prediction error
            pred = self._model(X)
            loss = self._loss_function(pred, y)

            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            training_loss += loss.item()

        return training_loss / num_batches

    def _validate(self):
        num_batches = len(self._validation_dataloader)
        self._model.eval()
        validation_loss = 0

        with torch.no_grad():
            for X, y in tqdm(
                self._validation_dataloader, leave=False, unit="batch", desc="Validate"
            ):
                X, y = X.to(self._device), y.to(self._device)
                pred = self._model(X)
                validation_loss += self._loss_function(pred, y).item()
        validation_loss /= num_batches

        return validation_loss

    def _save_model(self):
        torch.save(self._model.state_dict(), self._args.network_path)

    def _plot_losses(self, training_loss, validation_loss):
        tqdm.write(f"Reached loss: {min(training_loss)}")

        epochs = np.arange(self._args.epochs)
        plt.plot(epochs, training_loss, label="training loss")
        plt.plot(epochs, validation_loss, label="validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.suptitle(self._args.suptitle)
        plt.legend()
        plt.show()
