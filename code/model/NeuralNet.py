import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from copy import deepcopy
from .optimizer import CosineAnnealingLRWithWarmup, SWALookahead
from .base import BaseModel
from .architectures import Architectures
import sys

sys.path.append("/Users/jtam/projects/AI_RIAYN/code/")
from data.augmentation import CutOut1D, MixUp1D, CutMix1D


class NeuralNet(nn.Module, BaseModel, Architectures):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        search_space,
        n_layers=9,
        n_hidden_units=512,
        learning_rate=0.001,
    ):
        super(NeuralNet, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.search_space = search_space
        self.model = None
        self.optimizer = None
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.learning_rate = learning_rate

    def build_model(self, **params):
        MB_choice = self.search_space["MB-choice"]["values"][int(params["MB-choice"])]
        print(MB_choice)
        if params["SC-active"] > 0.5:
            if MB_choice == "Standard":
                self.SC(**params)
            elif MB_choice == "SD":
                self.SD(**params)
            elif MB_choice == "SS":
                self.SS(**params)
            else:
                raise ValueError(f"Invalid model choice: {params['MB-choice']}")
        else:
            self.MLP(**params)

    def _prepare_data(
        self, X_train, y_train, X_val, y_val, batch_size=32, device="cpu"
    ):
        train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).to(device),
            torch.tensor(y_train, dtype=torch.long).to(device),
        )

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        val_data = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32).to(device),
            torch.tensor(y_val, dtype=torch.long).to(device),
        )
        val_loader = DataLoader(val_data, batch_size=batch_size)

        return train_loader, val_loader

    def _prepare_augmentation(self, params, data_shape):
        augmentation = self.search_space["Augment"]["values"][int(params["Augment"])]

        if augmentation == "MU":
            augmentation = MixUp1D(alpha=params["MU-mix_mag"])
        elif augmentation == "CM":
            augmentation = CutMix1D(alpha=params["CM-prob"])
        elif augmentation == "CO":
            augmentation = CutOut1D(
                p=int(params["CO-prob"] * data_shape[1]),
            )
        else:
            augmentation = None

        return augmentation

    def fit(self, X_train, y_train, X_val, y_val, params, device="cpu", batch_size=32):
        self.build_model(**params)
        self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        best_model = None
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        n_snapshots = 5
        snapshot_interval = 20
        n_epochs = n_snapshots * snapshot_interval

        weight_decay = (params["WD-active"] > 0.5) * params["WD-decay_factor"]
        base_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )

        optimizer = SWALookahead(
            base_optimizer,
            swa=(params["SWA-active"] > 0.5),
            swa_start=50,
            swa_freq=5,
            swa_lr=None,
            lookahead=(params["LA-active"] > 0.5),
            la_steps=params["LA-num_steps"],
            la_alpha=params["LA-step_size"],
        )

        warmup_epochs = 10
        scheduler = CosineAnnealingLRWithWarmup(
            optimizer, warmup_epochs, n_epochs, eta_min=0, last_epoch=-1
        )

        train_loader, val_loader = self._prepare_data(
            X_train, y_train, X_val, y_val, batch_size=batch_size, device=device
        )
        augmentation = self._prepare_augmentation(params, data_shape=X_train.shape)
        for epoch in range(n_epochs):
            self.model.train()

            if epoch % snapshot_interval == 0:
                snapshot_index = epoch // snapshot_interval
                snapshot_model_filename = f"snapshot_model_{snapshot_index}.pt"
                torch.save(self.model.state_dict(), snapshot_model_filename)
                snapshot_optimizer_state_filename = (
                    f"snapshot_optimizer_state_{snapshot_index}.pt"
                )
                torch.save(optimizer.state_dict(), snapshot_optimizer_state_filename)
                snapshot_val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        snapshot_val_loss += loss.item()

                snapshot_val_loss /= len(val_loader)
                if snapshot_val_loss < best_val_loss:
                    best_val_loss = snapshot_val_loss
                    best_model = deepcopy(self.model)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            for inputs, targets in train_loader:
                if augmentation is not None:
                    inputs = augmentation(inputs)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(self.model)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        self.model = best_model

    def evaluate(self, X, y, device="cpu"):
        self.model.eval()
        self.model.to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)
        y_pred = self.model(X_tensor)
        _, y_pred = torch.max(y_pred, 1)
        acc = accuracy_score(y_tensor.cpu(), y_pred.cpu())
        return acc
