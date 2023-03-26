import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy


class RegularizedMLP(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, search_space, n_layers=9, n_hidden_units=512
    ):
        super(RegularizedMLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.search_space = search_space
        self.model = None
        self.optimizer = None
        self.regularizers = []
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units

    def build_model(self, **params):
        layers = [
            nn.Linear(self.n_inputs, self.n_hidden_units),
            nn.ReLU(),
        ]

        for _ in range(7):
            layers.extend(
                [
                    nn.Linear(self.n_hidden_units, self.n_hidden_units),
                    nn.ReLU(),
                ]
            )
            if params["BN-active"]:
                layers.append(nn.BatchNorm1d(self.n_hidden_units))
            layers.append(nn.Dropout(p=params["DO-dropout_rate"]))

        layers.extend(
            [
                nn.Linear(self.n_hidden_units, self.n_outputs),
                nn.Softmax(dim=1),
            ]
        )

        self.model = nn.Sequential(*layers)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=params["lr"], weight_decay=params["WD-l2"]
        )

    def fit(self, X_train, y_train, X_val, y_val, params, device="cpu", batch_size=32):
        self.build_model(**params)
        self.model.to(device)

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
        criterion = nn.CrossEntropyLoss()
        best_model = None
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(100):
            self.model.train()

            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

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

    def hyperparameter_tuning(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_iterations=5,
        device="cpu",
        batch_size=32,
    ):
        def objective_function(**params):
            for k in self.search_space.keys():
                if self.search_space[k]["type"] == "nominal":
                    params[k] = self.search_space[k]["values"][int(params[k])]
            self.fit(X_train, y_train, X_val, y_val, params, device, batch_size)
            acc = self.evaluate(X_val, y_val, device)
            return -acc

        pbounds = {}
        for k, v in self.search_space.items():
            if v["type"] == "bool":
                pbounds[k] = (0, 1)
            elif v["type"] == "nominal":
                pbounds[k] = (0, len(v["values"]) - 1)
            else:
                pbounds[k] = (v["min"], v["max"])

        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=42,
        )

        optimizer.maximize(n_iter=n_iterations)
        best_params = {}
        for k, v in optimizer.max["params"].items():
            if self.search_space[k]["type"] == "nominal":
                best_params[k] = self.search_space[k]["values"][int(v)]
            else:
                best_params[k] = v

        self.fit(X_train, y_train, X_val, y_val, best_params, device, batch_size)
        return best_params
