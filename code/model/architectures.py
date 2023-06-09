import torch
import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, layer):
        super(SkipConnection, self).__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Shake(nn.Module):
    def __init__(self, path1, path2):
        super(Shake, self).__init__()
        self.path1 = path1
        self.path2 = path2

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        alpha = torch.rand(x1.shape[0], 1).to(x1.device)
        if not self.training:
            # During evaluation, use the average of the two paths
            alpha = 0.5 * torch.ones_like(alpha)

        return alpha * x1 + (1 - alpha) * x2


def shake_drop(x, p_drop):
    if p_drop == 0.0:
        return x
    else:
        batch_size = x.shape[0]
        mask = torch.bernoulli(torch.ones(batch_size, 1) * (1 - p_drop)).to(x.device)
        return mask * x / (1 - p_drop)


class Architectures:
    def create_hidden_layer(self, arch_type, **params):
        hidden_layer = [
            nn.Linear(self.n_hidden_units, self.n_hidden_units),
            nn.ReLU(),
        ]

        if params["BN-active"] > 0.5:
            hidden_layer.append(nn.BatchNorm1d(self.n_hidden_units))
        if params["DO-active"] > 0.5:
            hidden_layer.append(nn.Dropout(p=params["DO-dropout_rate"]))

        if arch_type == "SC":
            return SkipConnection(nn.Sequential(*hidden_layer))
        elif arch_type == "SS":
            return hidden_layer
        elif arch_type == "SD":
            return nn.Sequential(*hidden_layer)

        return hidden_layer

    def create_model(self, arch_type, **params):
        layers = [
            nn.Linear(self.n_inputs, self.n_hidden_units),
            nn.ReLU(),
        ]

        for _ in range(self.n_layers):
            if arch_type == "MLP":
                layers.extend(self.create_hidden_layer(arch_type, **params))
            elif arch_type == "SC":
                layers.append(self.create_hidden_layer(arch_type, **params))
            elif arch_type == "SS":
                path1 = self.create_hidden_layer(arch_type, **params)
                path2 = self.create_hidden_layer(arch_type, **params)
                layers.append(Shake(nn.Sequential(*path1), nn.Sequential(*path2)))
            elif arch_type == "SD":
                path1 = self.create_hidden_layer(arch_type, **params)
                path2 = self.create_hidden_layer(arch_type, **params)
                shake_drop_layer = Lambda(
                    lambda x: shake_drop(x, params["SD-max_prob"])
                    if self.training
                    else x
                )
                layers.append(nn.Sequential(*path1, *path2, shake_drop_layer))

        layers.append(nn.Linear(self.n_hidden_units, self.n_outputs))
        if not params["regression"]:
            layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def MLP(self, **params):
        self.create_model("MLP", **params)

    def SC(self, **params):
        self.create_model("SC", **params)

    def SS(self, **params):
        self.create_model("SS", **params)

    def SD(self, **params):
        self.create_model("SD", **params)
