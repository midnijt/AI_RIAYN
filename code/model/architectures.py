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


class Architectures:
    def MLP(self, **params):
        layers = [
            nn.Linear(self.n_inputs, self.n_hidden_units),
            nn.ReLU(),
        ]

        for _ in range(self.n_layers):
            layers.extend(
                [
                    nn.Linear(self.n_hidden_units, self.n_hidden_units),
                    nn.ReLU(),
                ]
            )
            if params["BN-active"] > 0.5:
                layers.append(nn.BatchNorm1d(self.n_hidden_units))
            if params["DO-active"] > 0.5:
                layers.append(nn.Dropout(p=params["DO-dropout_rate"]))

        layers.append(nn.Linear(self.n_hidden_units, self.n_outputs))
        if not params["regression"]:
            layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def SC(self, **params):
        layers = [
            nn.Linear(self.n_inputs, self.n_hidden_units),
            nn.ReLU(),
        ]

        for _ in range(self.n_layers):
            hidden_layer = [
                nn.Linear(self.n_hidden_units, self.n_hidden_units),
                nn.ReLU(),
            ]
            if params["BN-active"] > 0.5:
                hidden_layer.append(nn.BatchNorm1d(self.n_hidden_units))
            if params["DO-active"] > 0.5:
                hidden_layer.append(nn.Dropout(p=params["DO-dropout_rate"]))

            layers.append(SkipConnection(nn.Sequential(*hidden_layer)))

        layers.append(nn.Linear(self.n_hidden_units, self.n_outputs))
        if not params["regression"]:
            layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def SS(self, **params):
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

        layers = [
            nn.Linear(self.n_inputs, self.n_hidden_units),
            nn.ReLU(),
        ]

        for _ in range(self.n_layers):
            # First path through the residual block
            path1 = [
                nn.Linear(self.n_hidden_units, self.n_hidden_units),
                nn.ReLU(),
            ]
            if params["BN-active"] > 0.5:
                path1.append(nn.BatchNorm1d(self.n_hidden_units))
            if params["DO-active"] > 0.5:
                path1.append(nn.Dropout(p=params["DO-dropout_rate"]))

            # Second path through the residual block
            path2 = [
                nn.Linear(self.n_hidden_units, self.n_hidden_units),
                nn.ReLU(),
            ]
            if params["BN-active"] > 0.5:
                path2.append(nn.BatchNorm1d(self.n_hidden_units))
            if params["DO-active"] > 0.5:
                path2.append(nn.Dropout(p=params["DO-dropout_rate"]))

            # Combine the two paths with Shake-Shake regularization
            layers.append(Shake(nn.Sequential(*path1), nn.Sequential(*path2)))

        # Final layers

        layers.append(nn.Linear(self.n_hidden_units, self.n_outputs))
        if not params["regression"]:
            layers.append(nn.Softmax(dim=1))
        # Build the model
        self.model = nn.Sequential(*layers)

    def SD(self, **params):
        # Define the Shake-Drop function
        def shake_drop(x, p_drop):
            if p_drop == 0.0:
                return x
            else:
                batch_size = x.shape[0]
                mask = torch.bernoulli(torch.ones(batch_size, 1) * (1 - p_drop)).to(
                    x.device
                )
                return mask * x / (1 - p_drop)

        layers = [
            nn.Linear(self.n_inputs, self.n_hidden_units),
            nn.ReLU(),
        ]

        for _ in range(self.n_layers):
            # First path through the residual block
            path1 = [
                nn.Linear(self.n_hidden_units, self.n_hidden_units),
                nn.ReLU(),
            ]
            if params["BN-active"] > 0.5:
                path1.append(nn.BatchNorm1d(self.n_hidden_units))
            if params["DO-active"] > 0.5:
                path1.append(nn.Dropout(p=params["DO-dropout_rate"]))

            # Second path through the residual block
            path2 = [
                nn.Linear(self.n_hidden_units, self.n_hidden_units),
                nn.ReLU(),
            ]
            if params["BN-active"] > 0.5:
                path2.append(nn.BatchNorm1d(self.n_hidden_units))
            if params["DO-active"] > 0.5:
                path2.append(nn.Dropout(p=params["DO-dropout_rate"]))

            # Combine the two paths with Shake-Shake regularization
            layers.append(
                nn.Sequential(
                    *[
                        nn.Sequential(*path1),
                        nn.Sequential(*path2),
                        Lambda(
                            lambda x: shake_drop(x, params["SD-max_prob"])
                            if self.training
                            else x
                        ),
                    ]
                )
            )

        # Final layers

        layers.append(nn.Linear(self.n_hidden_units, self.n_outputs))
        if not params["regression"]:
            layers.append(nn.Softmax(dim=1))

        # Build the model
        self.model = nn.Sequential(*layers)
