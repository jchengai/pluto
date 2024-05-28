import torch.nn as nn


def build_mlp(c_in, channels, norm=None, activation="relu"):
    layers = []
    num_layers = len(channels)

    if norm is not None:
        norm = get_norm(norm)

    activation = get_activation(activation)

    for k in range(num_layers):
        if k == num_layers - 1:
            layers.append(nn.Linear(c_in, channels[k], bias=True))
        else:
            if norm is None:
                layers.extend([nn.Linear(c_in, channels[k], bias=True), activation()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, channels[k], bias=False),
                        norm(channels[k]),
                        activation(),
                    ]
                )
            c_in = channels[k]

    return nn.Sequential(*layers)


def get_norm(norm: str):
    if norm == "bn":
        return nn.BatchNorm1d
    elif norm == "ln":
        return nn.LayerNorm
    else:
        raise NotImplementedError


def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    else:
        raise NotImplementedError
