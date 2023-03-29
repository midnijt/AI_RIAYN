eps = 1e-8

mlp = {
    "BN-active": {
        "type": "bool",
        "default": False,
    },
    "SWA-active": {
        "type": "bool",
        "default": False,
    },
    "LA-active": {
        "type": "bool",
        "default": False,
    },
    "LA-step_size": {
        "type": "float",
        "default": 0.5,
        "conditions": ["LA-active"],
        "range": [0.5 + eps, 0.8 - eps],
    },
    "LA-num_steps": {
        "type": "int",
        "default": 1,
        "conditions": ["LA-active"],
        "range": [1, 5],
    },
    "WD-active": {
        "type": "bool",
        "default": False,
    },
    "WD-decay_factor": {
        "type": "float",
        "default": 1e-5,
        "conditions": ["WD-active"],
        "range": [1e-5 + eps, 0.1 - eps],
    },
    "DO-active": {
        "type": "bool",
        "default": False,
    },
    "DO-dropout_rate": {
        "type": "float",
        "default": 0.0,
        "conditions": ["DO-active"],
        "range": [0.0 + eps, 0.8 - eps],
    },
    "SE-active": {
        "type": "bool",
        "default": False,
    },
    "SC-active": {
        "type": "bool",
        "default": False,
    },
    "MB-choice": {
        "type": "nominal",
        "default": "Standard",
        "conditions": ["SC-active"],
        "values": ["SS", "SD", "Standard"],
    },
    "SD-max_prob": {
        "type": "float",
        "default": 0.0,
        "conditions": ["SC-active", "MB-choice=SD"],
        "range": [0.0 + eps, 1.0 - eps],
    },
    "Augment": {
        "type": "nominal",
        "default": "None",
        "values": ["MU", "CM", "CO", "AT", "None"],
    },
    "MU-mix_mag": {
        "type": "float",
        "default": 0.0,
        "conditions": ["Augment=MU"],
        "range": [0.0 + eps, 1.0 - eps],
    },
    "CM-prob": {
        "type": "float",
        "default": 0.0,
        "conditions": ["Augment=CM"],
        "range": [0.0 + eps, 1.0 - eps],
    },
    "CO-prob": {
        "type": "float",
        "default": 0.0,
        "conditions": ["Augment=CO"],
        "range": [0.0 + eps, 1.0 - eps],
    },
}

xgb = {
    "eta": {"type": "float", "range": [0.001 + eps, 1.0 - eps]},
    "lambda": {"type": "float", "range": [1e-10 + eps, 1.0 - eps]},
    "alpha": {"type": "float", "range": [1e-10 + eps, 1.0 - eps]},
    "num_round": {"type": "int", "range": [1, 1000]},
    "gamma": {"type": "float", "range": [0.1 + eps, 1.0 - eps]},
    "colsample_bylevel": {"type": "float", "range": [0.1 + eps, 1.0 - eps]},
    "colsample_bynode": {"type": "float", "range": [0.1 + eps, 1.0 - eps]},
    "colsample_bytree": {"type": "float", "range": [0.5 + eps, 1.0 - eps]},
    "max_depth": {"type": "int", "range": [1, 20]},
    "max_delta_step": {"type": "int", "range": [0, 10]},
    "min_child_weight": {"type": "float", "range": [0.1 + eps, 20.0 - eps]},
    "subsample": {"type": "float", "range": [0.01 + eps, 1.0 - eps]},
}
