from model import *


params = {
    "task": "regression",
    "trials_train": 5,
    "trials_test": 5,
    "initialization": {
        "type": "uniform",
        "min": -0.5,
        "max": 0.5
    },
    "early_stopping": {
        "monitor": "val_loss",
        "patience": 20,
        "delta": 0.00
    },
    "max_epochs": [500],
    "learning_rate": [0.01],
    "batch_size": [16],
    "nesterov": False,
    "momentum": [0.9],
    "learning_rate_decay": {
        "epochs": 100,
        "lr_final": 0.001
    },
    "regularization": [
        {
            "type": None,
        }
    ],
    "layers": [
        {
            "activations": ["relu"],
            "units": [10],
        }
    ]
}


k_fold_val = {
    "type": "k-fold",
    "n_folds": 5
}

hold_out_val = {
    "type": "hold-out",
    "val_split": 0.15
}


config = get_configurations(params)
# print(config)
net = get_model(config[0], 9, 2)
torch.save(net, "model.pt")