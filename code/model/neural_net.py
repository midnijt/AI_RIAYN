import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization


class RegularizedMLP:
    def __init__(self, n_inputs, n_outputs, search_space):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.search_space = search_space
        self.model = None
        self.optimizer = None
        self.regularizers = []

    def build_model(self, **params):
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Dense(64, activation="relu", input_shape=(self.n_inputs,))
        )
        self.model.add(tf.keras.layers.BatchNormalization())

        if params["BN-active"]:
            self.regularizers.append(tf.keras.regularizers.l2(params["BN-l2"]))
            self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        self.model.add(tf.keras.layers.Dropout(params["DO-dropout_rate"]))

        if params["DO-active"]:
            dropout_shape = params["DO-shape"]
            if dropout_shape == "funnel":
                dropout_shape = [1.0, 0.75, 0.5, 0.25]
            elif dropout_shape == "long funnel":
                dropout_shape = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25]
            elif dropout_shape == "diamond":
                dropout_shape = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            elif dropout_shape == "hexagon":
                dropout_shape = [1.0, 0.67, 0.33, 0.67, 0.33, 0.67, 0.33]
            elif dropout_shape == "brick":
                dropout_shape = [
                    1.0,
                    0.95,
                    0.9,
                    0.85,
                    0.8,
                    0.75,
                    0.7,
                    0.65,
                    0.6,
                    0.55,
                    0.5,
                ]
            elif dropout_shape == "triangle":
                dropout_shape = [1.0, 0.8, 0.6, 0.4, 0.2]
            elif dropout_shape == "stairs":
                dropout_shape = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            else:
                dropout_shape = [params["DO-dropout_rate"]]
            self.regularizers.append(tf.keras.layers.Dropout(dropout_shape))

        self.model.add(tf.keras.layers.Dense(self.n_outputs, activation="softmax"))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])

        if params["WD-active"]:
            self.regularizers.append(tf.keras.regularizers.l2(params["WD-l2"]))
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=params["lr"], decay=params["WD-decay"]
            )

        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def fit(self, X_train, y_train, X_val, y_val, params):
        self.build_model(**params)

        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            callbacks=tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            ),
            epochs=100,
            verbose=0,
        )

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y, axis=1)
        acc = accuracy_score(y_true, y_pred)
        return acc

    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, n_iterations=50):
        def objective_function(**params):
            for k in self.search_space.keys():
                if self.search_space[k]["type"] == "nominal":
                    params[k] = self.search_space[k]["values"][int(params[k])]
            self.fit(X_train, y_train, X_val, y_val, params)
            acc = self.evaluate(X_val, y_val)
            return -acc

        pbounds = {}
        for k, v in self.search_space.items():
            pbounds[k] = (v["min"], v["max"])

            if v["type"] == "nominal":
                pbounds[k] = (0, len(v["values"]) - 1)

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

        self.fit(X_train, y_train, X_val, y_val, best_params)
        return best_params


# Note that this code assumes that X_train, y_train, X_val, and y_val are numpy arrays representing the training and validation datasets, respectively. You will need to replace the dataset loading and preprocessing code with your own. Additionally, the code only implements the MLP model and not the XGBoost model mentioned in the paper. You will need to implement the XGBoost model separately and modify the code to compare the performance of the two models.


search_space = {
    "BN-active": {"type": "bool"},
    "BN-l2": {"type": "float", "min": 1e-7, "max": 1e-1},
    "DO-active": {"type": "bool"},
    "DO-dropout_rate": {"type": "float", "min": 0.0, "max": 0.8},
    "DO-shape": {
        "type": "nominal",
        "values": [
            "funnel",
            "long funnel",
            "diamond",
            "hexagon",
            "brick",
            "triangle",
            "stairs",
        ],
    },
    "WD-active": {"type": "bool"},
    "WD-l2": {"type": "float", "min": 1e-7, "max": 1e-1},
    "WD-decay": {"type": "float", "min": 1e-7, "max": 1e-1},
    "lr": {"type": "float", "min": 1e-5, "max": 1e-1},
}

model = RegularizedMLP(
    n_inputs=X_train.shape[1], n_outputs=y_train.shape[1], search_space=search_space
)
best_params = model.hyperparameter_tuning(
    X_train, y_train, X_val, y_val, n_iterations=50
)
print(best_params)
