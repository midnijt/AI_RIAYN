import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization


class XGBoost:
    def __init__(self, search_space):
        self.search_space = search_space
        self.model = None

    def build_model(self, **params):
        self.model = XGBClassifier(
            learning_rate=params["eta"],
            reg_lambda=params["lambda"],
            reg_alpha=params["alpha"],
            n_estimators=int(params["num_round"]),
            gamma=params["gamma"],
            colsample_bylevel=params["colsample_bylevel"],
            colsample_bynode=params["colsample_bynode"],
            colsample_bytree=params["colsample_bytree"],
            max_depth=int(params["max_depth"]),
            max_delta_step=int(params["max_delta_step"]),
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            objective="multi:softmax",
            num_class=10,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=10,
        )

    def fit(self, X_train, y_train, X_val, y_val, params):
        self.build_model(**params)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        return acc

    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, n_iterations=50):
        def objective_function(**params):
            for k, v in self.search_space.items():
                if v["type"] == "int":
                    params[k] = int(params[k])
            self.fit(X_train, y_train, X_val, y_val, params)
            acc = self.evaluate(X_val, y_val)
            return -acc

        pbounds = {}
        for k, v in self.search_space.items():
            pbounds[k] = (v["min"], v["max"])

        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=42,
        )

        optimizer.maximize(n_iter=n_iterations)

        best_params = optimizer.max["params"]

        for k, v in self.search_space.items():
            if v["type"] == "int":
                best_params[k] = int(best_params[k])

        self.fit(X_train, y_train, X_val, y_val, best_params)

        return best_params


xgb_search_space = {
    "eta": {"type": "float", "min": 0.001, "max": 1.0},
    "lambda": {"type": "float", "min": 1e-10, "max": 1.0},
    "alpha": {"type": "float", "min": 1e-10, "max": 1.0},
    "num_round": {"type": "int", "min": 1, "max": 1000},
    "gamma": {"type": "float", "min": 0.1, "max": 1.0},
    "colsample_bylevel": {"type": "float", "min": 0.1, "max": 1.0},
    "colsample_bynode": {"type": "float", "min": 0.1, "max": 1.0},
    "colsample_bytree": {"type": "float", "min": 0.5, "max": 1.0},
    "max_depth": {"type": "int", "min": 1, "max": 20},
    "max_delta_step": {"type": "int", "min": 0, "max": 10},
    "min_child_weight": {"type": "float", "min": 0.1, "max": 20.0},
    "subsample": {"type": "float", "min": 0.01, "max": 1.0},
}