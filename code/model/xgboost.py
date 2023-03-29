import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from base import BaseModel


class XGBoost(BaseModel):
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


xgb_search_space = {
    "eta": {"type": "float", "range": [0.001, 1.0]},
    "lambda": {"type": "float", "range": [1e-10, 1.0]},
    "alpha": {"type": "float", "range": [1e-10, 1.0]},
    "num_round": {"type": "int", "range": [1, 1000]},
    "gamma": {"type": "float", "range": [0.1, 1.0]},
    "colsample_bylevel": {"type": "float", "range": [0.1, 1.0]},
    "colsample_bynode": {"type": "float", "range": [0.1, 1.0]},
    "colsample_bytree": {"type": "float", "range": [0.5, 1.0]},
    "max_depth": {"type": "int", "range": [1, 20]},
    "max_delta_step": {"type": "int", "range": [0, 10]},
    "min_child_weight": {"type": "float", "range": [0.1, 20.0]},
    "subsample": {"type": "float", "range": [0.01, 1.0]},
}
