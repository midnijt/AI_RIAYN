from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from bayes_opt import BayesianOptimization
from .base import BaseModel


class XGBoost(BaseModel):
    def __init__(self, num_class, search_space):
        self.num_class = num_class
        self.regression = num_class == 1
        self.search_space = search_space
        self.model = None

    def build_model(self, **params):
        if self.regression:
            self.model = XGBRegressor(
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
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=10,
            )
        else:
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
                num_class=self.num_class,
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
        if self.regression:
            metric = mean_squared_error(y, y_pred)
        else:
            # smaller is better
            metric = -accuracy_score(y, y_pred)
        return metric
