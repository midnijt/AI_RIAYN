from abc import ABC, abstractmethod
from bayes_opt import BayesianOptimization


class BaseModel(ABC):
    def __init__(self, search_space):
        self.search_space = search_space
        self.model = None

    @abstractmethod
    def build_model(self, **params):
        pass

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, params):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass

    def hyperparameter_tuning(
        self, X_train, y_train, X_val, y_val, n_iterations=50, **fit_kwargs
    ):
        def objective_function(**params):
            for k, v in self.search_space.items():
                if v["type"] == "int":
                    params[k] = int(params[k])
            self.fit(X_train, y_train, X_val, y_val, params)
            metric = self.evaluate(X_val, y_val)
            return -metric

        eps = 1e-6
        pbounds = {}
        for k, v in self.search_space.items():
            if v["type"] == "bool":
                pbounds[k] = (0, 1)
            elif v["type"] == "nominal":
                pbounds[k] = (0 + eps, len(v["values"]) - eps)
            else:
                pbounds[k] = v["range"]

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

        self.fit(X_train, y_train, X_val, y_val, best_params, **fit_kwargs)

        return best_params
