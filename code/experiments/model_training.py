import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append("/Users/jtam/projects/AI_RIAYN/code/")
from model.NeuralNet import NeuralNet
from model.xgboost import XGBoost
import model.search_space as search_space


def train_and_evaluate_model(dataset, is_regression=False):
    X = dataset.data
    y = dataset.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_outputs = 1 if is_regression else len(np.unique(y))

    # Train and evaluate MLP
    mlp = NeuralNet(
        n_inputs=X_train.shape[1],
        n_outputs=n_outputs,
        search_space=search_space.mlp,
        n_layers=2,
        n_hidden_units=64,
    )

    mlp_params = mlp.hyperparameter_tuning(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        n_iterations=1,
    )

    if is_regression:
        mse = mlp.evaluate(X_test, y_test)
        print(f"MLP mean squared error: {mse:.2f}")
    else:
        accuracy = mlp.evaluate(X_test, y_test)
        print(f"MLP accuracy: {accuracy:.2f}")
    print("Best MLP parameters:", mlp_params)

    # Train and evaluate XGBoost
    xgb = XGBoost(num_class=n_outputs, search_space=search_space.xgb)

    xgb_params = xgb.hyperparameter_tuning(
        X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test
    )

    if is_regression:
        mse = xgb.evaluate(X_test, y_test)
        print(f"XGBoost mean squared error: {mse:.2f}")
    else:
        accuracy = xgb.evaluate(X_test, y_test)
        print(f"XGBoost accuracy: {accuracy:.2f}")
    print("Best XGBoost parameters:", xgb_params)
