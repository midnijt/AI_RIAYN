import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append("/Users/jtam/projects/AI_RIAYN/code/")
from model.mlp import RegularizedMLP, mlp_search_space
from model.xgboost import XGBoost, xgb_search_space


def main():
    # Load dataset
    dataset = load_breast_cancer()
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

    # Train and evaluate MLP
    mlp = RegularizedMLP(
        n_inputs=X_train.shape[1],
        n_outputs=len(np.unique(y_train)),
        search_space=mlp_search_space,
        n_layers=1,
        n_hidden_units=64,
    )

    mlp_params = mlp.hyperparameter_tuning(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        n_iterations=1,
    )

    acc = mlp.evaluate(X_test, y_test)
    print(f"MLP accuracy: {acc:.2f}")
    print("Best MLP parameters:", mlp_params)

    # Train and evaluate XGBoost
    xgb = XGBoost(search_space=xgb_search_space)

    xgb_params = xgb.hyperparameter_tuning(
        X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test
    )

    acc = xgb.evaluate(X_test, y_test)
    print(f"XGBoost accuracy: {acc:.2f}")
    print("Best XGBoost parameters:", xgb_params)


if __name__ == "__main__":
    main()
