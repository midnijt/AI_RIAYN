import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append("/Users/jtam/projects/AI_RIAYN/code/")
from model.NeuralNet import NeuralNet
from model.xgboost import XGBoost
import model.search_space as search_space


def main():
    # Load dataset
    dataset = fetch_covtype()
    X = dataset.data
    y = dataset.target - 1  # 0-index labels

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate MLP
    mlp = NeuralNet(
        n_inputs=X_train.shape[1],
        n_outputs=len(np.unique(y)),  # classification
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

    accuracy = mlp.evaluate(X_test, y_test)
    print(f"MLP accuracy: {accuracy:.2f}")
    print("Best MLP parameters:", mlp_params)

    # Train and evaluate XGBoost
    xgb = XGBoost(num_class=len(np.unique(y)), search_space=search_space.xgb)

    xgb_params = xgb.hyperparameter_tuning(
        X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test
    )

    accuracy = xgb.evaluate(X_test, y_test)
    print(f"XGBoost accuracy: {accuracy:.2f}")
    print("Best XGBoost parameters:", xgb_params)


if __name__ == "__main__":
    main()
