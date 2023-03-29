from sklearn.datasets import fetch_california_housing
from model_training import train_and_evaluate_model


def main():
    dataset = fetch_california_housing()
    train_and_evaluate_model(dataset, is_regression=True)


if __name__ == "__main__":
    main()
