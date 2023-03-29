from sklearn.datasets import fetch_covtype
from model_training import train_and_evaluate_model


def main():
    dataset = fetch_covtype()
    dataset.target -= 1  # zero-index the labels
    train_and_evaluate_model(dataset, is_regression=False)


if __name__ == "__main__":
    main()
