from sklearn.datasets import load_wine
from model_training import train_and_evaluate_model


def main():
    dataset = load_wine()
    train_and_evaluate_model(dataset, is_regression=False)


if __name__ == "__main__":
    main()
