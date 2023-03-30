from sklearn.datasets import fetch_openml
from model_training import train_and_evaluate_model


def main():
    dataset = fetch_openml(data_id=44964, as_frame=False, parser="auto")
    # Note: The 'as_frame=True' argument will return the dataset as a pandas DataFrame
    train_and_evaluate_model(dataset, is_regression=True)


if __name__ == "__main__":
    main()
