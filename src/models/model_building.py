# model building
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
import sys
import os
from typing import Tuple
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ---------------- LOGGING CONFIG ---------------- #
def setup_logger(name: str = "model_building") -> logging.Logger:
    """Set up and return a configured logger."""
    
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
# ------------------------------------------------ #


def load_params(params_path: str) -> Tuple[float, str, str]:
    """
    Load Logistic Regression hyperparameters from YAML file.
    """

    logger.info("Loading parameters from: %s", params_path)

    params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(f"{params_path} does not exist")

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    try:
        c_value = params["model_building"]["C"]
        solver = params["model_building"]["solver"]
        penalty = params["model_building"]["penalty"]

    except KeyError as e:
        logger.error("Missing key in YAML file: %s", e)
        raise

    logger.debug(
        "Loaded parameters -> C=%s, solver=%s, penalty=%s",
        c_value,
        solver,
        penalty,
    )

    return c_value, solver, penalty


def load_data(train_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data.
    """

    logger.info("Loading training data from: %s", train_path)

    train_df = pd.read_csv(train_path)

    if train_df.shape[1] < 2:
        raise ValueError(
            f"Training data must have at least 2 columns, "
            f"got: {train_df.shape[1]}"
        )

    if train_df.isnull().any().any():
        null_count = train_df.isnull().sum().sum()

        logger.warning(
            "%d NaN value(s) found in training data. Filling with 0.",
            null_count,
        )

        train_df = train_df.fillna(0)

    if "label" not in train_df.columns:
        raise ValueError("Expected 'label' column in training data")

    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values

    logger.debug(
        "Training data loaded. Features shape: %s, Labels shape: %s",
        X_train.shape,
        y_train.shape,
    )

    return X_train, y_train


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    solver: str,
    penalty: str,
) -> LogisticRegression:
    """
    Train Logistic Regression model.
    """

    logger.info(
        "Training Logistic Regression model -> C=%s, solver=%s, penalty=%s",
        c_value,
        solver,
        penalty,
    )

    clf = LogisticRegression(
        C=c_value,
        solver=solver,
        penalty=penalty,
        random_state=42,
        max_iter=1000,
    )

    clf.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, clf.predict(X_train))

    logger.info(
        "Training completed successfully. Train Accuracy: %.4f",
        train_accuracy,
    )

    return clf


def save_model(clf: LogisticRegression, model_path: str) -> None:
    """
    Save trained model.
    """

    model_dir = os.path.dirname(model_path)

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    logger.info("Model saved to: %s", model_path)


def main() -> None:
    """
    Run complete model building pipeline.
    """

    logger.info("Starting model building pipeline...")

    try:
        project_root = Path(__file__).resolve().parents[2]

        params_path = project_root / "params.yaml"

        c_value, solver, penalty = load_params(str(params_path))

        X_train, y_train = load_data(
            train_path=os.path.join(
                "data",
                "feature",
                "train_bow.csv",
            )
        )

        clf = train_model(
            X_train,
            y_train,
            c_value,
            solver,
            penalty,
        )

        save_model(
            clf,
            model_path=os.path.join(
                "models",
                "model.pkl",
            ),
        )

        logger.info("Pipeline completed successfully.")

    except FileNotFoundError as e:
        logger.error("Input file not found: %s", e)
        sys.exit(1)

    except ValueError as e:
        logger.error("Validation error: %s", e)
        sys.exit(1)

    except Exception:
        logger.exception(
            "Pipeline failed due to an unexpected error."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()