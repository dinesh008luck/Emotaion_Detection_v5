import os
import json
import logging
import sys
import pickle

import dagshub
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient


# ---------------- DAGSHUB + MLFLOW ---------------- #

dagshub.init(
    repo_owner="dinesh008luck",
    repo_name="Emotaion_Detection_v5",
    mlflow=True,
)

mlflow.set_tracking_uri(
    "https://dagshub.com/dinesh008luck/Emotaion_Detection_v5.mlflow"
)

mlflow.set_experiment(
    "emotion_detection_registration"
)

# ---------------- LOGGING ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

MODEL_NAME = "emotion_detection_logistic_regression"


def load_metrics(metrics_path: str):

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics


def load_model(model_path: str):

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def register_model():

    try:

        logger.info("Starting model registration...")

        metrics = load_metrics(
            "reports/metrics.json"
        )

        model = load_model(
            "models/model.pkl"
        )

        f1_score = metrics.get("f1_score")

        if f1_score is None:
            raise ValueError(
                "f1_score not found in metrics.json"
            )

        logger.info(
            "Current F1 Score: %.4f",
            f1_score,
        )

        # ---------------- REGISTRATION CONDITION ---------------- #

        if f1_score >= 0.80:

            with mlflow.start_run(
                run_name="model_registration"
            ):

                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME,
                )

                mlflow.log_metrics(metrics)

                logger.info(
                    "Model registered successfully."
                )

                logger.info(
                    "Model URI: %s",
                    model_info.model_uri,
                )

            # ---------------- CLIENT ---------------- #

            client = MlflowClient()

            # ---------------- GET LATEST VERSION ---------------- #

            latest_versions = client.search_model_versions(
                f"name='{MODEL_NAME}'"
            )

            latest_version = max(
                int(v.version)
                for v in latest_versions
            )

            logger.info(
                "Latest model version: %s",
                latest_version,
            )

            # ---------------- MOVE TO STAGING ---------------- #

            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest_version,
                stage="Staging",
                archive_existing_versions=True,
            )

            logger.info(
                "Model version %s moved to STAGING.",
                latest_version,
            )
            mlflow.set_tags({
                "model_type": "logistic_regression",
                "project": "emotion_detection",
                "vectorizer": "LogisticRegression",
                "developer": "Dinesh"
            })
        else:

            logger.warning(
                "Model not registered. "
                "F1 score below threshold."
            )

    except Exception:

        logger.exception(
            "Model registration failed."
        )

        sys.exit(1)


if __name__ == "__main__":
    register_model()