Emotaion_Detection_v5
# Emotion Detection MLOps Pipeline

## Project Overview

This project demonstrates an end-to-end MLOps pipeline for an NLP-based Emotion Detection system using modern machine learning engineering practices.

The pipeline covers:

* Data preprocessing
* Text feature engineering
* Model training
* Experiment tracking
* Pipeline orchestration
* Model evaluation
* Model registry and staging
* Reproducibility with DVC
* Remote experiment tracking using DagsHub + MLflow

The goal of this project is not only to train a machine learning model, but also to productionize the complete ML workflow.

---

# Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* DVC
* MLflow
* DagsHub
* Git
* YAML

---

# Project Structure

```bash
Emotaion_Detection_v5/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── feature/
│
├── models/
│   └── model.pkl
│
├── reports/
│   ├── confusion_matrix.npy
│   └── metrics.json
│
├── src/
│   ├── data/
│   │   ├── data_preprocess.py
│   │   └── make_dataset.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── model_building.py
│   │   ├── model_evaluation.py
│   │   └── model_registration.py
│   │
│   └── visualization/
│       └── visualize.py
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```

---

# Workflow Pipeline

## 1. Data Preprocessing

* Clean raw text data
* Handle null values
* Train-test split
* Save processed datasets

### Run

```bash
python src/data/data_preprocess.py
```

---

## 2. Dataset Preparation

* Load train/test datasets
* Additional preprocessing
* Save processed CSV files

### Run

```bash
python src/data/make_dataset.py
```

---

## 3. Feature Engineering

Text vectorization using:

* Bag of Words (BoW)
* TF-IDF

### Run

```bash
python src/features/build_features.py
```

---

## 4. Model Building

Model used:

* Logistic Regression

Best parameters:

```python
LogisticRegression(
    C=1,
    solver='liblinear',
    penalty='l2'
)
```

### Run

```bash
python src/models/model_building.py
```

---

## 5. Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

### Run

```bash
python -m src.models.model_evaluation
```

---

## 6. Experiment Tracking

Implemented using:

* MLflow
* DagsHub

Tracked items:

* Parameters
* Metrics
* Models
* Artifacts
* Confusion Matrix

---

# DVC Pipeline

Pipeline stages are managed using DVC.

### Run Complete Pipeline

```bash
dvc repro
```

### View Pipeline

```bash
dvc dag
```

---

# MLflow Model Registry

Implemented:

* Model registration
* Versioning
* Staging transition

### Model Registration

```bash
python -m src.models.model_registration
```

---

# DagsHub Integration

Remote experiment tracking and artifact storage configured using DagsHub.

Features:

* Centralized experiment tracking
* Remote artifact storage
* MLflow UI integration
* Model version tracking

---

# Current Achievements

Completed:

* End-to-end MLOps pipeline
* DVC integration
* MLflow tracking
* DagsHub setup
* Model Registry
* Staging transition
* Metrics logging
* Artifact logging
* Reproducible ML workflow

---

# Future Improvements

Planned next steps:

* Hyperparameter tuning
* Dockerization
* FastAPI deployment
* CI/CD integration
* Cloud deployment
* Monitoring pipeline
* Drift detection
* Automated retraining

---

# How to Run

## Clone Repository

```bash
git clone <your-repo-url>
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Run Full Pipeline

```bash
dvc repro
```

---

# Learning Outcome

This project helped in understanding:

* Production-grade ML workflows
* Pipeline reproducibility
* Experiment tracking
* Model lifecycle management
* MLOps best practices
* End-to-end machine learning engineering

---
Machine Learning & MLOps Enthusiast

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Author

Dinesh Kumar
