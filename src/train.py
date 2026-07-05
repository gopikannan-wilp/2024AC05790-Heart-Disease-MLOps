from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from src.data import load_dataset, build_preprocessor, prepare_train_test

MODEL_DIR = Path("artifacts/model")
PLOT_DIR = Path("artifacts/plots")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def save_plots(model, X_test, y_test, model_name):
    cm_path = PLOT_DIR / f"{model_name}_confusion_matrix.png"
    roc_path = PLOT_DIR / f"{model_name}_roc_curve.png"

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    return cm_path, roc_path


def train():
    mlflow.set_experiment("heart-disease-mlops-assignment")

    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    preprocessor = build_preprocessor(df)

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
            },
        ),
    }

    results = {}
    best_model = None
    best_score = -1
    best_name = None

    for model_name, (classifier, params) in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", classifier),
                ]
            )

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=params,
                scoring="roc_auc",
                cv=5,
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)

            tuned_model = grid.best_estimator_
            metrics = evaluate_model(tuned_model, X_test, y_test)
            cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=5, scoring="roc_auc")

            cm_path, roc_path = save_plots(tuned_model, X_test, y_test, model_name)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.sklearn.log_model(tuned_model, artifact_path="model")

            results[model_name] = {
                "best_params": grid.best_params_,
                "metrics": metrics,
                "cv_roc_auc_mean": float(cv_scores.mean()),
            }

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = tuned_model
                best_name = model_name

    joblib.dump(best_model, MODEL_DIR / "heart_disease_pipeline.joblib")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"Best model: {best_name}")
    print(f"Saved model to {MODEL_DIR / 'heart_disease_pipeline.joblib'}")


if __name__ == "__main__":
    train()
