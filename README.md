# AIMLCZG523 - MLOps Assignment 01
## Heart Disease Risk Prediction - End-to-End MLOps Project

This project implements an end-to-end MLOps pipeline for predicting heart disease risk using the Heart Disease UCI Dataset.

## Project Scope

The project covers:
1. Data acquisition and EDA
2. Data preprocessing and feature engineering
3. Model training and comparison
4. MLflow experiment tracking
5. Model packaging and reproducibility
6. FastAPI model serving API
7. Unit testing with Pytest
8. Docker containerization
9. GitHub Actions CI/CD pipeline
10. Kubernetes deployment manifests
11. API logging and basic monitoring setup
12. Final report and video demonstration support

## Recommended Local Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Step 1 - Download Dataset

```bash
python scripts/download_data.py
```

This creates:

```text
data/raw/heart_disease_uci.csv
```

## Step 2 - Run EDA

```bash
python src/eda.py
```

EDA charts will be saved to:

```text
artifacts/plots/
```

## Step 3 - Train Models with MLflow

```bash
python src/train.py
```

This trains Logistic Regression and Random Forest models, performs basic hyperparameter tuning, logs metrics/artifacts to MLflow, and saves the best model.

## Step 4 - Start MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

## Step 5 - Run API Locally

```bash
uvicorn src.api:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Step 6 - Test Prediction API

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d @sample_input.json
```

## Step 7 - Run Tests

```bash
pytest -v
```

## Step 8 - Build and Run Docker

```bash
docker build -t heart-disease-mlops-api .
docker run -p 8000:8000 heart-disease-mlops-api
```

## Step 9 - Deploy to Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc
```

## Repository Structure

```text
.
├── .github/workflows/mlops-ci.yml
├── artifacts/
├── data/
├── docs/
├── k8s/
├── monitoring/
├── notebooks/
├── scripts/
├── src/
├── tests/
├── Dockerfile
├── requirements.txt
├── sample_input.json
└── README.md
```
