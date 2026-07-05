# API Testing Steps

## Start API

```bash
uvicorn src.api:app --reload
```

## Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

## Endpoints to Test

1. GET `/health`
2. GET `/model-info`
3. GET `/features`
4. GET `/metrics`
5. POST `/predict`

## Sample Predict Request

Use `sample_input.json` or paste the same payload in Swagger UI.

## Required Screenshots

Capture:
- Swagger UI screen
- Health API response
- Model info response
- Metrics response
- Features response
- Predict request and response
- Docker build/run terminal
- Kubernetes pod/service output
- GitHub Actions workflow run
- MLflow UI runs and artifacts
