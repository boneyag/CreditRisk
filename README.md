# Credit Risk Modeling

End-to-end credit risk training workflow using a modular Python package under `src/`, with notebooks reserved for exploration and statistical comparison.

## Project Flow

`load data -> type cast -> feature engineering -> train/test split -> preprocessing -> model training -> evaluation -> save artifact -> serve via API`

## Repository Layout

```text
CreditRisk/
	data/
		Loan_approval_data_2025.csv
	notebooks/
		credit_risk.ipynb
		credit_risk2.ipynb
	artifacts/
		<model>_pipeline.joblib
	src/creditrisk/
		__init__.py
		main.py
		data.py
		preprocess.py
		train.py
		evaluate.py
		artifacts.py
		api.py
		logger.py
	pyproject.toml
```

## Modules

- `data.py`: loads CSV, validates row count, applies memory-friendly dtype casting.
- `preprocess.py`: target split, feature engineering, train/test split, and sklearn preprocessing pipeline.
- `train.py`: trains selected model (`lr`, `svc`, `xgb`) using a full sklearn pipeline.
- `evaluate.py`: evaluates a provided classifier or loads one from disk.
- `artifacts.py`: saves/loads full fitted pipelines (`preprocessor + classifier`) via joblib with versioning and manifest metadata.
- `main.py`: orchestrates the full training/evaluation process with CLI argument support for dataset versioning and training policy.
- `api.py`: FastAPI application for model serving with `/health`, `/model_info`, and `/predict` endpoints. Supports model versioning and immutable artifact management.

## Setup

This project uses `uv`.

1. Create/sync environment and dependencies:

```bash
uv sync
```

2. Install the package in editable mode:

```bash
uv pip install -e .
```

## Run Training Pipeline

Train with default settings (XGBoost model, initial data policy):

```bash
PYTHONPATH=src uv run python -m creditrisk.main
```

Train with custom settings for versioning and metadata tracking:

```bash
PYTHONPATH=src uv run python -m creditrisk.main \
  --model xgb \
  --dataset-version v1.1 \
  --training-data-policy combined \
  --feature-schema-version v1 \
  --notes "retrained due to drift alert"
```

### CLI Arguments

- `--data`: Path to training CSV (default: `data/Loan_approval_data_2025.csv`)
- `--model`: Model type: `lr`, `svc`, or `xgb` (default: `xgb`)
- `--dataset-version`: Version identifier for dataset (e.g., `v1.0`, `2026-04-12`)
- `--training-data-policy`: Data selection strategy: `initial`, `combined`, or `new_only` (default: `initial`)
- `--feature-schema-version`: Version of feature engineering logic (default: `v1`)
- `--notes`: Metadata notes (e.g., reason for retraining)
- `--no-persist`: Skip saving the trained pipeline

## Model Selection

Supported model names:

- `lr` (Logistic Regression)
- `svc` (Support Vector Classifier)
- `xgb` (XGBoost)

The current default in `main.py` is `xgb`.

## Artifacts

Training saves a full pipeline artifact to:

```text
artifacts/<model_name>_<datetime>-<dataset_name>_pipeline.joblib
```

This artifact includes both preprocessing and model, making it ready for API inference.

## Evaluation Outputs

`evaluate.py` returns:

- accuracy
- ROC-AUC
- classification report (precision/recall/f1/support)

## Serve via REST API

Start the FastAPI server:

```bash
PYTHONPATH=src uv run uvicorn creditrisk.api:app --reload
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### API Endpoints

**GET /health**
```json
{
  "model": "xgb",
  "health": "live",
  "model_version": "20260413T060654Z-data_v1"
}
```

**GET /model_info**
Returns active model metadata including training policy, feature schema version, metrics, and training notes.

**POST /predict**
Accepts a loan application as JSON and returns prediction:
```json
{
  "pred": 1,
  "proba": 0.92,
  "model_version": "20260413T060654Z-data_v1"
}
```

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "occupation_status": "employed",
    "years_employed": 10,
    "annual_income": 85000.0,
    "credit_score": 720.0,
    "credit_history_years": 12,
    "savings_assets": 15000.0,
    "current_debt": 12000.0,
    "defaults_on_file": 0,
    "delinquencies_last_2yrs": 1,
    "derogatory_marks": 0,
    "product_type": "personal",
    "loan_intent": "debt_consolidation",
    "loan_amount": 15000.0,
    "interest_rate": 0.12,
    "debt_to_income_ratio": 0.18,
    "loan_to_income_ratio": 0.35,
    "payment_to_income_ratio": 0.08,
  }'
```

## Running Tests

Run all tests:

```bash
PYTHONPATH=src uv run pytest
```

Run tests with verbose output:

```bash
PYTHONPATH=src uv run pytest -v
```

Run specific test file:

```bash
PYTHONPATH=src uv run pytest tests/test_api.py
```

## Model Versioning

Each trained pipeline is versioned with an immutable artifact name following the pattern:
```
<model_name>_<YYYY-MM-DD>-<data_fingerprint>_pipeline.joblib
```

Metadata for each version is tracked in `artifacts/manifest.json`, including:
- Training dataset version and policy
- Feature engineering schema version
- Model performance metrics
- Training notes and timestamps
- Active version pointer

When retraining, specify `--dataset-version`, `--training-data-policy`, and `--feature-schema-version` to track versioning metadata automatically.

## Exploration and Statistical Testing

Exploratory analysis and model-comparison rationale are in notebooks under `notebooks/` and summary notes in `model_comparison.md`.

McNemar and bootstrap analyses are best kept in notebooks (or in a dedicated analysis script) rather than the production training path.

