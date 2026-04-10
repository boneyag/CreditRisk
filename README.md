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
- `artifacts.py`: saves/loads full fitted pipelines (`preprocessor + classifier`) via joblib.
- `main.py`: orchestrates the full training/evaluation process.
- `api.py`: *TODO* -- API entrypoint placeholder for deployment.

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

Run as a module (recommended):

```bash
uv run python -m creditrisk.main
```

If needed in environments where package resolution is not active yet:

```bash
PYTHONPATH=src uv run python -m creditrisk.main
```

## Model Selection

Supported model names:

- `lr` (Logistic Regression)
- `svc` (Support Vector Classifier)
- `xgb` (XGBoost)

The current default in `main.py` is `xgb`.

## Artifacts

Training saves a full pipeline artifact to:

```text
artifacts/<model_name>_pipeline.joblib
```

This artifact includes both preprocessing and model, making it ready for API inference.

## Evaluation Outputs

`evaluate.py` returns:

- accuracy
- ROC-AUC
- classification report (precision/recall/f1/support)

## Exploration and Statistical Testing

Exploratory analysis and model-comparison rationale are in notebooks under `notebooks/` and summary notes in `model_comparison.md`.

McNemar and bootstrap analyses are best kept in notebooks (or in a dedicated analysis script) rather than the production training path.

## Next Step: API

`api.py` is intentionally left as TODO.

Recommended API pattern:

1. Load `<model>_pipeline.joblib` once at startup.
2. Accept raw feature payloads.
3. Predict via pipeline directly (preprocessing is included).
4. Return class + score (`predict_proba` or decision score).

