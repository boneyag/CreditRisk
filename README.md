# 💳 Credit Risk Decision System: End-to-End ML Service

An industry-grade machine learning system designed to automate credit risk adjudication while maintaining high auditability and statistical rigor. This project features a modular Python architecture, automated training pipelines, and a production-ready REST API.

## 🎯 Project Goals & SLOs

* Scientific Decisioning: Evaluate the statistical significance of model improvements (XGBoost vs. Baseline) to ensure deployment is justified.
* Auditability: Maintain an immutable record of model versions, training policies, and feature schemas.
* Risk-Centric Optimization: Model performance is optimized for the detection of high-risk applicants to minimize potential defaults.

## 📊 Statistical Validation & Model Comparison

Before moving to production, we performed a comparative analysis between the baseline (Logistic Regression) and the challenger (XGBoost) to ensure the performance gain was not due to random noise.

| Metric   | Logistic Regression | XGBoost | Delta |
|----------|---------------------|---------|-------|
| Accuracy | 0.8897              | 0.9266  | +3.7% |
| ROC-AUC  | 0.9626              | 0.9836  | +0.21 |
| F1-Score | 0.88                | 0.92    | +4.5% |

*Note: F1-score is reported for Class 0 (Loan Rejected). In a credit risk context, we prioritize the precision and recall of high-risk identifications to minimize financial exposure.*

#### Rigor Checks (Summary from notebooks/)
* McNemar’s Test: A McNemar’s test on the paired model errors yielded a $p$-value of $4.1463e-38$. This confirms that the error distributions differ significantly at the $\alpha = 0.05$ level.
* Bootstrap Analysis: Conducted $10,000$ bootstrap iterations to calculate 95% Confidence Intervals for the accuracy. he CI of the difference in accuracy between XGBoost and Logistic Regression is $[0.0315, 0.0423]$, with an effect size of $3.67\%$. Since the interval does not contain zero, the improvement is statistically robust.

## 🏗 System Architecture & Observability
1. Data & Storage
Retrieval: `data.py` implements memory-optimized type casting and schema validation for CSV/SQL sources.

Artifacts: Fully fitted `sklearn` pipelines (preprocessor + classifier) are serialized with metadata manifests to `artifacts/`.

2. The Training Pipeline (main.py)
Decoupled training logic allows for high-velocity experimentation:

```bash
# Example: Retraining due to data drift
uv run python -m creditrisk.main --model xgb --dataset-version v1 --training-data-policy initial --notes "Adjusting for drift"
```

3. Reliability & Evaluation
API Tests: Automated tests in `tests/test_api.py` validate endpoint responses and input edge cases.

Observability: Structured JSON logging in `api.py` provides an audit trail for every prediction, including the model version and probability score.

## 🚀 Deployment (Docker & Cloud Ready)
This project is built to eliminate "it works on my machine" issues. The Docker Compose setup orchestrates a sequential workflow:

Pipeline Container: Trains the model and persists the artifact.

API Container: Spins up only after successful training to serve the latest artifact.

```bash
# Build and run the full stack
docker compose up --build
```

## 🛣 Future Roadmap & Risk Mitigation
Cloud Migration: Transitioning compute to AWS Fargate and storage to S3 for higher scalability.

Observability: Integrating Prometheus metrics to track real-time drift in the debt_to_income_ratio feature.

Explainability: Incorporating SHAP values directly into the /predict response for adjudicator transparency.

Production Reliability: Calculate inference latency ($p95$) with 100% environment parity via Docker.

## 🛠 Tech Stack
Core: Python 3.11, uv, scikit-learn, XGBoost.

Serving: FastAPI, Uvicorn, Pydantic.

DevOps: Docker, Docker Compose, Pytest.