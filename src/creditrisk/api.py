from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from creditrisk.artifacts import get_active_model_metadata, get_active_model_version, load_pipeline
from creditrisk.explain import explain_prediction
from creditrisk.preprocess import feature_engineering
from creditrisk.logger import setup_logger

logger = setup_logger(__name__)


model_pipeline = None
MODEL_NAME = "xgb"
ACTIVE_MODEL_VERSION = None

class PredictionRequest(BaseModel):
    age: int
    occupation_status: str
    years_employed: int
    annual_income: float
    credit_score: float
    credit_history_years: int
    savings_assets: float
    current_debt: float
    defaults_on_file: int
    delinquencies_last_2yrs: int
    derogatory_marks: int
    product_type: str
    loan_intent: str
    loan_amount: float
    interest_rate: float
    debt_to_income_ratio: float
    loan_to_income_ratio: float
    payment_to_income_ratio: float

class PredictionResponse(BaseModel):
    pred: int
    proba: float
    model_version: str


class ExplanationItem(BaseModel):
    feature: str
    shap_value: float
    abs_shap_value: float


class TransformedExplanationItem(ExplanationItem):
    pass


class ExplanationResponse(PredictionResponse):
    expected_value: float
    feature_explanations: list[ExplanationItem]
    transformed_feature_explanations: list[TransformedExplanationItem]
    transformed_feature_count: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, ACTIVE_MODEL_VERSION
    logger.info("Loading the pipeline...")
    try:
        try:
            ACTIVE_MODEL_VERSION = get_active_model_version(model_name=MODEL_NAME)
        except FileNotFoundError:
            ACTIVE_MODEL_VERSION = "legacy"
        model_pipeline = load_pipeline(MODEL_NAME)
    except FileNotFoundError as exc:
        logger.error(f"Could not find {MODEL_NAME}_pipeline.joblib in artifacts")
        raise RuntimeError(f"Could not load model artifact for {MODEL_NAME}: {exc}") from exc
    logger.info("Application is starting up...")
    yield
    model_pipeline = None
    logger.info("Application is shutting down...")

app = FastAPI(title="CreditRisk API", lifespan=lifespan)

@app.get("/model_info")
def model_info():
    metadata = {}
    try:
        metadata = get_active_model_metadata(model_name=MODEL_NAME)
    except FileNotFoundError:
        pass

    return {
        "model_name": MODEL_NAME,
        "version": ACTIVE_MODEL_VERSION,
        "artifact_name": metadata.get("artifact_name"),
        "dataset_version": metadata.get("dataset_version"),
        "training_data_policy": metadata.get("training_data_policy"),
        "feature_schema_version": metadata.get("feature_schema_version"),
    }

@app.get("/health")
def model_health():
    return {
        "model": MODEL_NAME,
        "health": "live" if model_pipeline is not None else "not loaded"
    }

@app.post("/predict", response_model=PredictionResponse)
def model_predict(payload: PredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    row = payload.model_dump()
    df = pd.DataFrame([row])
    df = feature_engineering(df)

    pred_raw = model_pipeline.predict(df)[0]
    pred = int(pred_raw)

    if hasattr(model_pipeline, "predict_proba"):
        proba = float(model_pipeline.predict_proba(df)[0][1])
    elif hasattr(model_pipeline, "decision_function"):
        proba = float(model_pipeline.decision_function(df)[0])
    else:
        proba = float(pred)

    return {"pred": pred, "proba": proba, "model_version": ACTIVE_MODEL_VERSION}


@app.post("/explain", response_model=ExplanationResponse)
def model_explain(payload: PredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = payload.model_dump()
    df = pd.DataFrame([row])
    df = feature_engineering(df)

    pred_raw = model_pipeline.predict(df)[0]
    pred = int(pred_raw)

    if hasattr(model_pipeline, "predict_proba"):
        proba = float(model_pipeline.predict_proba(df)[0][1])
    elif hasattr(model_pipeline, "decision_function"):
        proba = float(model_pipeline.decision_function(df)[0])
    else:
        proba = float(pred)

    try:
        explanation = explain_prediction(model_pipeline, row)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "pred": pred,
        "proba": proba,
        "model_version": ACTIVE_MODEL_VERSION,
        **explanation,
    }