from pathlib import Path
import joblib

from src.creditrisk.logger import setup_logger

logger = setup_logger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def save_pipeline(model_pipeline, model_name: str) -> Path:
    """Save the full fitted sklearn pipeline (preprocessor + classifier)."""
    path = ARTIFACTS_DIR / f"{model_name}_pipeline.joblib"
    joblib.dump(model_pipeline, path)
    logger.info("Saved pipeline artifact to %s", path)
    return path


def load_pipeline(model_name: str):
    """Load the full fitted sklearn pipeline for inference/evaluation."""
    path = ARTIFACTS_DIR / f"{model_name}_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    logger.info("Loaded pipeline artifact from %s", path)
    return joblib.load(path)
