from creditrisk.logger import setup_logger
from creditrisk.artifacts import load_pipeline

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np

logger = setup_logger(__name__)

MODEL_NAMES = {"lr", "svc", "xgb"}


def eval_classifier(model_name, X_test, y_test, classifier):
    if model_name not in MODEL_NAMES:
        logger.error("Unsupported model: %s. Choose from %s", model_name, ", ".join(sorted(MODEL_NAMES)))
        return None

    if classifier is None:
        clf = load_pipeline(model_name)
    else:
        clf = classifier
    pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        score = clf.decision_function(X_test)
    else:
        # Fallback keeps evaluation functional for classifiers without scoring APIs.
        score = np.asarray(pred, dtype=np.float32)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true=y_test, y_pred=pred),
        "roc_auc": roc_auc_score(y_true=y_test, y_score=score),
        "classification_report": classification_report(y_true=y_test, y_pred=pred),
    }

    logger.info("Evaluated model: %s | accuracy=%.4f | roc_auc=%.4f", model_name, metrics["accuracy"], metrics["roc_auc"])
    return metrics