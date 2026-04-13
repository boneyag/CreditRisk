from creditrisk.logger import setup_logger
from creditrisk.artifacts import save_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = setup_logger(__name__)

MODEL_NAMES = {
    "lr": LogisticRegression,
    "svc": SVC,
    "xgb": XGBClassifier
}

def train_classifier(
    model_name,
    preprocessor,
    X_train,
    y_train,
    persist=True,
    manifest_metadata=None,
):
    if model_name not in MODEL_NAMES:
        logger.error("Unsupported model: %s. Choose from %s", model_name, ", ".join(MODEL_NAMES.keys()))
        return None

    if model_name == "lr":
        estimator = LogisticRegression(max_iter=2000, class_weight='balanced')
    elif model_name == "svc":
        estimator = SVC(
            kernel='rbf',
            C=2.0,
            gamma='scale',
            probability=False,
            class_weight='balanced'
        )
    else:
        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            colsample_bytree=0.8,
            subsample=0.8,
            eval_metric='logloss',
            objective='binary:logistic',
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        )

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])

    clf.fit(X_train, y_train)
    logger.info("Trained model: %s", model_name)

    if persist:
        metadata = manifest_metadata or {}
        dataset_version = metadata.get("dataset_version", "unknown")
        save_pipeline(
            clf,
            model_name=model_name,
            dataset_version=dataset_version,
            training_data_policy=metadata.get("training_data_policy", "initial"),
            feature_schema_version=metadata.get("feature_schema_version", "v1"),
            metrics=metadata.get("metrics"),
            notes=metadata.get("notes", ""),
            version=metadata.get("version"),
            activate=metadata.get("activate", True),
        )

    return clf