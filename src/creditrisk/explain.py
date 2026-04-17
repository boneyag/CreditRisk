from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from creditrisk.preprocess import feature_engineering


@dataclass(frozen=True)
class FeatureExplanation:
    feature: str
    shap_value: float
    abs_shap_value: float


def _as_dataframe(payload: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    return pd.DataFrame([payload])


def _select_classifier(model_pipeline):
    if not hasattr(model_pipeline, "named_steps"):
        raise TypeError("Expected a fitted sklearn Pipeline with named_steps")
    return model_pipeline.named_steps["classifier"]


def _select_preprocessor(model_pipeline):
    if not hasattr(model_pipeline, "named_steps"):
        raise TypeError("Expected a fitted sklearn Pipeline with named_steps")
    return model_pipeline.named_steps["preprocessor"]


def _build_feature_groups(preprocessor) -> list[str]:
    groups: list[str] = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue

        column_names = list(columns)
        if transformer_name == "numerical":
            groups.extend(column_names)
            continue

        if transformer_name != "categorical":
            groups.extend(column_names)
            continue

        encoder = transformer.named_steps.get("encoder") if hasattr(transformer, "named_steps") else transformer
        encoded_categories = getattr(encoder, "categories_", None)
        if encoded_categories is None:
            raise RuntimeError("Categorical encoder is not fitted")

        for column_name, categories in zip(column_names, encoded_categories, strict=True):
            groups.extend([column_name] * len(categories))

    return groups


def _build_transformed_feature_names(preprocessor) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass

    names: list[str] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue

        column_names = list(columns)
        if transformer_name == "numerical":
            names.extend(column_names)
            continue

        if transformer_name != "categorical":
            names.extend(column_names)
            continue

        encoder = transformer.named_steps.get("encoder") if hasattr(transformer, "named_steps") else transformer
        encoded_categories = getattr(encoder, "categories_", None)
        if encoded_categories is None:
            raise RuntimeError("Categorical encoder is not fitted")

        for column_name, categories in zip(column_names, encoded_categories, strict=True):
            names.extend([f"{column_name}_{category}" for category in categories])

    return names


def _to_dense_matrix(transformed):
    if hasattr(transformed, "toarray"):
        return transformed.toarray()
    return np.asarray(transformed)


def _extract_shap_row(shap_values, row_index: int = 0) -> np.ndarray:
    values = shap_values

    if isinstance(values, list):
        values = values[-1]

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[row_index]
    elif values.ndim == 2:
        values = values[row_index]
    elif values.ndim != 1:
        raise RuntimeError(f"Unsupported SHAP value shape: {values.shape}")

    return values.astype(float, copy=False)


def _group_shap_values(shap_row: np.ndarray, feature_groups: list[str]) -> list[FeatureExplanation]:
    if len(shap_row) != len(feature_groups):
        raise RuntimeError(
            f"SHAP value length {len(shap_row)} does not match feature groups {len(feature_groups)}"
        )

    grouped_values: dict[str, float] = defaultdict(float)
    for feature_name, shap_value in zip(feature_groups, shap_row, strict=True):
        grouped_values[feature_name] += float(shap_value)

    explanations = [
        FeatureExplanation(
            feature=feature_name,
            shap_value=value,
            abs_shap_value=abs(value),
        )
        for feature_name, value in grouped_values.items()
    ]

    return sorted(explanations, key=lambda item: item.abs_shap_value, reverse=True)


def _raw_shap_values(shap_row: np.ndarray, feature_names: list[str]) -> list[FeatureExplanation]:
    if len(shap_row) != len(feature_names):
        raise RuntimeError(
            f"SHAP value length {len(shap_row)} does not match transformed features {len(feature_names)}"
        )

    explanations = [
        FeatureExplanation(
            feature=feature_name,
            shap_value=float(shap_value),
            abs_shap_value=abs(float(shap_value)),
        )
        for feature_name, shap_value in zip(feature_names, shap_row, strict=True)
    ]

    return sorted(explanations, key=lambda item: item.abs_shap_value, reverse=True)


def explain_prediction(model_pipeline, payload: pd.DataFrame | dict[str, Any]) -> dict[str, Any]:
    """Explain a single prediction using SHAP on the fitted tree classifier."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("shap is required for prediction explanations") from exc

    input_frame = _as_dataframe(payload)
    engineered_frame = feature_engineering(input_frame)

    preprocessor = _select_preprocessor(model_pipeline)
    classifier = _select_classifier(model_pipeline)

    transformed = preprocessor.transform(engineered_frame)
    transformed_dense = _to_dense_matrix(transformed)

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed_dense)
        expected_value = explainer.expected_value
    except Exception as exc:  # pragma: no cover - defensive wrapper around shap internals
        raise RuntimeError(f"Unable to compute SHAP explanations: {exc}") from exc

    shap_row = _extract_shap_row(shap_values)
    feature_groups = _build_feature_groups(preprocessor)
    transformed_feature_names = _build_transformed_feature_names(preprocessor)
    explanations = _group_shap_values(shap_row, feature_groups)
    raw_explanations = _raw_shap_values(shap_row, transformed_feature_names)

    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = float(np.asarray(expected_value).reshape(-1)[-1])
    else:
        expected_value = float(expected_value)

    return {
        "expected_value": expected_value,
        "feature_explanations": [item.__dict__ for item in explanations],
        "transformed_feature_explanations": [item.__dict__ for item in raw_explanations],
        "transformed_feature_count": len(feature_groups),
    }
