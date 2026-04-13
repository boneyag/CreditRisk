import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from creditrisk.logger import setup_logger

logger = setup_logger(__name__)

def _resolve_artifacts_dir() -> Path:
    env_dir = os.getenv("CREDITRISK_ARTIFACTS_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path.cwd() / "artifacts")
    candidates.append(Path(__file__).resolve().parent.parent.parent / "artifacts")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback to CWD/artifacts to keep local runs predictable.
    return Path.cwd() / "artifacts"


ARTIFACTS_DIR = _resolve_artifacts_dir()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

DEFAULT_MANIFEST = {
    "active_version": None,
    "versions": {},
}

def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        save_manifest(DEFAULT_MANIFEST)
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Keep backward compatibility if schema is partially missing.
    manifest.setdefault("active_version", None)
    manifest.setdefault("versions", {})
    return manifest


def save_manifest(manifest: dict) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_version(dataset_version: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{dataset_version}"


def get_active_model_metadata(model_name: str | None = None) -> dict[str, Any]:
    manifest = load_manifest()
    active_version = manifest.get("active_version")
    if not active_version:
        raise FileNotFoundError("No active model version in manifest")

    versions = manifest.get("versions", {})
    if active_version not in versions:
        raise FileNotFoundError(f"Active version not found in manifest: {active_version}")

    metadata = versions[active_version]
    if model_name and metadata.get("model_name") != model_name:
        raise FileNotFoundError(
            f"Active version {active_version} does not match requested model {model_name}"
        )

    return metadata


def get_active_model_version(model_name: str | None = None) -> str:
    # version is indexed by key in manifest; expose from active pointer.
    manifest = load_manifest()
    return manifest["active_version"]

def save_pipeline(
    model_pipeline,
    model_name: str,
    dataset_version: str,
    training_data_policy: str = "initial",
    feature_schema_version: str = "v1",
    metrics: dict[str, float] | None = None,
    notes: str = "",
    version: str | None = None,
    activate: bool = True,
) -> Path:
    """Save the full fitted sklearn pipeline (preprocessor + classifier)."""
    model_version = version or _default_version(dataset_version)
    artifact_name = f"{model_name}_{model_version}_pipeline.joblib"
    path = ARTIFACTS_DIR / artifact_name

    joblib.dump(model_pipeline, path)
    logger.info("Saved pipeline artifact to %s", path)

    manifest = load_manifest()
    manifest["versions"][model_version] = {
        "model_name": model_name,
        "artifact_name": artifact_name,
        "created_at": _utc_now_iso(),
        "training_data_policy": training_data_policy,
        "dataset_version": dataset_version,
        "feature_schema_version": feature_schema_version,
        "metrics": metrics or {},
        "notes": notes,
    }
    if activate:
        manifest["active_version"] = model_version
    save_manifest(manifest)

    return path


def load_pipeline(model_name: str):
    """Load the full fitted sklearn pipeline for inference/evaluation."""
    # Prefer active model from manifest; fallback to legacy unversioned artifact.
    try:
        metadata = get_active_model_metadata(model_name=model_name)
        path = ARTIFACTS_DIR / metadata["artifact_name"]
    except FileNotFoundError:
        path = ARTIFACTS_DIR / f"{model_name}_pipeline.joblib"

    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    logger.info("Loaded pipeline artifact from %s", path)
    return joblib.load(path)
