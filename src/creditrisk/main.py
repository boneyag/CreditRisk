import argparse
from pathlib import Path

from creditrisk import data
from creditrisk import preprocess
from creditrisk import train
from creditrisk import evaluate

def main(
    dataset_name,
    model_name,
    persist,
    dataset_version=None,
    training_data_policy="initial",
    feature_schema_version="v1",
    notes="",
):
    df = data.load_typecast_data(dataset_name)

    if df is None:
        raise FileNotFoundError(
            f"Training dataset could not be loaded: {dataset_name}. "
            f"Resolved data directory: {data.DATA_DIR}"
        )

    X, y = preprocess.data_target_split(df)
    X = preprocess.feature_engineering(X)
    X_train, X_test, y_train, y_test = preprocess.partition_data(X, y)
    preprocessor = preprocess.preprocessing_pipeline(X_train)
    resolved_dataset_version = dataset_version or Path(dataset_name).stem
    clf = train.train_classifier(
        model_name=model_name,
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        persist=persist,
        manifest_metadata={
            "dataset_version": resolved_dataset_version,
            "training_data_policy": training_data_policy,
            "feature_schema_version": feature_schema_version,
            "notes": notes,
        },
    )
    eval_metrics = evaluate.eval_classifier(model_name, X_test, y_test, clf)

    print(f"{model_name} evaluation metrics:\n{eval_metrics}")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser("Credit risk training pipeline")
    arg_parser.add_argument("-d", "--data", type=str, default="data_v1.csv")
    arg_parser.add_argument("-m", "--model", type=str, choices=["lr", "svc", "xgb"], default="xgb")
    arg_parser.add_argument("--no-persist", action="store_true", help="Disable artifact persistence")
    arg_parser.add_argument("--dataset-version", type=str)
    arg_parser.add_argument(
        "--training-data-policy",
        type=str,
        choices=["initial", "combined", "new_only"],
    )
    arg_parser.add_argument("--feature-schema-version", type=str)
    arg_parser.add_argument("--notes", type=str)

    args = arg_parser.parse_args()
    main(
        dataset_name=args.data,
        model_name=args.model,
        persist=not args.no_persist,
        dataset_version=args.dataset_version or None,
        training_data_policy=args.training_data_policy or "initial",
        feature_schema_version=args.feature_schema_version or "v1",
        notes=args.notes or "",
    )