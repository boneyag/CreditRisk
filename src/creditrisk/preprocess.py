from creditrisk.logger import setup_logger

import numpy as np
import os
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = setup_logger(__name__)

def _resolve_data_dir() -> Path:
    env_dir = os.getenv("CREDITRISK_DATA_DIR")
    if env_dir:
        env_path = Path(env_dir)
        if not env_path.exists():
            raise FileNotFoundError(
                f"CREDITRISK_DATA_DIR is set but does not exist: {env_path}"
            )
        return env_path

    candidates = []
    candidates.append(Path(__file__).resolve().parent.parent.parent / "data")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path.cwd() / "data")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback to CWD/data to keep local runs predictable.
    return Path.cwd() / "data"

def data_target_split(df):
    y = df['loan_status']
    X = df.drop(['loan_status', 'customer_id'], axis=1)

    return X, y

def partition_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    data_path = _resolve_data_dir()
    data_path.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(data_path / 'x_train.csv')
    X_test.to_csv(data_path / 'x_test.csv')
    y_train.to_csv(data_path / 'y_train.csv')
    y_test.to_csv(data_path / 'y_test.csv')
    logger.info(f"Saved train/test datasets for later use.")
    
    return X_train, X_test, y_train, y_test

def feature_engineering(X):
    X_ext = X.copy()
    X_ext['income_after_debt'] = X['annual_income'] - X['current_debt']
    X_ext['debt_x_dti'] = X['current_debt'] * X['debt_to_income_ratio']
    X_ext['rate_x_lti'] = X['interest_rate'] * X['loan_to_income_ratio']

    return X_ext

def preprocessing_pipeline(X):
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numerical_features]

    numerical_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32))
    ])

    full_preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_preprocessor, numerical_features),
            ('categorical', categorical_preprocessor, categorical_features),
        ]
    )

    logger.info(
        "Built preprocessing pipeline with %d numeric and %d categorical features",
        len(numerical_features),
        len(categorical_features),
    )

    return full_preprocessor