import os
import pandas as pd
from pathlib import Path
from creditrisk.logger import setup_logger

logger = setup_logger(__name__)

def _resolve_data_dir() -> Path:
    env_dir = os.getenv("CREDITRISK_DATA_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path.cwd() / "data")
    candidates.append(Path(__file__).resolve().parent.parent.parent / "data")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback to CWD/data to keep local runs predictable.
    return Path.cwd() / "data"


DATA_DIR = _resolve_data_dir()

def load_typecast_data(dataset_name) -> pd.DataFrame | None:
    dataset_path = Path(dataset_name)
    if not dataset_path.is_absolute() and dataset_path.parent == Path('.'):
        dataset_path = DATA_DIR / dataset_path

    try:
        df = pd.read_csv(dataset_path)

        if len(df) != 50000:
            logger.warning(f"The original file should have 50000 records. Found {len(df)}")
        
        logger.info("File is loaded")

        int_cols = df.select_dtypes(include=['int64']).columns
        float_cols = df.select_dtypes(include=['float64']).columns
        str_cols = df.select_dtypes(include=['string', 'str']).columns
        obj_cols = df.columns[df.dtypes == object]

        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in set(str_cols).union(obj_cols):
            df[col] = df[col].astype('category')

        logger.info("Applied dtype downcasting and categorical conversion")

        return df
    except FileNotFoundError:
        logger.error(f"Could not find dataset at {dataset_path}")
        return None
    