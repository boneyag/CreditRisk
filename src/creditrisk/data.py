import pandas as pd
from pathlib import Path
from creditrisk.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / 'data'

def load_typecast_data(file_name="Loan_approval_data_2025.csv") -> pd.DataFrame | None:
    try:
        df = pd.read_csv(DATA_DIR / file_name)

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
        logger.error(f"Could not find {file_name}")
        return None
    