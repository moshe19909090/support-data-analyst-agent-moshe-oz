from functools import lru_cache

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


@lru_cache(maxsize=1)
def load_support_dataset() -> pd.DataFrame:
    """
    Load the Bitext customer-support dataset directly from Hugging Face
    and keep it cached in memory for the current Python process.
    """
    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()

    # Normalize column names just in case.
    df.columns = [str(col).strip().lower() for col in df.columns]

    required_columns = {"instruction", "response", "category", "intent"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Make text fields safe and consistent.
    for col in ["instruction", "response", "category", "intent"]:
        df[col] = df[col].astype(str).str.strip()

    return df