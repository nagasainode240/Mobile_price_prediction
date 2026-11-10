from typing import List
import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and align raw mobile input data before prediction.
    Ensures correct feature columns and order expected by the preprocessor.
    """
    df = df.copy()

    # Drop identifier columns if present
    for col in ["mobile_id", "id", "customerID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop target column if sent in error
    if "price_range" in df.columns:
        df = df.drop(columns=["price_range"])

    # Ensure all required features are present
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Return DataFrame aligned strictly in expected feature order
    return df[FEATURE_COLUMNS].astype(float)
