import os
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema


def get_mobile_schema(include_target: bool = True) -> DataFrameSchema:
    """Return Pandera DataFrameSchema for mobile phone features validation."""
    schema_dict = {
        "battery_power": Column(pa.Int, Check.in_range(500, 2500), nullable=False),
        "blue": Column(pa.Int, Check.isin([0, 1]), nullable=False),
        "clock_speed": Column(pa.Float, Check.in_range(0.4, 3.5), nullable=False),
        "dual_sim": Column(pa.Int, Check.isin([0, 1]), nullable=False),
        "fc": Column(pa.Int, Check.in_range(0, 25), nullable=False),
        "four_g": Column(pa.Int, Check.isin([0, 1]), nullable=False),
        "int_memory": Column(pa.Int, Check.in_range(2, 512), nullable=False),
        "m_dep": Column(pa.Float, Check.in_range(0.05, 1.5), nullable=False),
        "mobile_wt": Column(pa.Int, Check.in_range(50, 300), nullable=False),
        "n_cores": Column(pa.Int, Check.in_range(1, 16), nullable=False),
        "pc": Column(pa.Int, Check.in_range(0, 50), nullable=False),
        "px_height": Column(pa.Int, Check.in_range(0, 3000), nullable=False),
        "px_width": Column(pa.Int, Check.in_range(400, 3000), nullable=False),
        "ram": Column(pa.Int, Check.in_range(256, 16384), nullable=False),
        "sc_h": Column(pa.Int, Check.in_range(4, 30), nullable=False),
        "sc_w": Column(pa.Int, Check.in_range(0, 25), nullable=False),
        "talk_time": Column(pa.Int, Check.in_range(1, 30), nullable=False),
        "three_g": Column(pa.Int, Check.isin([0, 1]), nullable=False),
        "touch_screen": Column(pa.Int, Check.isin([0, 1]), nullable=False),
        "wifi": Column(pa.Int, Check.isin([0, 1]), nullable=False),
    }

    if include_target:
        schema_dict["price_range"] = Column(
            pa.Int, Check.isin([0, 1, 2, 3]), nullable=False
        )

    return DataFrameSchema(schema_dict, coerce=True, strict=False)


def validate_schema(df: pd.DataFrame, output_report_name: str = "baseline_validation.csv") -> bool:
    """Validate DataFrame against mobile price schema and write validation report."""
    os.makedirs("artifacts", exist_ok=True)
    report_path = os.path.join("artifacts", output_report_name)

    has_target = "price_range" in df.columns
    schema = get_mobile_schema(include_target=has_target)

    validation_records = []
    all_passed = True

    # 1. Check required feature columns
    expected_cols = [
        "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
        "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
        "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi"
    ]
    if has_target:
        expected_cols.append("price_range")

    for col in expected_cols:
        col_present = col in df.columns
        null_count = int(df[col].isnull().sum()) if col_present else -1
        col_type = str(df[col].dtype) if col_present else "missing"

        status = "PASSED" if col_present and null_count == 0 else "FAILED"
        if status == "FAILED":
            all_passed = False

        validation_records.append({
            "column": col,
            "present": col_present,
            "dtype": col_type,
            "null_count": null_count,
            "status": status
        })

    # 2. Run Pandera schema validation
    try:
        schema.validate(df, lazy=True)
        schema_check = "PASSED"
    except Exception as exc:
        schema_check = f"FAILED: {str(exc)[:120]}"
        all_passed = False

    validation_records.append({
        "column": "__ALL_COLUMNS_SCHEMA_CHECK__",
        "present": True,
        "dtype": "schema_validation",
        "null_count": 0,
        "status": schema_check
    })

    report_df = pd.DataFrame(validation_records)
    report_df.to_csv(report_path, index=False)

    print(f"[INFO] Schema validation complete. Passed: {all_passed}. Report saved to {report_path}")
    return all_passed


if __name__ == "__main__":
    train_path = "data/raw/mobile_price_train.csv"
    if os.path.exists(train_path):
        sample_df = pd.read_csv(train_path)
        validate_schema(sample_df)
    else:
        print(f"Data not found at {train_path}")

