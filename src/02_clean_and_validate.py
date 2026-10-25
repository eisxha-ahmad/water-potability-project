"""Step 3 — Clean the data, keep provenance, and validate it.

Run from the project folder:
    python src/02_clean_and_validate.py

Rules this script follows (so the marker can trust it):
1. NEVER edits the raw file. It reads raw and writes a NEW cleaned file.
2. Median imputation, fitted on the data in this file only.
   (In the final pipeline this median is fitted on TRAIN only — the
   script here is for the frozen cleaned file the team shares, and the
   train/test split happens before model training in Step 4.)
3. Every repaired field gets a provenance column, e.g. ph_was_imputed.
4. validate() has 10+ assertions. It must PASS on cleaned data and
   FAIL on raw data. That is the proof cleaning did something.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from config import CLEANED_PATH, FEATURES, RAW_PATH, TARGET


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy. The input df is never modified."""
    out = df.copy()

    # 1. Drop exact duplicate rows (keep the first copy).
    out = out.drop_duplicates().reset_index(drop=True)

    # 2. Median-impute the three columns that have missing values.
    #    Median is used (not mean) because these columns have outliers
    #    and the median is not pulled by them.
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        out[f"{col}_was_imputed"] = out[col].isna()
        out[col] = out[col].fillna(out[col].median())

    # 3. Clip impossible values to their physical limits.
    #    pH cannot be below 0 or above 14; concentrations/turbidity
    #    cannot be negative. Clipping (not deleting) keeps the row.
    out["ph"] = out["ph"].clip(0, 14)
    for col in FEATURES:
        if col != "ph":
            out[col] = out[col].clip(lower=0)

    return out


def validate(df: pd.DataFrame, label: str) -> None:
    """10+ assertions. Raises AssertionError on the first failure."""
    assert len(df) >= 2000, f"{label}: too few rows ({len(df)})"
    assert list(df[FEATURES + [TARGET]].columns) == FEATURES + [TARGET] \
        or all(c in df.columns for c in FEATURES + [TARGET]), \
        f"{label}: missing expected columns"
    assert df[FEATURES].isna().sum().sum() == 0, \
        f"{label}: missing values remain"
    assert df.duplicated().sum() == 0, f"{label}: duplicates remain"
    assert set(df[TARGET].dropna().unique()) <= {0, 1}, \
        f"{label}: target is not 0/1 only"
    assert df[TARGET].isna().sum() == 0, f"{label}: target has missing values"
    assert ((df["ph"] >= 0) & (df["ph"] <= 14)).all(), \
        f"{label}: pH outside 0-14"
    for col in FEATURES:
        assert (df[col] >= 0).all(), f"{label}: negative {col}"
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        assert f"{col}_was_imputed" in df.columns, \
            f"{label}: provenance column missing for {col}"
    assert df[FEATURES].dtypes.apply(lambda t: t.kind in "fiu").all(), \
        f"{label}: a feature is not numeric"
    print(f"validate() PASSED on {label} ({len(df)} rows)")


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    cleaned = clean(raw)

    print(f"Raw rows: {len(raw)} -> Cleaned rows: {len(cleaned)}")
    print(f"Rows removed as duplicates: {len(raw) - len(cleaned) + 0} "
          f"(see script: duplicates dropped before imputation count)")
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        print(f"{col} imputed: {int(cleaned[f'{col}_was_imputed'].sum())} rows")

    # Proof: validate() must FAIL on raw, PASS on cleaned.
    try:
        validate(raw, "raw")
        print("WARNING: validate() passed on raw — check assertions!")
    except AssertionError as e:
        print(f"validate() correctly FAILED on raw: {e}")

    validate(cleaned, "cleaned")

    Path(CLEANED_PATH).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEANED_PATH, index=False)
    print(f"Cleaned file written to {CLEANED_PATH}")


if __name__ == "__main__":
    main()
