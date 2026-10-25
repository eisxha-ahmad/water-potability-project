"""Step 2 — Profile the raw data and write the audit report.

Run from the project folder:
    python src/01_profile_and_audit.py

What it does, in plain words:
1. Loads ONLY the raw file (never edits it).
2. Prints shape, column types, missing counts, duplicates, target balance.
3. Checks 6+ fault categories and counts affected records for each.
4. Writes reports/audit_report.md so the report writer can copy numbers
   straight from it — no hand-typed numbers, no mistakes.

Fault vocabulary (Lecture 2):
- missing values, duplicates, outliers (IQR rule), invalid range,
  inconsistent values, implausible values.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from config import FEATURES, RAW_PATH, AUDIT_PATH

# Plausible ranges, agreed before looking at model results.
# pH is 0-14 by definition. Others must be >= 0 (a negative
# concentration / turbidity cannot exist in the real world).
RANGES = {
    "ph": (0, 14),
    "Hardness": (0, 500),
    "Solids": (0, 60000),
    "Chloramines": (0, 20),
    "Sulfate": (0, 500),
    "Conductivity": (0, 1500),
    "Organic_carbon": (0, 30),
    "Trihalomethanes": (0, 200),
    "Turbidity": (0, 20),
}


def iqr_outlier_count(s: pd.Series) -> int:
    """Count outliers by the IQR rule: outside Q1 - 1.5*IQR .. Q3 + 1.5*IQR."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    print(f"Shape: {df.shape}  (rows, columns)")
    print(f"Columns: {list(df.columns)}")
    print("\n--- Types ---")
    print(df.dtypes.to_string())
    print("\n--- Missing values per column ---")
    missing = df.isna().sum()
    print(missing.to_string())
    print(f"\nTotal cells missing: {int(missing.sum())}")
    print(f"Rows with at least one missing value: {int(df.isna().any(axis=1).sum())}")

    print("\n--- Duplicates ---")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")
    print(f"Duplicate rows ignoring target: "
          f"{int(df.duplicated(subset=FEATURES).sum())}")

    print("\n--- Target balance (Potability) ---")
    print(df["Potability"].value_counts(dropna=False).to_string())
    print(df["Potability"].value_counts(normalize=True).round(3).to_string())

    print("\n--- Invalid ranges ---")
    invalid = {}
    for col, (lo, hi) in RANGES.items():
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        invalid[col] = n
        print(f"{col}: {n} outside [{lo}, {hi}]")

    print("\n--- Outliers (IQR rule) ---")
    outliers = {c: iqr_outlier_count(df[c].dropna()) for c in FEATURES}
    for col, n in outliers.items():
        print(f"{col}: {n}")

    print("\n--- Missingness pattern (is missingness random?) ---")
    # If missingness were random, the missing-rate would be similar for
    # safe and unsafe samples. Compare the two groups.
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        rate = df.groupby("Potability")[col].apply(lambda s: s.isna().mean())
        print(f"{col}: missing rate by Potability =\n{rate.round(3).to_string()}")

    print("\n--- Descriptive statistics ---")
    print(df[FEATURES].describe().round(2).to_string())

    # Write the report file the team will copy numbers from.
    lines = ["# Data Quality Audit Report (raw data)\n"]
    lines.append(f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    lines.append(f"- Columns: {', '.join(df.columns)}")
    lines.append("\n## Missing values\n")
    for col, n in missing.items():
        lines.append(f"- {col}: {n} ({n / len(df):.1%})")
    lines.append(f"\n- Rows with at least one missing value: "
                 f"{int(df.isna().any(axis=1).sum())}")
    lines.append("\n## Duplicates\n")
    lines.append(f"- Exact duplicate rows: {int(df.duplicated().sum())}")
    lines.append(f"- Duplicate feature rows: "
                 f"{int(df.duplicated(subset=FEATURES).sum())}")
    lines.append("\n## Invalid ranges\n")
    for col, n in invalid.items():
        lines.append(f"- {col}: {n}")
    lines.append("\n## Outliers (IQR rule)\n")
    for col, n in outliers.items():
        lines.append(f"- {col}: {n}")
    lines.append("\n## Target balance\n")
    for k, v in df["Potability"].value_counts().items():
        lines.append(f"- Potability={k}: {v} ({v / len(df):.1%})")

    Path(AUDIT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(AUDIT_PATH).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nAudit report written to {AUDIT_PATH}")


if __name__ == "__main__":
    main()
