# Data Quality Audit Report (raw data)

- Shape: 3276 rows, 10 columns
- Columns: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability

## Missing values

- ph: 491 (15.0%)
- Hardness: 0 (0.0%)
- Solids: 0 (0.0%)
- Chloramines: 0 (0.0%)
- Sulfate: 781 (23.8%)
- Conductivity: 0 (0.0%)
- Organic_carbon: 0 (0.0%)
- Trihalomethanes: 162 (4.9%)
- Turbidity: 0 (0.0%)
- Potability: 0 (0.0%)

- Rows with at least one missing value: 1265

## Duplicates

- Exact duplicate rows: 0
- Duplicate feature rows: 0

## Invalid ranges

- ph: 0
- Hardness: 0
- Solids: 1
- Chloramines: 0
- Sulfate: 0
- Conductivity: 0
- Organic_carbon: 0
- Trihalomethanes: 0
- Turbidity: 0

## Outliers (IQR rule)

- ph: 46
- Hardness: 83
- Solids: 47
- Chloramines: 61
- Sulfate: 41
- Conductivity: 11
- Organic_carbon: 25
- Trihalomethanes: 33
- Turbidity: 19

## Target balance

- Potability=0: 1998 (61.0%)
- Potability=1: 1278 (39.0%)