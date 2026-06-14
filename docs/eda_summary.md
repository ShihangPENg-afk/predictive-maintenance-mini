# Manufacturing Quality EDA Summary

- Source file: `data/raw/manufacturing_quality.csv`
- Task: manufacturing quality classification

## Dataset Shape

- Rows: **500**
- Columns: **6**

## Column Types

|             | dtype   |
|:------------|:--------|
| temperature | float64 |
| pressure    | float64 |
| vibration   | float64 |
| speed       | float64 |
| humidity    | float64 |
| target      | str     |

## Missing Values

无缺失值。

## Target Distribution

目标列：`target`

| target   |   count |   percentage |
|:---------|--------:|-------------:|
| defect   |     174 |         34.8 |
| normal   |     326 |         65.2 |

## Numeric Describe

|             |   count |      mean |       std |   min |      25% |     50% |      75% |    max |
|:------------|--------:|----------:|----------:|------:|---------:|--------:|---------:|-------:|
| temperature |     500 |  71.9343  |  4.79983  | 59.17 |  68.6625 |  72.015 |  74.935  |  86.57 |
| pressure    |     500 |   5.17312 |  0.611152 |  3.01 |   4.765  |   5.205 |   5.555  |   7.11 |
| vibration   |     500 |   2.00122 |  0.806192 |  0.2  |   1.47   |   2     |   2.5225 |   4.25 |
| speed       |     500 | 117.627   | 14.9502   | 80    | 108.153  | 117.965 | 127.627  | 160    |
| humidity    |     500 |  47.9912  |  8.14346  | 25    |  42.4425 |  48.29  |  53.2725 |  70    |
