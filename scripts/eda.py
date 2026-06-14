"""Exploratory data analysis for manufacturing quality classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "manufacturing_quality.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "eda_summary.md"

TARGET_CANDIDATES = [
    "target",
    "label",
    "quality",
    "defect",
    "failure",
    "pass_fail",
]


def detect_target_column(df: pd.DataFrame) -> str | None:
    return next((col for col in TARGET_CANDIDATES if col in df.columns), None)


def build_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    table = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": (missing / len(df) * 100).round(2),
        }
    )
    return table[table["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def build_target_table(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    counts = df[target_col].value_counts(dropna=False).sort_index()
    return pd.DataFrame(
        {
            "count": counts,
            "percentage": (counts / len(df) * 100).round(2),
        }
    )


def build_report(df: pd.DataFrame, target_col: str | None) -> str:
    lines: list[str] = [
        "# Manufacturing Quality EDA Summary",
        "",
        f"- Source file: `{RAW_PATH.relative_to(PROJECT_ROOT)}`",
        "- Task: manufacturing quality classification",
        "",
        "## Dataset Shape",
        "",
        f"- Rows: **{len(df):,}**",
        f"- Columns: **{len(df.columns):,}**",
        "",
        "## Column Types",
        "",
        df.dtypes.astype(str).to_frame("dtype").to_markdown(),
        "",
        "## Missing Values",
        "",
    ]

    missing_table = build_missing_table(df)
    if missing_table.empty:
        lines.append("无缺失值。")
    else:
        lines.append(missing_table.to_markdown())

    lines.extend(["", "## Target Distribution", ""])
    if target_col:
        lines.append(f"目标列：`{target_col}`\n")
        lines.append(build_target_table(df, target_col).to_markdown())
    else:
        lines.append("未自动识别 target 列，请手动确认目标列。")

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        lines.extend(["", "## Numeric Describe", ""])
        lines.append(numeric_df.describe().T.to_markdown())

    lines.append("")
    return "\n".join(lines)


def print_summary(df: pd.DataFrame, target_col: str | None) -> None:
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\nColumn types:")
    print(df.dtypes.to_string())

    missing_table = build_missing_table(df)
    print("\nMissing values:")
    if missing_table.empty:
        print("无缺失值。")
    else:
        print(missing_table.to_string())

    if target_col:
        print(f"\nTarget distribution ({target_col}):")
        print(build_target_table(df, target_col).to_string())
    else:
        print("\nTarget column: 未自动识别")

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        print("\nNumeric describe:")
        print(numeric_df.describe().to_string())


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{RAW_PATH}。请先将 CSV 放到 data/raw/ 下。"
        )

    df = pd.read_csv(RAW_PATH)
    target_col = detect_target_column(df)

    print_summary(df, target_col)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(df, target_col), encoding="utf-8")
    print(f"\n✅ EDA 完成：{REPORT_PATH}")


if __name__ == "__main__":
    main()
