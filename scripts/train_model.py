"""Train a manufacturing quality classification model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "manufacturing_quality.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SCHEMA_PATH = ARTIFACTS_DIR / "schema.json"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"
MLFLOW_EXPERIMENT = "industrial-health-demo"

# 与 scripts/eda.py 保持一致；若 CSV 目标列不在列表中，请在此追加
TARGET_CANDIDATES = [
    "target",
    "label",
    "quality",
    "defect",
    "failure",
    "pass_fail",
]

MODEL_TYPE = "RandomForestClassifier"
N_ESTIMATORS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42


def find_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"未找到 target 列。请将目标列重命名为 `{TARGET_CANDIDATES[0]}`，"
        f"或在脚本中修改 TARGET_CANDIDATES。当前列：{list(df.columns)}"
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{RAW_PATH}。请先将 CSV 放到 data/raw/ 下。"
        )

    df = pd.read_csv(RAW_PATH)
    target_col = find_target_column(df)

    before_rows = len(df)
    df = df.dropna(subset=[target_col]).copy()
    dropped_rows = before_rows - len(df)
    if dropped_rows:
        print(f"已删除 target 为空的行：{dropped_rows} 行")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    features = list(X.columns)
    classes = [str(value) for value in sorted(y.unique())]

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=None,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "target_column": target_col,
        "rows": int(df.shape[0]),
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classes": classes,
        "classification_report": report,
    }

    schema = {
        "target_column": target_col,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classes": classes,
        "model_type": MODEL_TYPE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    SCHEMA_PATH.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_type": MODEL_TYPE,
                "n_estimators": N_ESTIMATORS,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "target_column": target_col,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            }
        )
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(SCHEMA_PATH))
        run_id = run.info.run_id

    print("✅ 模型训练完成")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print(f"\n✅ 模型已保存：{MODEL_PATH}")
    print(f"✅ 指标已保存：{METRICS_PATH}")
    print(f"✅ Schema 已保存：{SCHEMA_PATH}")
    print(f"✅ MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
