#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLFLOW_DB="${PROJECT_ROOT}/mlflow.db"
HOST="${MLFLOW_UI_HOST:-127.0.0.1}"
PORT="${MLFLOW_UI_PORT:-5001}"

if [[ -x "${PROJECT_ROOT}/.venv/bin/mlflow" ]]; then
  MLFLOW_BIN="${PROJECT_ROOT}/.venv/bin/mlflow"
else
  MLFLOW_BIN="mlflow"
fi

echo "MLflow UI: http://${HOST}:${PORT}"
echo "Backend store: sqlite:///${MLFLOW_DB}"

exec "${MLFLOW_BIN}" ui \
  --backend-store-uri "sqlite:///${MLFLOW_DB}" \
  --host "${HOST}" \
  --port "${PORT}"
