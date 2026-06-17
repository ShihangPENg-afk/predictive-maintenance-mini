#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAG_DIR="${RAG_AGENTIC_SYSTEM_DIR:-${PROJECT_ROOT}/../rag-agentic-system}"

if ! curl -fsS http://127.0.0.1:8010/health >/dev/null; then
  echo "❌ predictive-maintenance-mini 未在 :8010 运行"
  echo "   请先执行: uvicorn app.main:app --host 127.0.0.1 --port 8010"
  exit 1
fi

if ! curl -fsS http://127.0.0.1:8501 >/dev/null; then
  echo "❌ Streamlit 未在 :8501 运行"
  echo "   请在 rag-agentic-system 目录执行:"
  echo "   HEALTH_API_URL=http://127.0.0.1:8010 python -m streamlit run ui/streamlit_app.py --server.port 8501"
  exit 1
fi

PYTHON="${RAG_PYTHON:-${RAG_DIR}/.venv/bin/python3.12}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3.12 || command -v python3)"
fi

exec "${PYTHON}" "${SCRIPT_DIR}/capture_showcase_screenshots.py"
