# predictive-maintenance-mini

> Repository: https://github.com/ShihangPENg-afk/predictive-maintenance-mini  
> 中文文档: [README.md](README.md)

End-to-end industrial prediction mini demo for **predictive maintenance** and **manufacturing quality prediction**: EDA, model training, MLflow tracking, FastAPI inference, and Docker deployment.

## Related Repositories

| Repository | GitHub | Description |
|------------|--------|-------------|
| **predictive-maintenance-mini** | https://github.com/ShihangPENg-afk/predictive-maintenance-mini | This repo: industrial ML training & inference API |
| **rag-agentic-system** | https://github.com/ShihangPENg-afk/rag-agentic-system | Agentic RAG app; calls this service over HTTP |
| **llm-finetune-for-manufacturing** | https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing | LoRA fine-tuning experiments (unrelated to this pipeline) |

---

## 1. Project Overview

**predictive-maintenance-mini** is a compact, end-to-end demo of an industrial ML workflow. It targets two closely related use cases:

- **Predictive maintenance** — infer equipment / batch health from sensor readings
- **Manufacturing quality prediction** — classify production batches as `normal` or `defect`

The task is binary classification with scikit-learn: given line sensor readings (temperature, pressure, vibration, speed, humidity), predict batch quality.

| Stage | Script / Component | Output |
|-------|-------------------|--------|
| EDA | `scripts/eda.py` | `docs/eda_summary.md` |
| Training | `scripts/train_model.py` | `artifacts/model.pkl`, `metrics.json`, `schema.json` |
| Experiment tracking | MLflow (SQLite) | `mlflow.db` (gitignored) |
| Inference API | `app/main.py` | `/health`, `/model-info`, `/predict` |
| Container | `Dockerfile` + `docker-compose.yml` | Port `8010` |

**Dataset**: `data/raw/manufacturing_quality.csv` — reproducible simulated data (500 rows, no missing values). Regenerate with `python scripts/generate_sample_data.py`.

---

## 2. ML Pipeline

### EDA

```bash
git clone https://github.com/ShihangPENg-afk/predictive-maintenance-mini.git
cd predictive-maintenance-mini
pip install -r requirements.txt
python scripts/eda.py
```

`scripts/eda.py` reads the CSV, detects the target column, prints shape / types / missing values, summarizes the target distribution and numeric features, and writes `docs/eda_summary.md`.

### Feature preprocessing

Training pipeline (`scripts/train_model.py`):

1. Load CSV; drop rows with empty target
2. Split numeric vs categorical features (all numeric in the current dataset)
3. Preprocessing:
   - Numeric: `SimpleImputer(median)` → `StandardScaler`
   - Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder` (reserved for future datasets)
4. 80/20 stratified train/test split

### RandomForest baseline

Current model: `RandomForestClassifier` (`n_estimators=200`, `class_weight="balanced"`, `random_state=42`).

```bash
python scripts/train_model.py
# or
make train
```

Metrics: `accuracy`, `f1_macro`, full classification report.

### MLflow tracking

Runs are stored in local SQLite `mlflow.db` (gitignored):

- Experiment: `predictive-maintenance-mini`
- Params: `model_type`, `n_estimators`, `test_size`, `random_state`, `target_column`
- Metrics: `accuracy`, `f1_macro`

```bash
bash scripts/start_mlflow_ui.sh   # http://127.0.0.1:5001
```

| Run overview (RandomForest baseline) | Artifacts |
|--------------------------------------|-----------|
| ![MLflow run overview](docs/images/mlflow_ui.png) | ![MLflow artifacts](docs/images/mlflow_artifacts.png) |

### Artifacts

Each training run produces three files under `artifacts/`:

| File | Description |
|------|-------------|
| `model.pkl` | Serialized sklearn pipeline (preprocessor + classifier) |
| `metrics.json` | Test-set accuracy, F1, classification report |
| `schema.json` | Feature names, class labels, model metadata |

**Baseline results** (500 rows, 80/20 split):

| Metric | Value |
|--------|-------|
| accuracy | 0.71 |
| f1_macro | 0.67 |
| defect recall | 0.51 |

Full report: [`docs/experiment_report.md`](docs/experiment_report.md).

---

## 3. API Serving

FastAPI inference service in `app/main.py`. Requires trained `artifacts/model.pkl`.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model-info` | GET | Features, classes, training metrics |
| `/predict` | POST | Single-sample inference |

**Request body** (`POST /predict`):

```json
{
  "features": {
    "temperature": 73.5,
    "pressure": 5.2,
    "vibration": 2.1,
    "speed": 118.0,
    "humidity": 48.0
  }
}
```

**Response** includes `prediction`, `prediction_label`, `probabilities`, `risk_level` (high / medium / low), and a text `recommendation`.

![FastAPI Swagger docs](docs/images/swagger_docs.png)

**Local API examples:**

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/model-info
curl -X POST http://127.0.0.1:8010/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"temperature":73.5,"pressure":5.2,"vibration":2.1,"speed":118.0,"humidity":48.0}}'
```

**Tests:**

```bash
pytest tests/
```

---

## 4. Docker

Requires `artifacts/` (run `make train` or `python scripts/train_model.py` first):

```bash
docker compose up --build -d
```

Host port: **8010**.

```bash
curl http://127.0.0.1:8010/health

curl -X POST http://127.0.0.1:8010/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"temperature":73.5,"pressure":5.2,"vibration":2.1,"speed":118.0,"humidity":48.0}}'
```

Makefile shortcuts: `make docker-up`, `make docker-verify`, `make docker-logs`, `make docker-down`.

---

## 5. Agent Integration

This service integrates with [rag-agentic-system](https://github.com/ShihangPENg-afk/rag-agentic-system) via **decoupled HTTP deployment** (separate repos, no shared code or database):

| Service | Port | Role |
|---------|------|------|
| predictive-maintenance-mini | 8010 | `/health`, `/model-info`, `POST /predict` |
| rag-agentic-system | 8000 | Agent tool `check_machine_health` calls this API |
| Streamlit UI | 8501 | "Device health" tab via `HEALTH_API_URL` |

The `check_machine_health` tool in rag-agentic-system sends sensor readings to `POST /predict` and surfaces `prediction`, `risk_level`, and `recommendation` to the agent.

See [industrial_demo_guide.md](https://github.com/ShihangPENg-afk/rag-agentic-system/blob/main/docs/industrial_demo_guide.md) in rag-agentic-system.

![rag-agentic-system device health tab](docs/images/rag_agent_tab.png)

---

## 6. Known Limitations

- **Baseline model only** — RandomForest with fixed hyperparameters; no hyperparameter search or ensemble tuning
- **Not SOTA** — not designed to chase state-of-the-art accuracy
- **Not production validated** — demo / POC only; do not use for real production decisions
- **Simulated / sample data** — the dataset is reproducible simulated data (500 rows), not real factory telemetry; generalization is limited
- **Class imbalance** — defect recall is low (~0.51) under the current split
- **Static artifacts** — no online retraining, A/B testing, or model hot-swap

---

## Project Layout

```
predictive-maintenance-mini/
├── app/              # FastAPI inference service
├── artifacts/        # model.pkl, metrics.json, schema.json
├── data/raw/         # simulated CSV
├── docs/             # EDA summary, experiment report, screenshots
├── scripts/          # EDA, training, data generation, utilities
├── tests/            # API smoke tests
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## License

MIT — see [LICENSE](LICENSE).
