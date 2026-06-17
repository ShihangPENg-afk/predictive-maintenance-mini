# Predictive Maintenance Mini

> 项目仓库：https://github.com/ShihangPENg-afk/predictive-maintenance-mini  
> English README: [README.en.md](README.en.md)

工业制造质量分类 Mini Demo：从 EDA、模型训练、MLflow 实验追踪到 FastAPI 推理服务与 Docker 部署的完整链路示例。

## 关联 GitHub 仓库

| 仓库 | GitHub | 说明 |
|------|--------|------|
| **predictive-maintenance-mini** | https://github.com/ShihangPENg-afk/predictive-maintenance-mini | 本仓库：工业 ML 训练与推理 API |
| **rag-agentic-system** | https://github.com/ShihangPENg-afk/rag-agentic-system | Agentic RAG 主应用；通过 HTTP 调用本服务 |
| **llm-finetune-for-manufacturing** | https://github.com/ShihangPENg-afk/llm-finetune-for-manufacturing | LoRA 微调实验（与工业预测链路无关） |

> **定位说明**：本项目用于演示工业预测项目的工程化流程，**不是生产级模型**，**baseline 未针对 SOTA 调优**。目标是展示数据探索、训练、实验记录与 API 部署的标准做法。

## 项目简介

基于 scikit-learn 的二分类任务：根据产线传感器读数（温度、压力、振动、线速、湿度）预测批次质量标签（`normal` / `defect`）。

主要能力：

| 阶段 | 脚本 / 组件 | 产出 |
|------|-------------|------|
| EDA | `scripts/eda.py` | `docs/eda_summary.md` |
| 训练 | `scripts/train_model.py` | `artifacts/model.pkl`、`metrics.json`、`schema.json` |
| 实验追踪 | MLflow（SQLite） | `mlflow.db` |
| 推理 API | `app/main.py` | `/health`、`/model-info`、`/predict` |
| 容器化 | `Dockerfile` + `docker-compose.yml` | 端口 `8010` |

## 数据集说明

**数据来源**：仓库内 CSV 为**可复现的模拟数据**（非真实工厂数据，无第三方数据版权风险）。由 `scripts/generate_sample_data.py` 按传感器分布与缺陷概率规则生成（`random_state=42`，500 行）。

重新生成数据：

```bash
python scripts/generate_sample_data.py
python scripts/eda.py
python scripts/train_model.py
```

当前数据文件：`data/raw/manufacturing_quality.csv`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `temperature` | float | 温度 |
| `pressure` | float | 压力 |
| `vibration` | float | 振动 |
| `speed` | float | 线速 |
| `humidity` | float | 湿度 |
| `target` | str | 标签：`normal`（合格）/ `defect`（不合格） |

当前数据集规模：**500 行 × 6 列**，无缺失值。类别分布略不均衡（normal 约 65%，defect 约 35%）。详见 `docs/eda_summary.md` 与 `docs/experiment_report.md`。

## EDA 方法

克隆本仓库：

```bash
git clone https://github.com/ShihangPENg-afk/predictive-maintenance-mini.git
cd predictive-maintenance-mini
```

运行探索性数据分析：

```bash
pip install -r requirements.txt
python scripts/eda.py
```

`scripts/eda.py` 会：

1. 读取 `data/raw/manufacturing_quality.csv`
2. 自动识别目标列（候选名：`target`、`label`、`quality` 等）
3. 输出数据集行列数、列类型、缺失值统计
4. 统计 `target` 分布（计数与占比）
5. 对数值特征生成 `describe` 摘要
6. 将结果写入 `docs/eda_summary.md`，并在终端打印摘要

## 模型训练方法

```bash
python scripts/train_model.py
```

训练流程（`scripts/train_model.py`）：

1. **数据加载**：读取 CSV，删除 `target` 为空的行
2. **特征划分**：数值列走数值管道，类别列走 One-Hot 管道（当前数据集均为数值特征）
3. **预处理**：
   - 数值：`SimpleImputer(median)` → `StandardScaler`
   - 类别：`SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`
4. **模型（当前 baseline）**：`RandomForestClassifier`（`n_estimators=200`，`class_weight="balanced"`，`random_state=42`）。**尚未实现 XGBoost / LightGBM 对比**，见「后续计划」。
5. **划分**：80/20 分层抽样（`test_size=0.2`，`stratify=y`）
6. **评估**：计算 `accuracy`、`f1_macro` 及完整 classification report
7. **保存**：写入 `artifacts/` 并在 MLflow 中记录一次 run

也可使用 Makefile 在 Docker 构建前自动训练（若模型不存在）：

```bash
make train
```

## MLflow 实验记录

训练脚本会将每次 run 写入本地 SQLite 后端 `mlflow.db`（已加入 `.gitignore`）：

- **Experiment 名称**：`predictive-maintenance-mini`
- **记录参数**：`model_type`、`n_estimators`、`test_size`、`random_state`、`target_column`
- **记录指标**：`accuracy`、`f1_macro`
- **记录产物**：`model.pkl`、`metrics.json`、`schema.json`

启动 MLflow UI 查看实验：

```bash
bash scripts/start_mlflow_ui.sh
```

默认地址：<http://127.0.0.1:5001>。可通过环境变量修改：

```bash
MLFLOW_UI_HOST=0.0.0.0 MLFLOW_UI_PORT=5001 bash scripts/start_mlflow_ui.sh
```

**MLflow UI 示例**（参数、指标与 `artifacts/` 一致）：

| Run 概览（RandomForest baseline） | Artifacts（`model.pkl` / `metrics.json` / `schema.json`） |
|-----------------------------------|-----------------------------------------------------------|
| ![MLflow run overview](docs/images/mlflow_ui.png) | ![MLflow artifacts](docs/images/mlflow_artifacts.png) |

> 若本地 UI 仍显示旧 Experiment 名 `industrial-health-demo`，删除 `mlflow.db` 与 `mlruns/` 后重新执行 `python scripts/train_model.py`，即可在 `predictive-maintenance-mini` 下看到新 run。

## FastAPI `/predict` 服务

本地启动（需先完成训练，存在 `artifacts/model.pkl`）：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/model-info` | GET | 特征列表、类别、训练指标 |
| `/predict` | POST | 单条样本推理 |

`/predict` 请求体：

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

响应包含 `prediction`、`prediction_label`、`probabilities`、`risk_level`（high / medium / low）及 `recommendation` 文本建议。

**Swagger 交互文档**（<http://127.0.0.1:8010/docs>）：

![FastAPI Swagger docs](docs/images/swagger_docs.png)

## Docker 启动方法

**前提**：已存在 `artifacts/` 模型产物（`make train` 或 `python scripts/train_model.py`）。

```bash
docker compose up --build -d
```

或使用 Makefile：

```bash
make docker-up      # 若缺模型会自动训练
make docker-verify  # 健康检查 + model-info + 示例预测
make docker-logs    # 查看日志
make docker-down    # 停止容器
```

服务映射到宿主机 **8010** 端口。镜像仅包含 `app/`、`artifacts/`、`scripts/` 与运行依赖，不包含 `.venv`、`.env`、`mlflow.db` 及训练数据。

## API 调用示例

**健康检查：**

```bash
curl http://127.0.0.1:8010/health
```

**查看模型信息：**

```bash
curl http://127.0.0.1:8010/model-info
```

**预测：**

```bash
curl -X POST http://127.0.0.1:8010/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "temperature": 73.5,
      "pressure": 5.2,
      "vibration": 2.1,
      "speed": 118.0,
      "humidity": 48.0
    }
  }'
```

**Python 示例脚本**（从 `/model-info` 读取特征名后调用 `/predict`）：

```bash
python scripts/sample_predict.py
```

交互式 API 文档：<http://127.0.0.1:8010/docs>

## 运行测试

```bash
pip install -r requirements.txt
pytest tests/
```

覆盖 `/health`、`/model-info`、`/predict` 及缺失特征 422 响应。

## 实验结果摘要

当前 baseline（RandomForest，`artifacts/metrics.json`，500 行数据、80/20 分层划分）：

| 指标 | 值 |
|------|-----|
| **accuracy** | 0.71 |
| **f1_macro** | 0.67 |
| defect recall | 0.51（类别不均衡，defect 召回偏低） |

完整报告与限制说明见 [`docs/experiment_report.md`](docs/experiment_report.md)。**不可用于生产决策**。

## 当前限制

- **演示用途**：模型与 API 用于流程演示与实验验证，未经充分验证，**不可直接用于生产决策**。
- **baseline 定位**：未做超参搜索与集成学习；**不以 SOTA 为目标**。
- **数据规模小**：500 条合成/示例数据，泛化能力有限。
- **类别不均衡**：defect 类 recall 偏低，需进一步采样或调参。
- **无在线再训练**：推理服务只加载静态 `artifacts/`，不支持 A/B 或模型热更新。

更详细的实验结论与后续计划见 [`docs/experiment_report.md`](docs/experiment_report.md)。

## 与 rag-agentic-system 的集成

本服务与 [rag-agentic-system](https://github.com/ShihangPENg-afk/rag-agentic-system) **解耦部署**（独立仓库、HTTP 调用，无共享代码或数据库）：

| 服务 | 端口 | 集成方式 |
|------|------|----------|
| predictive-maintenance-mini | 8010 | 提供 `/health`、`/model-info`、`POST /predict` |
| rag-agentic-system | 8000 | Agent 工具 `check_machine_health` HTTP 调用本服务 |
| Streamlit UI | 8501 | 「设备健康预测」Tab 直连 `HEALTH_API_URL` |

详见 rag-agentic-system 文档：[industrial_demo_guide.md](https://github.com/ShihangPENg-afk/rag-agentic-system/blob/main/docs/industrial_demo_guide.md)。

**Streamlit「设备健康预测」Tab 联调示例**（`HEALTH_API_URL` → `:8010`，输入传感器读数后展示 `prediction` / `risk_level` / `recommendation`）：

![rag-agentic-system device health tab](docs/images/rag_agent_tab.png)

本地复现截图（需先启动 `:8010` 推理服务，再在 rag-agentic-system 目录启动 Streamlit `:8501`）：

```bash
# 终端 1：本仓库
uvicorn app.main:app --host 127.0.0.1 --port 8010

# 终端 2：rag-agentic-system（若 .venv 脚本报错，用 python -m streamlit）
cd ../rag-agentic-system
HEALTH_API_URL=http://127.0.0.1:8010 python -m streamlit run ui/streamlit_app.py --server.port 8501
```

或使用一键脚本重新生成 README 展示图（依赖 rag-agentic-system 的 Playwright）：

```bash
bash scripts/capture_showcase_screenshots.sh
```

## 后续计划

- 扩展特征工程与超参搜索；**新增 XGBoost / LightGBM baseline 对比**
- 将 `model.pkl` 迁至 Release 分发（可选），减小仓库体积
- 增加 GitHub Actions CI（`pytest` + Docker 冒烟测试）

## 展示截图

| 素材 | 路径 | 状态 |
|------|------|------|
| MLflow Run 概览 | `docs/images/mlflow_ui.png` | 已内置 |
| MLflow Artifacts | `docs/images/mlflow_artifacts.png` | 已内置 |
| FastAPI Swagger | `docs/images/swagger_docs.png` | 已内置 |
| rag-agentic-system 联调 | `docs/images/rag_agent_tab.png` | 已内置 |

## 项目结构

```
predictive-maintenance-mini/
├── app/                      # FastAPI 推理服务
├── artifacts/                # 模型、指标、schema
├── data/raw/                 # 模拟 CSV
├── docs/                     # EDA、实验报告与展示截图
│   └── images/               # MLflow UI 等截图
├── scripts/
│   ├── generate_sample_data.py
│   ├── eda.py
│   ├── train_model.py
│   ├── start_mlflow_ui.sh
│   └── sample_predict.py
├── tests/                    # API 冒烟测试
├── LICENSE
├── README.en.md              # English README
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## License

MIT — 详见 [LICENSE](LICENSE)。
