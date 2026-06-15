# 实验报告：Manufacturing Quality Classification

本文档记录 `industrial-health-demo` 在当前数据集与默认训练配置下的实验结果。数据与指标来自 `data/raw/manufacturing_quality.csv` 与 `artifacts/metrics.json`（由 `scripts/train_model.py` 生成）。

## 1. 数据集规模

| 项目 | 值 |
|------|-----|
| 行数 | 500 |
| 列数 | 6（5 个特征 + 1 个标签） |
| 缺失值 | 无 |
| 数据源 | `data/raw/manufacturing_quality.csv` |

特征列均为 `float64`，标签列 `target` 为字符串类型。

## 2. Target 分布

目标列：`target`（二分类）

| 类别 | 样本数 | 占比 |
|------|--------|------|
| `normal` | 326 | 65.2% |
| `defect` | 174 | 34.8% |

类别略不均衡，训练时使用 `class_weight="balanced"` 缓解。测试集采用分层抽样（`stratify=y`），测试集大小 100 条（20%）。

## 3. 特征处理

### 3.1 特征列表

| 特征 | 类型 | 简要含义 |
|------|------|----------|
| `temperature` | 数值 | 温度 |
| `pressure` | 数值 | 压力 |
| `vibration` | 数值 | 振动 |
| `speed` | 数值 | 线速 |
| `humidity` | 数值 | 湿度 |

当前数据集无类别型特征（`categorical_features` 为空）。

### 3.2 预处理管道

通过 `sklearn.compose.ColumnTransformer` 构建：

**数值特征：**

```
SimpleImputer(strategy="median") → StandardScaler()
```

**类别特征（预留，当前未使用）：**

```
SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")
```

预处理与模型封装在同一 `Pipeline` 中，推理时由 `joblib` 加载的 `model.pkl` 一并完成变换与预测。

### 3.3 数据划分

| 参数 | 值 |
|------|-----|
| `test_size` | 0.2 |
| `random_state` | 42 |
| 训练集 | 400 条 |
| 测试集 | 100 条 |

## 4. 模型

| 项目 | 配置 |
|------|------|
| 算法 | `RandomForestClassifier` |
| `n_estimators` | 200 |
| `max_depth` | None（不限制） |
| `class_weight` | `"balanced"` |
| `random_state` | 42 |

选择 Random Forest 的原因：实现简单、对表格数据鲁棒、无需额外调参即可得到可解释的 baseline，适合本 Demo 的工程演示目标。

## 5. 评估指标

在 held-out 测试集（100 条）上的结果：

| 指标 | 值 |
|------|-----|
| **accuracy** | **0.7100** |
| **f1_macro** | **0.6695** |

### 各类别表现

| 类别 | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| defect | 0.6000 | 0.5143 | 0.5538 | 35 |
| normal | 0.7571 | 0.8154 | 0.7852 | 65 |

**观察：**

- 整体 accuracy 约 71%，作为 baseline 可接受，但远非最优。
- `defect` 类 recall 偏低（约 51%），漏检风险较高，与类别不均衡及模型容量有限有关。
- `normal` 类表现更好，precision / recall 均在 75% 以上。

完整 classification report 保存在 `artifacts/metrics.json`。

## 6. MLflow Run 说明

每次执行 `python scripts/train_model.py` 会创建一条 MLflow run：

| 项目 | 值 |
|------|-----|
| Tracking URI | `sqlite:///<项目根>/mlflow.db` |
| Experiment | `industrial-health-demo` |

**记录的 Params：**

- `model_type`：`RandomForestClassifier`
- `n_estimators`：`200`
- `test_size`：`0.2`
- `random_state`：`42`
- `target_column`：`target`

**记录的 Metrics：**

- `accuracy`
- `f1_macro`

**记录的 Artifacts：**

- `artifacts/model.pkl`
- `artifacts/metrics.json`
- `artifacts/schema.json`

训练结束后终端会打印 `MLflow run_id`，也可通过以下命令在 UI 中查看：

```bash
bash scripts/start_mlflow_ui.sh
# 浏览器打开 http://127.0.0.1:5001
```

## 7. 当前限制与后续计划

### 当前限制

1. **非生产级**：模型未经业务验证，API 中的 `risk_level` 与 `recommendation` 为规则化演示逻辑，不能替代人工质检或专业 PHM 系统。
2. **不追求 SOTA**：未尝试 XGBoost、LightGBM、神经网络或 AutoML；无超参搜索（GridSearch / Optuna）。
3. **数据量有限**：500 条样本不足以评估真实产线漂移与季节性变化。
4. **特征工程不足**：未做交互项、时序窗口、异常检测预处理等工业常见步骤。
5. **评估维度单一**：仅报告 accuracy / f1_macro，缺少 PR 曲线、校准度、成本敏感指标（如漏检代价）。
6. **实验不可复现于 Git**：`mlflow.db` 在 `.gitignore` 中，协作者需本地重新训练以生成 run 记录。

### 后续计划（建议）

| 方向 | 具体动作 |
|------|----------|
| 数据 | 扩充样本、引入真实产线数据、处理类别不均衡（SMOTE / 加权采样） |
| 特征 | 增加时序聚合、设备 ID、班次等上下文；探索特征重要性 |
| 模型 | 对比 Gradient Boosting、逻辑回归 baseline；Optuna 调参 |
| 评估 | 交叉验证、混淆矩阵可视化、defect recall 优先的阈值调优 |
| 工程 | CI 中跑训练 smoke test；MLflow Model Registry；Prometheus 监控推理延迟 |
| 部署 | 多副本 K8s 部署、模型版本切换、批量预测接口 |

---

*报告生成依据：`docs/eda_summary.md`、`artifacts/metrics.json`、`artifacts/schema.json`。重新训练后指标可能变化，请以最新 `artifacts/metrics.json` 为准。*
