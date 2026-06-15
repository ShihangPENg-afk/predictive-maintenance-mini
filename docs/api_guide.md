# Industrial Health API 指南

> 关联文档：[README.md](../README.md) · [experiment_report.md](experiment_report.md)

## 服务地址

默认：`http://127.0.0.1:8010`（Docker 或 `uvicorn app.main:app --host 0.0.0.0 --port 8010`）

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查；返回 `status`、`model_loaded` |
| GET | `/model-info` | 特征列表、类别、训练指标 |
| POST | `/predict` | 单条样本推理 |

交互式文档：`http://127.0.0.1:8010/docs`

## `/predict` 请求与响应

**请求体：**

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

**响应字段：** `prediction`、`prediction_label`、`probabilities`、`risk_level`（high / medium / low）、`recommendation`

**缺失特征：** 返回 HTTP 422，body 含 `missing_fields` 与 `required_fields`。

## 快速验证

```bash
curl -fsS http://127.0.0.1:8010/health
curl -fsS http://127.0.0.1:8010/model-info
python scripts/sample_predict.py
make docker-verify   # 含 health + model-info + predict
```

## 与 rag-agent 联动

rag-agent 通过 `HEALTH_API_URL`（默认 `http://127.0.0.1:8010`）调用本服务；Agent 工具名为 `check_machine_health`。两仓库无共享代码或数据库。

## 限制

- RandomForest **baseline**，非生产级模型
- 推理服务加载静态 `artifacts/`，不支持在线再训练
- 未云部署；无鉴权限流
