# industrial-health-demo 验收清单

> 最后核对：2026-06-15  
> 关联文档：[README.md](../README.md) · [api_guide.md](api_guide.md)

## 验收状态总览

| 类别 | 状态 |
|------|------|
| EDA 与训练 | 已完成 |
| FastAPI 推理 | 已完成 |
| Docker 部署 | 已完成 |
| 与 rag-agent 联动 | 已完成（HTTP 松耦合） |

---

## 1. EDA 与训练产物

- [x] `python scripts/eda.py` → `docs/eda_summary.md`
- [x] `python scripts/train_model.py` → `artifacts/model.pkl`、`metrics.json`、`schema.json`
- [x] MLflow 记录（SQLite `mlflow.db`，本地忽略提交）

## 2. API 端点

- [x] `GET /health` 可访问
- [x] `GET /model-info` 返回特征与指标
- [x] `POST /predict` 返回 `prediction` 与 `risk_level`
- [x] 缺失特征返回 422 与明确错误信息

## 3. Docker

- [x] `make docker-up` 构建并启动 `:8010`
- [x] `make docker-verify` 通过（建议 `docker-up` 后等待数秒再 verify）

## 4. rag-agent 联动

- [x] rag-agent `check_machine_health` 调用 `POST /predict`
- [x] `make stack-verify`（在 rag-agent 仓库）双服务通过

**推荐验收命令：**

```bash
make docker-up && sleep 5 && make docker-verify
cd ../rag-agent && make stack-verify
```

## 当前未完成项

- [ ] pin scikit-learn 版本，消除 Docker 与本地训练版本警告
- [ ] 生产级模型精度与 A/B 部署
- [ ] CI 流水线

## 验收结论

满足上述勾选项后，可认为本仓库 **POC 级工业预测服务验收通过**；模型为 baseline，**不可用于生产决策**。
