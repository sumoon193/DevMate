# DevMate

## 项目简介与适用场景

DevMate 是一个面向企业制度和知识库场景的 Agent Runtime。它把文档入库、RAG 检索、证据引用、权限过滤、长流程状态和受控副作用放在同一条可追踪链路中，适合搭建内部制度问答、流程助手和需要审批的自动化任务。

项目包含一个 FastAPI 后端和一个 Vue/Vite 前端。后端默认使用内存适配器，便于本地开发；切换到 `APP_MODE=full` 后可以连接 PostgreSQL、Redis、Milvus、Elasticsearch 和 MinIO。

## 功能清单

- 支持 Markdown、TXT、PDF、DOCX、XLSX、PPTX 文档上传和入库任务查询。
- 入库流程包括解析、切分、Embedding、向量索引、关键词索引和版本记录。
- 检索层支持向量检索、BM25 检索、结果融合、重排和证据引用。
- 每条回答可以返回文档、章节、页码、片段和相关性分数，便于回溯原文。
- 使用租户、部门和可见性字段执行 ACL 过滤，检索结果不会跨权限泄露。
- Agent Run 提供计划、工具调用、审批、事件时间线和 SSE 流式输出。
- Checkpoint 保存执行位置，Event Store 保存业务事件，Projection 提供查询视图。
- 支持沙箱工具、审批决策、UNKNOWN 状态和恢复演练。
- 提供 MCP、A2A、运行指标和 GitHub Webhook 接口。

## 系统架构与核心流程

```text
文档上传 -> 对象存储 -> 解析/切分 -> Embedding + BM25 -> Milvus/Elasticsearch
                                                         |
用户问题 -> ACL 过滤 -> 混合检索 -> 融合/重排 -> 证据包 -> Qwen/Agent Runtime
                                                         |
                         计划 -> 工具调用 -> 审批 -> 沙箱执行 -> 事件/Checkpoint
```

`full` 模式下，PostgreSQL 保存文档、任务、运行和审计数据，Redis 用于 Celery 和缓存，Milvus 保存向量，Elasticsearch 保存关键词索引，MinIO 保存原始文件。模型只负责生成结构化计划或解释，不负责绕过服务端权限和状态校验。

### Checkpoint、Event Store 与 Projection

Checkpoint 只保存执行位置，用于暂停、审批后继续和进程恢复；Event Store 保存不可变的业务事件；Projection 根据事件构建 API 和界面使用的查询视图。需要跨进程恢复时设置 `GRAPH_CHECKPOINTER_BACKEND=postgres`，并通过 `GRAPH_CHECKPOINTER_POSTGRES_URL` 或 `POSTGRES_URL` 提供连接串。

## 技术栈与运行依赖

- Python 3.12、FastAPI、Pydantic、SQLAlchemy、LangGraph
- Vue 3、Vite
- PostgreSQL、Redis、Milvus、Elasticsearch、MinIO
- Qwen/DashScope、OpenTelemetry/Phoenix
- Docker Compose、Pytest

## 目录结构说明

```text
app/api/              FastAPI 路由、请求模型和错误处理
app/services/ingestion 文档解析、切分和入库任务
app/services/retrieval 混合检索、融合和重排
app/services/graph/    LangGraph 编排、Checkpoint 和 SSE
app/services/tools/    受控工具和沙箱执行
app/db/                PostgreSQL ORM 与 CRUD
frontend/              Vue/Vite 前端
scripts/devmate/       live smoke 与恢复演练
tests/                 单元、契约和集成测试
```

## 环境要求

- Python 3.12+
- Node.js 20+（仅开发前端时需要）
- Docker Desktop（运行完整依赖时需要）
- 真实模型验证需要 `QWEN_API_KEY` 和 `QWEN_CHAT_MODEL`

## 本地快速启动

### 仅运行后端

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS 可使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 启动前端

```powershell
npm --prefix frontend install
npm --prefix frontend run dev
```

后端文档地址为 <http://127.0.0.1:8000/docs>，健康检查为 <http://127.0.0.1:8000/health>，前端默认地址为 <http://127.0.0.1:5173>。

## Docker 或中间件启动方式

### 启动完整依赖

```powershell
Copy-Item .env.example .env
docker compose up -d postgres redis minio minio-init etcd milvus elasticsearch
$env:APP_MODE = "full"
$env:GRAPH_CHECKPOINTER_BACKEND = "postgres"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

首次启动后可使用 `docker compose ps` 检查依赖健康状态。真实密钥只放在本机 `.env` 或环境变量中。

## 配置项和环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_MODE` | `fallback` | `fallback` 使用本地适配器，`full` 连接外部依赖 |
| `AGENT_RUN_ENGINE` | `demo` | `demo` 为确定性链路，`langgraph` 启用真实编排 |
| `APPROVAL_MODE` | `manual` | 审批模式，可选 `manual`、`policy`、`auto` |
| `POSTGRES_URL` | 本地示例值 | PostgreSQL 连接串 |
| `GRAPH_CHECKPOINTER_BACKEND` | `memory` | `memory` 或 `postgres` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 地址 |
| `MILVUS_HOST`/`MILVUS_PORT` | `localhost`/`19530` | Milvus 地址 |
| `ES_URL` | `http://localhost:9201` | Elasticsearch 地址 |
| `MINIO_ENDPOINT` | `http://localhost:9000` | 对象存储地址 |
| `INGESTION_EXECUTION_MODE` | `sync` | `sync` 同步入库，`celery` 异步入库 |
| `QWEN_API_KEY` | 空 | Qwen API 密钥，不要提交 |
| `QWEN_CHAT_MODEL` | `qwen-plus` | 文本模型名称 |
| `QWEN_EMBEDDING_MODEL` | `text-embedding-v4` | Embedding 模型 |
| `QWEN_RERANK_MODEL` | `qwen3-rerank` | 重排模型 |

完整示例见 `.env.example`。

## 主要 API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/documents` | 上传并创建入库任务 |
| `GET` | `/ingestions/{task_id}` | 查询入库进度 |
| `POST` | `/agent-runs` | 创建 Agent Run |
| `GET` | `/agent-runs/{run_id}` | 查询运行详情、证据和审批 |
| `GET` | `/agent-runs/{run_id}/stream` | 订阅 SSE 事件 |
| `POST` | `/agent-runs/{run_id}/approvals/{approval_id}` | 提交审批决定 |
| `POST` | `/mcp` | Streamable HTTP MCP 接口 |
| `GET` | `/.well-known/agent-card.json` | A2A Agent Card |

## 请求示例与返回结果

### 上传文档

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents `
  -F "file=@.\README.md" `
  -F "tenant_id=tenant_001" `
  -F "department_id=dept_hr" `
  -F "visibility=department"
```

返回值包含 `id`、`document_version`、`task_id` 和 `status`。使用 `task_id` 查询入库阶段，状态可能为 `queued`、`processing`、`ready` 或 `failed`。

### 创建 Agent Run

```powershell
curl.exe -X POST http://127.0.0.1:8000/agent-runs `
  -H "Content-Type: application/json" `
  -d '{"user_id":"user_001","query":"新员工入职需要哪些材料？"}'
```

响应中的 `run_id` 可用于查询详情或连接 `/stream`。写操作会根据 `APPROVAL_MODE` 返回待审批状态，不能由模型直接执行。

## 离线测试

```powershell
python -m pytest -q
npm --prefix frontend run build
```

离线测试使用 Fake/Recorded 适配器，只验证状态机、权限、接口和数据契约，不代表外部服务已经连通。

## 真实服务验证

```powershell
$env:DEVMATE_BASE_URL = "http://127.0.0.1:8000"
python .\scripts\devmate\live_smoke.py --component health
python .\scripts\devmate\live_smoke.py --component model
```

真实模型验证前设置：

```powershell
$env:QWEN_API_KEY = "本地密钥"
$env:QWEN_CHAT_MODEL = "qwen-plus"
```

live smoke 退出码统一为：`0` 通过，`1` 已连接但验证失败，`2` 缺少服务、密钥或授权。缺少真实配置时只报告 `BLOCKED`，不会将离线结果当作真实通过。

## 常见问题与故障排查

### 端口已被占用

使用 `Get-NetTCPConnection -LocalPort 8000` 查看占用进程，或更换 uvicorn 的 `--port`，同时更新 `DEVMATE_BASE_URL`。

### full 模式启动失败

先运行 `docker compose ps`，确认 PostgreSQL、Redis、Milvus、Elasticsearch 和 MinIO 都是 healthy，再检查 `.env` 中的连接串和端口。

### 入库任务停留在 queued

同步模式使用 `INGESTION_EXECUTION_MODE=sync`。异步模式需要 Redis、Celery broker 和独立 worker，不能只启动 API 进程。

## 安全边界和生产注意事项

- 不提交 `.env`、API key、Cookie、Token、私有日志或生产数据。
- 文档查询必须经过租户、部门和可见性过滤。
- 模型输出必须通过 Pydantic、工具权限和审批校验。
- GitHub、数据库、对象存储和模型的真实验证必须单独配置，缺少授权时明确报告 `blocked`。

## License

Apache-2.0，详见 [LICENSE](LICENSE)。
