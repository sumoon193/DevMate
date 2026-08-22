"""
FastAPI 依赖注入。

组装所有 service 实例，供路由层注入。
- fallback 模式：全部使用 in-memory / fake 实现。
- full 模式：接入真实外部服务（PostgreSQL / MinIO / Milvus / ES / Redis）。
"""

from __future__ import annotations

from app.config import get_settings
from app.core.exceptions import ValidationError
from app.schemas.enums import ToolRiskLevel
from app.schemas.tool import ToolDefinition
from app.services.agent.approval_manager import ApprovalManager
from app.services.agent.approval_policy import build_approval_policy
from app.services.agent.artifact_timeline import ArtifactTimelineBuilder
from app.services.agent.run_manager import AgentRunManager
from app.services.agent.step_logger import StepLogger
from app.services.agent.tool_executor import ToolExecutor
from app.services.agent.tool_registry import ToolRegistry
from app.services.agent.tools.clarification import ClarificationHandler
from app.services.agent.tools.hr_checklist import HRChecklistHandler
from app.services.agent.tools.policy_search import PolicySearchHandler
from app.services.agent.tools.user_profile import UserProfileHandler
from app.services.answer.grounded_answer import (
    FakeAnswerGenerator,
    GroundedAnswerService,
)
from app.services.evaluation.eval_runner import EvalRunner
from app.services.retrieval.embedding.base import Embedder
from app.services.retrieval.reranker.base import Reranker
from app.services.security.acl_validator import ACLValidator


def _build_tool_registry(
    settings: object | None = None,
    *,
    session_factory: object | None = None,
) -> ToolRegistry:
    """注册 V1 五个工具。"""
    registry = ToolRegistry()

    ticket_handler: object
    if getattr(settings, "app_mode", "fallback") == "full":
        if session_factory is None:
            from app.db.session import get_session_factory

            session_factory = get_session_factory()
        from app.services.mcp.adapter import McpToolHandler
        from app.services.mcp.sqlalchemy_server import SqlAlchemyMcpServer

        ticket_handler = McpToolHandler(
            SqlAlchemyMcpServer(session_factory),  # type: ignore[arg-type]
            "create_hr_ticket",
        )
    else:
        from app.services.agent.tools.mock_ticket import MockTicketHandler

        ticket_handler = MockTicketHandler()

    registry.register(
        ToolDefinition(
            name="policy_search",
            description="检索 HR 制度文档",
            permission_scope="hr.document.read",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,
        ),
        PolicySearchHandler(),
    )

    registry.register(
        ToolDefinition(
            name="user_profile",
            description="查询用户档案",
            permission_scope="hr.profile.read",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,
        ),
        UserProfileHandler(),
    )

    registry.register(
        ToolDefinition(
            name="hr_checklist",
            description="获取 HR 流程清单",
            permission_scope="hr.checklist.read",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,
        ),
        HRChecklistHandler(),
    )

    registry.register(
        ToolDefinition(
            name="clarification",
            description="请求用户澄清",
            permission_scope="hr.chat.write",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,
        ),
        ClarificationHandler(),
    )

    registry.register(
        ToolDefinition(
            name="create_hr_ticket",
            description="创建 HR 工单（需审批）",
            permission_scope="hr.ticket.write",
            risk_level=ToolRiskLevel.WRITE,
            requires_approval=True,
        ),
        ticket_handler,  # type: ignore[arg-type]
    )

    return registry


def _build_ai_adapters(settings: object) -> tuple[object, Embedder, Reranker]:
    """fallback 可显式使用 Fake；full 缺少 Qwen 配置时失败关闭。"""
    from app.services.retrieval.embedding.mock_embedding import MockEmbedder
    from app.services.retrieval.reranker.mock_reranker import MockReranker

    api_key = getattr(settings, "qwen_api_key", "")
    if not api_key:
        if getattr(settings, "app_mode", "fallback") == "full":
            raise ValidationError("QWEN_API_KEY is required when APP_MODE=full")
        return (
            FakeAnswerGenerator(),
            MockEmbedder(dimension=getattr(settings, "embedding_dim", 1024)),
            MockReranker(),
        )

    from app.services.ai.qwen import QwenAnswerGenerator, QwenEmbedder, QwenReranker

    timeout_seconds = getattr(settings, "qwen_timeout_seconds", 30.0)
    return (
        QwenAnswerGenerator(
            api_key=api_key,
            model=getattr(settings, "qwen_chat_model", "qwen-plus"),
            base_url=getattr(
                settings,
                "qwen_api_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            timeout_seconds=timeout_seconds,
        ),
        QwenEmbedder(
            api_key=api_key,
            model=getattr(settings, "qwen_embedding_model", "text-embedding-v4"),
            dimension=getattr(settings, "embedding_dim", 1024),
            base_url=getattr(
                settings,
                "qwen_api_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            timeout_seconds=timeout_seconds,
        ),
        QwenReranker(
            api_key=api_key,
            model=getattr(settings, "qwen_rerank_model", "qwen3-rerank"),
            rerank_base_url=getattr(
                settings,
                "qwen_rerank_base_url",
                "https://dashscope.aliyuncs.com/compatible-api/v1",
            ),
            timeout_seconds=timeout_seconds,
        ),
    )


def _build_ragas_metrics(settings: object) -> object:
    """full 使用真实 RAGAS；fallback 使用确定性离线指标。"""
    from app.services.evaluation.ragas_adapter import FakeRAGASMetrics, RealRAGASMetrics

    if getattr(settings, "app_mode", "fallback") != "full":
        return FakeRAGASMetrics()
    api_key = getattr(settings, "qwen_api_key", "")
    if not api_key:
        raise ValidationError("QWEN_API_KEY is required for real RAGAS evaluation")
    return RealRAGASMetrics(
        llm_model=getattr(settings, "qwen_chat_model", "qwen-plus"),
        api_key=api_key,
        embedding_model=getattr(settings, "qwen_embedding_model", "text-embedding-v4"),
        base_url=getattr(
            settings,
            "qwen_api_base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        timeout_seconds=getattr(settings, "ragas_timeout_seconds", 300.0),
        language=getattr(settings, "ragas_language", "chinese"),
    )


def _build_memory_semantic_index(settings: object) -> object:
    """构建生产长期记忆的 Qwen Embedding + Milvus 语义索引。"""
    from app.services.ai.qwen import QwenEmbedder
    from app.services.memory.milvus_index import MilvusMemorySemanticIndex

    dimension = getattr(settings, "embedding_dim", 1024)
    embedder = QwenEmbedder(
        api_key=getattr(settings, "qwen_api_key", ""),
        model=getattr(settings, "qwen_embedding_model", "text-embedding-v4"),
        dimension=dimension,
        base_url=getattr(
            settings,
            "qwen_api_base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        timeout_seconds=getattr(settings, "qwen_timeout_seconds", 30.0),
    )
    return MilvusMemorySemanticIndex(
        embedder=embedder,
        dimension=dimension,
        host=getattr(settings, "milvus_host", "localhost"),
        port=getattr(settings, "milvus_port", 19530),
    )


def _build_mcp_server(settings: object, session_factory: object) -> object:
    """full 使用数据库持久化 MCP；fallback 才允许显式 Fake。"""
    if getattr(settings, "app_mode", "fallback") == "full":
        from app.services.mcp.sqlalchemy_server import SqlAlchemyMcpServer

        return SqlAlchemyMcpServer(session_factory)  # type: ignore[arg-type]
    from app.services.mcp.fake_server import FakeMcpServer

    return FakeMcpServer()


def _build_trace_exporter(settings: object) -> object:
    """根据运行模式选择 trace exporter。"""
    if getattr(settings, "app_mode", "fallback") == "full":
        from app.services.observability.exporters.otel_exporter import OTelTraceExporter

        return OTelTraceExporter(
            endpoint=getattr(settings, "phoenix_endpoint", "http://localhost:6006"),
            service_name="enterprisemind",
            strict=True,
        )

    from app.services.observability.exporters.log_exporter import LogExporter

    return LogExporter()


def _build_tracer(settings: object) -> object:
    """构建统一 Tracer，fallback 输出日志，full 输出到 OTLP/Phoenix。"""
    from app.services.observability.tracer import Tracer

    return Tracer(exporter=_build_trace_exporter(settings))


# ── Service 容器 ─────────────────────────────────────────────────────


class ServiceContainer:
    """
    服务容器，持有全部 service 实例。

    路由层通过 Depends 注入具体 service。
    根据 app_mode 选择 in-memory 或真实外部服务。
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.tracer = _build_tracer(settings)

        # 基础组件（两种模式共用）
        self.step_logger = StepLogger()
        self.timeline_builder = ArtifactTimelineBuilder()
        self.approval_manager = ApprovalManager(step_logger=self.step_logger)
        self.approval_policy = build_approval_policy(
            settings.approval_mode,
            allow_admin=settings.approval_auto_allow_admin,
        )
        self.tool_registry = _build_tool_registry(settings)
        self.acl_validator = ACLValidator()
        self._init_runtime_services()
        self.tool_executor = ToolExecutor(
            registry=self.tool_registry,
            approval_manager=self.approval_manager,
            step_logger=self.step_logger,
            acl_validator=self.acl_validator,
            side_effect_ledger=self.side_effect_ledger,
        )

        # 根据模式选择服务实现
        if settings.app_mode == "full":
            self._init_full_mode(settings)
        else:
            self._init_fallback_mode()

    def _init_fallback_mode(self) -> None:
        """初始化 fallback 模式（全部 in-memory）。"""
        from app.services.ingestion.store import InMemoryIngestionTaskStore
        from app.services.retrieval.embedding.mock_embedding import MockEmbedder
        from app.services.retrieval.hybrid import HybridRetriever
        from app.services.retrieval.reranker.mock_reranker import MockReranker
        from app.services.retrieval.store.memory_bm25 import InMemoryBM25Store
        from app.services.retrieval.store.memory_vector import InMemoryVectorStore
        from app.services.storage.local_storage import LocalFileStorage

        self.answer_generator = FakeAnswerGenerator()
        self.embedder = MockEmbedder()
        self.reranker = MockReranker()
        self.vector_store = InMemoryVectorStore()
        self.bm25_store = InMemoryBM25Store()
        self.storage = LocalFileStorage(base_dir=self.settings.local_storage_dir)
        self.ingestion_task_store = InMemoryIngestionTaskStore()

        self.run_manager = AgentRunManager(
            tool_executor=self.tool_executor,
            approval_manager=self.approval_manager,
            step_logger=self.step_logger,
            approval_policy=self.approval_policy,
        )
        self.answer_service = GroundedAnswerService(
            answer_generator=self.answer_generator,
        )
        self.eval_runner = EvalRunner(
            answer_service=self.answer_service,
        )
        self.hybrid_retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
            reranker=self.reranker,
            document_version_registry=self.document_version_registry,
        )
        self._init_graph_runner()

    def _init_full_mode(self, settings: object) -> None:
        """初始化 full 模式（接入真实外部服务）。"""
        from app.db.session import get_session_factory
        from app.services.retrieval.hybrid import HybridRetriever

        session_factory = get_session_factory()

        # 真实 MinIO 存储
        from app.services.storage.minio_storage import MinIOStorage

        self.storage = MinIOStorage(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
        )

        from app.services.ingestion.worker import build_task_store

        self.ingestion_task_store = build_task_store(settings)

        # 真实 Milvus 向量存储
        from app.services.retrieval.store.milvus_vector import MilvusVectorStore

        self.vector_store = MilvusVectorStore(
            host=settings.milvus_host,
            port=settings.milvus_port,
            dim=settings.embedding_dim,
        )

        # 真实 ES BM25 存储
        from app.services.retrieval.store.es_bm25 import ElasticsearchBM25Store

        self.bm25_store = ElasticsearchBM25Store(es_url=settings.es_url)

        # full 模式只允许真实 AI adapter；缺少配置会在装配时失败关闭。
        answer_generator, embedder, reranker = _build_ai_adapters(settings)
        self.answer_generator = answer_generator
        self.embedder = embedder
        self.reranker = reranker

        # 混合检索器
        self.hybrid_retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
            reranker=self.reranker,
            document_version_registry=self.document_version_registry,
        )

        # Redis 速率限制器
        from app.services.security.redis_rate_limiter import RedisRateLimiter

        self.rate_limiter = RedisRateLimiter(redis_url=settings.redis_url)

        # Run Manager（带 PostgreSQL 持久化）
        self.run_manager = AgentRunManager(
            tool_executor=self.tool_executor,
            approval_manager=self.approval_manager,
            step_logger=self.step_logger,
            session_factory=session_factory,
            approval_policy=self.approval_policy,
        )

        self.answer_service = GroundedAnswerService(answer_generator=self.answer_generator)
        self.eval_runner = EvalRunner(
            answer_service=self.answer_service,
            ragas_metrics=_build_ragas_metrics(settings),
        )

        self._init_graph_runner()

    def _init_graph_runner(self) -> None:
        """构建 LangGraph Runner，供可选真实编排入口使用。"""
        from app.services.graph.checkpointer import create_checkpointer_manager
        from app.services.graph.graph import create_agent_graph
        from app.services.graph.runner import GraphRunner

        # checkpointer 后端由 settings 决定；postgres 后端的连接池
        # 生命周期挂在 FastAPI lifespan 上（见 app/main.py）。
        self.graph_checkpointer = create_checkpointer_manager(self.settings)
        compiled_graph = create_agent_graph(
            run_manager=self.run_manager,
            hybrid_retriever=self.hybrid_retriever,
            answer_service=self.answer_service,
            checkpointer=self.graph_checkpointer.checkpointer,
        )
        self.graph_runner = GraphRunner(
            graph=compiled_graph,
            run_manager=self.run_manager,
            step_logger=self.step_logger,
            tracer=self.tracer,
        )

    def _init_runtime_services(self) -> None:
        """组装长期 Case、Memory、Skill、MCP 与 A2A fallback 服务。"""
        from app.services.a2a.policy_research import (
            InProcessA2AClient,
            PolicyResearchA2AAgent,
        )
        from app.services.agent.tool_executor import ToolExecutor
        from app.services.agent.tool_registry import ToolRegistry
        from app.services.context.compactor import ContextCompactor
        from app.services.mcp.adapter import (
            McpApprovalBridge,
            McpToolAdapter,
            McpToolDiscovery,
        )
        from app.services.mcp.protocol_server import LocalMcpProtocolServer
        from app.services.memory.store import InMemoryEpisodicMemoryStore
        from app.services.observability.runtime_metrics import RuntimeMetrics
        from app.services.runtime.case_service import CaseService
        from app.services.runtime.clock import SystemClock
        from app.services.runtime.event_store import InMemoryEventStore
        from app.services.runtime.lease import InMemoryLeaseStore
        from app.services.runtime.onboarding_workflow import OnboardingCaseWorkflow
        from app.services.runtime.side_effects import InMemorySideEffectLedger
        from app.services.runtime.timer_coordinator import TimerCoordinator
        from app.services.runtime.timers import InMemoryTimerStore
        from app.services.skills.registry import SkillRegistry

        clock = SystemClock()
        self.runtime_metrics = RuntimeMetrics()
        if self.settings.app_mode == "full":
            from app.db.session import get_session_factory
            from app.services.ingestion.document_versions import (
                SqlAlchemyDocumentVersionRegistry,
            )
            from app.services.runtime.sqlalchemy_adapters import (
                SqlAlchemyCaseProjectionStore,
                SqlAlchemyEventStore,
                SqlAlchemyLeaseStore,
                SqlAlchemySideEffectLedger,
                SqlAlchemyTimerStore,
            )

            runtime_sessions = get_session_factory()
            self.event_store = SqlAlchemyEventStore(
                runtime_sessions,
                metrics=self.runtime_metrics,
            )
            self.projection_store = SqlAlchemyCaseProjectionStore(runtime_sessions)
            self.case_service = CaseService(
                event_store=self.event_store,
                projection_store=self.projection_store,
                metrics=self.runtime_metrics,
            )
            self.timer_store = SqlAlchemyTimerStore(runtime_sessions)
            self.side_effect_ledger = SqlAlchemySideEffectLedger(runtime_sessions)
            self.lease_store = SqlAlchemyLeaseStore(runtime_sessions)
            self.document_version_registry = SqlAlchemyDocumentVersionRegistry(runtime_sessions)
        else:
            from app.services.ingestion.document_versions import (
                InMemoryDocumentVersionRegistry,
            )

            self.event_store = InMemoryEventStore(metrics=self.runtime_metrics)
            self.projection_store = None
            self.case_service = CaseService(
                event_store=self.event_store,
                metrics=self.runtime_metrics,
            )
            self.timer_store = InMemoryTimerStore(clock=clock)
            self.side_effect_ledger = InMemorySideEffectLedger(clock=clock)
            self.lease_store = InMemoryLeaseStore(clock=clock)
            self.document_version_registry = InMemoryDocumentVersionRegistry()
        self.timer_coordinator = TimerCoordinator(
            case_service=self.case_service,
            timer_store=self.timer_store,
        )
        self.policy_research_agent = PolicyResearchA2AAgent()
        self.a2a_client = InProcessA2AClient(self.policy_research_agent)
        self.context_compactor = ContextCompactor()
        if self.settings.app_mode == "full":
            from app.services.memory.store import SqlAlchemyEpisodicMemoryStore

            self.memory_store = SqlAlchemyEpisodicMemoryStore(
                runtime_sessions,
                clock=clock,
                semantic_index=_build_memory_semantic_index(self.settings),
            )
        else:
            self.memory_store = InMemoryEpisodicMemoryStore(clock=clock)
        self.skill_registry = SkillRegistry(
            allowed_source_prefixes=["repo://skills/"],
            activation_threshold=0.9,
            clock=clock,
        )
        onboarding_skill = self.skill_registry.register(
            name="hr_onboarding",
            version="1.0.0",
            content="先研究当前制度，再生成跨角色计划；所有写操作必须审批。",
            source_uri="repo://skills/hr_onboarding/1.0.0",
            allowed_tools=["create_hr_ticket"],
            required_permissions=["hr.document.read", "hr.ticket.write"],
        )
        self.skill_registry.activate(onboarding_skill.id, eval_score=0.98)

        mcp_registry = ToolRegistry()
        mcp_executor = ToolExecutor(
            registry=mcp_registry,
            approval_manager=self.approval_manager,
            step_logger=self.step_logger,
            acl_validator=self.acl_validator,
            side_effect_ledger=self.side_effect_ledger,
        )
        self.mcp_adapter = McpToolAdapter(
            McpToolDiscovery(
                _build_mcp_server(
                    self.settings,
                    runtime_sessions if self.settings.app_mode == "full" else None,
                )
            ),
            mcp_registry,
            mcp_executor,
        )
        self.mcp_adapter.register_discovered_tools()
        self.mcp_approval_bridge = McpApprovalBridge(
            mcp_executor,
            self.approval_manager,
        )
        self.mcp_protocol_server = LocalMcpProtocolServer(
            tool_adapter=self.mcp_adapter,
            resources={
                "policy://hr/onboarding": {
                    "name": "员工入职与转正制度",
                    "mimeType": "text/markdown",
                    "text": "新员工入职材料、试用期目标与转正评估要求。",
                }
            },
            prompts={"plan_hr_case": "基于制度证据生成长期 Case 计划，写操作必须审批。"},
        )
        self.onboarding_workflow = OnboardingCaseWorkflow(
            case_service=self.case_service,
            event_store=self.event_store,
            skill_registry=self.skill_registry,
            memory_store=self.memory_store,
            context_compactor=self.context_compactor,
            a2a_client=self.a2a_client,
            mcp_adapter=self.mcp_adapter,
            mcp_approval_bridge=self.mcp_approval_bridge,
            approval_manager=self.approval_manager,
            timer_coordinator=self.timer_coordinator,
            clock=clock,
            metrics=self.runtime_metrics,
            tracer=self.tracer,
        )


_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    """获取全局服务容器（单例）。"""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def reset_container() -> None:
    """重置容器（测试用）。"""
    global _container
    _container = None
