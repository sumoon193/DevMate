"""
Schema 测试。

按模块规范要求的 5 个测试：
1. test_agent_run_status_transition_values_are_stable
2. test_tool_definition_requires_approval_for_write_tools
3. test_citation_serializes_source_page_section_score
4. test_document_chunk_contains_acl_metadata
5. test_schema_roundtrip_between_api_and_db_shape
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas.agent import AgentRunResponse
from app.schemas.approval import ApprovalDecision, ApprovalRequest
from app.schemas.chunk import Citation, DocumentChunk, EvidenceBundle
from app.schemas.document import DocumentResponse
from app.schemas.enums import DocumentStatus, RunStatus, ToolRiskLevel, Visibility
from app.schemas.tool import ToolDefinition
from app.schemas.user import UserContext


class TestRunStatusStability:
    """测试 1：AgentRun 状态转换值稳定。"""

    def test_agent_run_status_transition_values_are_stable(self):
        """验证 RunStatus 枚举值与规范文档一致。"""
        # 检查所有状态值
        expected_statuses = {
            "created": RunStatus.CREATED,
            "running": RunStatus.RUNNING,
            "retrieving_evidence": RunStatus.RETRIEVING_EVIDENCE,
            "planning": RunStatus.PLANNING,
            "awaiting_approval": RunStatus.AWAITING_APPROVAL,
            "resumed": RunStatus.RESUMED,
            "completed": RunStatus.COMPLETED,
            "failed": RunStatus.FAILED,
            "cancelled": RunStatus.CANCELLED,
        }

        for value, status in expected_statuses.items():
            assert status.value == value, f"RunStatus.{status.name}.value should be '{value}'"

        # 验证总状态数
        assert len(RunStatus) == 9, f"Expected 9 RunStatus values, got {len(RunStatus)}"

    def test_document_status_values_are_stable(self):
        """验证 DocumentStatus 枚举值与规范文档一致。"""
        expected_statuses = {
            "queued": DocumentStatus.QUEUED,
            "parsing": DocumentStatus.PARSING,
            "chunking": DocumentStatus.CHUNKING,
            "embedding": DocumentStatus.EMBEDDING,
            "indexing": DocumentStatus.INDEXING,
            "ready": DocumentStatus.READY,
            "failed": DocumentStatus.FAILED,
        }

        for value, status in expected_statuses.items():
            assert status.value == value, f"DocumentStatus.{status.name}.value should be '{value}'"

        assert len(DocumentStatus) == 7, (
            f"Expected 7 DocumentStatus values, got {len(DocumentStatus)}"
        )

    def test_tool_risk_level_values_are_stable(self):
        """验证 ToolRiskLevel 枚举值与规范文档一致。"""
        expected_levels = {
            "read": ToolRiskLevel.READ,
            "write": ToolRiskLevel.WRITE,
            "admin": ToolRiskLevel.ADMIN,
        }

        for value, level in expected_levels.items():
            assert level.value == value, f"ToolRiskLevel.{level.name}.value should be '{value}'"

        assert len(ToolRiskLevel) == 3, f"Expected 3 ToolRiskLevel values, got {len(ToolRiskLevel)}"


class TestToolApprovalRules:
    """测试 2：工具审批规则。"""

    def test_tool_definition_requires_approval_for_write_tools(self):
        """验证 WRITE 级别工具 requires_approval=True。"""
        write_tool = ToolDefinition(
            name="create_ticket",
            description="创建工单",
            permission_scope="hr.ticket.write",
            risk_level=ToolRiskLevel.WRITE,
            requires_approval=True,  # 必须为 True
            timeout_seconds=10,
            idempotent=True,
            parameters_schema={},
        )

        assert write_tool.risk_level == ToolRiskLevel.WRITE
        assert write_tool.requires_approval is True

    def test_tool_definition_requires_approval_for_admin_tools(self):
        """验证 ADMIN 级别工具 requires_approval=True。"""
        admin_tool = ToolDefinition(
            name="delete_user",
            description="删除用户",
            permission_scope="admin.user.delete",
            risk_level=ToolRiskLevel.ADMIN,
            requires_approval=True,  # 必须为 True
            timeout_seconds=10,
            idempotent=False,
            parameters_schema={},
        )

        assert admin_tool.risk_level == ToolRiskLevel.ADMIN
        assert admin_tool.requires_approval is True

    def test_tool_definition_no_approval_for_read_tools(self):
        """验证 READ 级别工具 requires_approval=False。"""
        read_tool = ToolDefinition(
            name="policy_search",
            description="检索制度证据",
            permission_scope="hr.document.read",
            risk_level=ToolRiskLevel.READ,
            requires_approval=False,  # 应该为 False
            timeout_seconds=10,
            idempotent=True,
            parameters_schema={},
        )

        assert read_tool.risk_level == ToolRiskLevel.READ
        assert read_tool.requires_approval is False


class TestCitationSerialization:
    """测试 3：Citation 序列化。"""

    def test_citation_serializes_source_page_section_score(self, sample_citation):
        """验证 Citation 序列化包含所有必需字段。"""
        # 序列化为字典
        citation_dict = sample_citation.model_dump()

        # 验证必需字段
        assert "id" in citation_dict
        assert "document_name" in citation_dict
        assert "section" in citation_dict
        assert "page" in citation_dict
        assert "chunk_text" in citation_dict
        assert "score" in citation_dict
        assert "rerank_score" in citation_dict

        # 验证字段值
        assert citation_dict["id"] == 1
        assert citation_dict["document_name"] == "员工入职与转正管理制度"
        assert citation_dict["section"] == "第二章 入职材料"
        assert citation_dict["page"] == 3
        assert 0.0 <= citation_dict["score"] <= 1.0
        assert 0.0 <= citation_dict["rerank_score"] <= 1.0

    def test_citation_score_range_validation(self):
        """验证 score 字段范围 0.0-1.0。"""
        # 合法的分数
        valid_citation = Citation(
            id=1,
            document_name="test",
            section="test",
            page=1,
            chunk_text="test",
            score=0.95,
            rerank_score=0.88,
        )
        assert valid_citation.score == 0.95

        # 非法的分数（应该抛出 ValidationError）
        with pytest.raises(ValidationError):
            Citation(
                id=1,
                document_name="test",
                section="test",
                page=1,
                chunk_text="test",
                score=1.5,  # 超出范围
                rerank_score=0.88,
            )

        with pytest.raises(ValidationError):
            Citation(
                id=1,
                document_name="test",
                section="test",
                page=1,
                chunk_text="test",
                score=0.95,
                rerank_score=-0.1,  # 超出范围
            )


class TestDocumentChunkACL:
    """测试 4：DocumentChunk ACL 元数据。"""

    def test_document_chunk_contains_acl_metadata(self, sample_document_chunk):
        """验证 DocumentChunk 包含 ACL 元数据字段。"""
        # 验证必需的 ACL 字段
        assert sample_document_chunk.tenant_id == "tenant_hr"
        assert sample_document_chunk.department_id == "dept_001"
        assert sample_document_chunk.visibility == Visibility.DEPARTMENT
        assert sample_document_chunk.acl_metadata == {"author": "HR"}

    def test_document_chunk_acl_metadata_default(self):
        """验证 acl_metadata 字段有默认值。"""
        chunk = DocumentChunk(
            id="chunk_002",
            document_id="doc_001",
            chunk_text="test",
            tenant_id="tenant_hr",
            department_id="dept_001",
            visibility=Visibility.PUBLIC,
        )

        # acl_metadata 应该默认为空字典
        assert chunk.acl_metadata == {}

    def test_document_chunk_visibility_enum(self):
        """验证 visibility 字段使用 Visibility 枚举。"""
        chunk = DocumentChunk(
            id="chunk_003",
            document_id="doc_001",
            chunk_text="test",
            tenant_id="tenant_hr",
            department_id="dept_001",
            visibility=Visibility.CONFIDENTIAL,
        )

        assert chunk.visibility == Visibility.CONFIDENTIAL
        assert chunk.visibility.value == "confidential"


class TestSchemaRoundtrip:
    """测试 5：Schema 与 DB Shape 转换。"""

    def test_schema_roundtrip_between_api_and_db_shape(self, sample_document):
        """验证 Pydantic schema 可以从 SQLAlchemy model 转换。"""
        # 模拟从 SQLAlchemy model 转换到 Pydantic schema
        # （实际测试中需要真实 SQLAlchemy model，这里测试 model_validate 逻辑）

        # 创建一个字典模拟 SQLAlchemy model 的属性
        db_shape = {
            "id": "doc_001",
            "title": "员工入职与转正管理制度",
            "file_path": "tenant_hr/2026/05/doc_001/入职转正制度.pdf",
            "mime_type": "application/pdf",
            "status": "ready",
            "tenant_id": "tenant_hr",
            "department_id": "dept_001",
            "visibility": "department",
            "metadata": {"author": "HR", "version": "1.0"},
            "created_at": "2026-05-28T10:00:00Z",
            "updated_at": "2026-05-28T10:05:00Z",
        }

        # 使用 model_validate 转换
        doc = DocumentResponse.model_validate(db_shape)

        # 验证转换结果
        assert doc.id == "doc_001"
        assert doc.title == "员工入职与转正管理制度"
        assert doc.status == DocumentStatus.READY
        assert doc.visibility == Visibility.DEPARTMENT

    def test_agent_run_schema_from_dict(self):
        """验证 AgentRun schema 可以从字典创建。"""
        run_data = {
            "id": "run_001",
            "user_id": "user_001",
            "thread_id": "thread_001",
            "original_query": "测试查询",
            "status": "created",
            "created_at": "2026-05-28T10:00:00Z",
        }

        run = AgentRunResponse.model_validate(run_data)

        assert run.id == "run_001"
        assert run.status == RunStatus.CREATED
        assert run.steps == []
        assert run.tool_calls == []
        assert run.result is None

    def test_tool_definition_schema_from_dict(self):
        """验证 ToolDefinition schema 可以从字典创建。"""
        tool_data = {
            "name": "policy_search",
            "description": "检索制度证据",
            "permission_scope": "hr.document.read",
            "risk_level": "read",
            "requires_approval": False,
        }

        tool = ToolDefinition.model_validate(tool_data)

        assert tool.name == "policy_search"
        assert tool.risk_level == ToolRiskLevel.READ
        assert tool.requires_approval is False
        assert tool.timeout_seconds == 30  # 默认值


class TestUserContext:
    """UserContext 测试。"""

    def test_user_context_creation(self, sample_user_context):
        """验证 UserContext 创建。"""
        assert sample_user_context.user_id == "user_001"
        assert sample_user_context.tenant_id == "tenant_hr"
        assert len(sample_user_context.department_ids) == 2
        assert sample_user_context.role == "hr"
        assert "hr.document.read" in sample_user_context.permissions

    def test_user_context_optional_fields(self):
        """验证 UserContext 可选字段有默认值。"""
        user = UserContext(user_id="user_002", tenant_id="tenant_eng", role="employee")

        assert user.department_ids == []
        assert user.permissions == []


class TestEvidenceBundle:
    """EvidenceBundle 测试。"""

    def test_evidence_bundle_creation(self, sample_citation):
        """验证 EvidenceBundle 创建。"""
        bundle = EvidenceBundle(
            evidence_list=[sample_citation], total_count=1, query_coverage_score=0.92
        )

        assert len(bundle.evidence_list) == 1
        assert bundle.total_count == 1
        assert bundle.query_coverage_score == 0.92

    def test_evidence_bundle_score_range(self):
        """验证 query_coverage_score 范围。"""
        # 合法的分数
        bundle = EvidenceBundle(evidence_list=[], total_count=0, query_coverage_score=0.5)
        assert bundle.query_coverage_score == 0.5

        # 非法的分数
        with pytest.raises(ValidationError):
            EvidenceBundle(evidence_list=[], total_count=0, query_coverage_score=1.5)


class TestDocumentStatusProgress:
    """测试 DocumentStatusResponse 的 progress 字段验证。"""

    def test_progress_valid_range(self):
        """验证 progress 字段范围 0.0-1.0。"""
        from app.schemas.document import DocumentStatusResponse
        from app.schemas.enums import DocumentStatus

        # 合法的进度值
        status = DocumentStatusResponse(
            id="doc_001",
            status=DocumentStatus.PARSING,
            progress=0.5,
            updated_at="2026-05-28T10:00:00Z",
        )
        assert status.progress == 0.5

        # 边界值测试
        status_zero = DocumentStatusResponse(
            id="doc_002",
            status=DocumentStatus.QUEUED,
            progress=0.0,
            updated_at="2026-05-28T10:00:00Z",
        )
        assert status_zero.progress == 0.0

        status_full = DocumentStatusResponse(
            id="doc_003",
            status=DocumentStatus.READY,
            progress=1.0,
            updated_at="2026-05-28T10:00:00Z",
        )
        assert status_full.progress == 1.0

    def test_progress_out_of_range_raises_error(self):
        """验证 progress 超出范围时抛出 ValidationError。"""
        from app.schemas.document import DocumentStatusResponse
        from app.schemas.enums import DocumentStatus

        with pytest.raises(ValidationError):
            DocumentStatusResponse(
                id="doc_001",
                status=DocumentStatus.PARSING,
                progress=1.5,  # 超出范围
                updated_at="2026-05-28T10:00:00Z",
            )

        with pytest.raises(ValidationError):
            DocumentStatusResponse(
                id="doc_001",
                status=DocumentStatus.PARSING,
                progress=-0.1,  # 超出范围
                updated_at="2026-05-28T10:00:00Z",
            )


class TestApprovalRequest:
    """测试 ApprovalRequest Schema。"""

    def test_approval_request_creation(self):
        """验证 ApprovalRequest 创建。"""
        from app.schemas.enums import ToolRiskLevel

        approval = ApprovalRequest(
            id="appr_001",
            run_id="run_001",
            tool_call_id="tool_001",
            tool_name="create_ticket",
            parameters={"title": "入职申请"},
            expected_effect="创建一个入职工单",
            evidence=[],
            risk_level=ToolRiskLevel.WRITE,
            options=["approve", "edit", "reject"],
        )

        assert approval.id == "appr_001"
        assert approval.status == "pending"
        assert approval.decision is None
        assert approval.decided_by is None
        assert approval.decided_at is None

    def test_approval_decision_with_edited_params(self):
        """验证 ApprovalDecision 带编辑参数。"""
        from app.schemas.approval import ApprovalDecision

        decision = ApprovalDecision(
            decision="edit", edited_parameters={"title": "修改后的入职申请", "priority": "high"}
        )

        assert decision.decision == "edit"
        assert decision.edited_parameters is not None
        assert decision.edited_parameters["title"] == "修改后的入职申请"


class TestOrmModels:
    """测试 SQLAlchemy ORM 模型可配置和建表。"""

    def test_models_package_import_registers_all_tables(self):
        """验证导入 app.models 包即可注册所有 ORM 表。"""
        import app.models  # noqa: F401
        from app.models.base import Base

        assert {
            "documents",
            "document_chunks",
            "agent_runs",
            "agent_steps",
            "tool_calls",
            "approval_requests",
            "eval_cases",
            "eval_runs",
            "ingestion_tasks",
        }.issubset(Base.metadata.tables.keys())

    def test_sqlalchemy_models_can_configure_and_create_tables(self):
        """验证所有 ORM 模型可以导入、配置并在 SQLite 内存库建表。"""
        from sqlalchemy import create_engine

        import app.models.agent_run
        import app.models.agent_step
        import app.models.approval
        import app.models.chunk
        import app.models.document
        import app.models.eval
        import app.models.tool_call  # noqa: F401
        from app.models.base import Base

        Base.registry.configure()
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        assert {
            "documents",
            "document_chunks",
            "agent_runs",
            "agent_steps",
            "tool_calls",
            "approval_requests",
            "eval_cases",
            "eval_runs",
            "ingestion_tasks",
        }.issubset(Base.metadata.tables.keys())


class TestSchemaSafetyConstraints:
    """测试 schema 层的安全约束。"""

    def test_tool_definition_rejects_write_tool_without_approval(self):
        """验证 WRITE 工具不能关闭审批。"""
        with pytest.raises(ValidationError):
            ToolDefinition(
                name="create_ticket",
                description="创建工单",
                permission_scope="hr.ticket.write",
                risk_level=ToolRiskLevel.WRITE,
                requires_approval=False,
            )

    def test_tool_definition_rejects_admin_tool_without_approval(self):
        """验证 ADMIN 工具不能关闭审批。"""
        with pytest.raises(ValidationError):
            ToolDefinition(
                name="delete_user",
                description="删除用户",
                permission_scope="admin.user.delete",
                risk_level=ToolRiskLevel.ADMIN,
                requires_approval=False,
            )

    def test_approval_decision_rejects_unknown_value(self):
        """验证审批决策只允许 approve/edit/reject。"""
        with pytest.raises(ValidationError):
            ApprovalDecision(decision="maybe")

    def test_document_response_reads_metadata_from_orm_metadata_alias(self):
        """验证 DocumentResponse 可以从 ORM 的 metadata_ 字段读取元数据。"""

        class DocumentOrmShape:
            id = "doc_001"
            title = "员工入职与转正管理制度"
            file_path = "tenant_hr/2026/05/doc_001/入职转正制度.pdf"
            mime_type = "application/pdf"
            status = DocumentStatus.READY
            tenant_id = "tenant_hr"
            department_id = "dept_001"
            visibility = Visibility.DEPARTMENT
            metadata_ = {"author": "HR", "version": "1.0"}
            created_at = datetime.fromisoformat("2026-05-28T10:00:00+00:00")
            updated_at = datetime.fromisoformat("2026-05-28T10:05:00+00:00")

        document = DocumentResponse.model_validate(DocumentOrmShape())

        assert document.metadata == {"author": "HR", "version": "1.0"}
