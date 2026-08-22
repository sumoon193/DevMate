"""
测试 fixtures。
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest

from app.schemas.agent import AgentRunResponse
from app.schemas.chunk import Citation, DocumentChunk
from app.schemas.document import DocumentResponse
from app.schemas.enums import DocumentStatus, RunStatus, ToolRiskLevel, Visibility
from app.schemas.tool import ToolDefinition
from app.schemas.user import UserContext

# ── 自动标记 ─────────────────────────────────────────────────────────

_DIR_MARKER_MAP = {
    "unit": pytest.mark.unit,
    "service": pytest.mark.service,
    "api": pytest.mark.api,
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """根据测试文件所在目录自动打上对应 marker。"""
    for item in items:
        fspath = str(item.fspath)
        for dir_name, marker in _DIR_MARKER_MAP.items():
            if f"/{dir_name}/" in fspath or f"\\{dir_name}\\" in fspath:
                item.add_marker(marker)
                break


@pytest.fixture
def fixtures_dir() -> str:
    """测试 fixtures 目录（始终指向 tests/fixtures/）。"""
    return os.path.join(os.path.dirname(__file__), "fixtures")


# ── 标准 HR 文档路径 fixtures ────────────────────────────────────────

_HR_DOCS_DIR = Path(__file__).parent / "fixtures" / "hr_docs"
_RUNTIME_TEST_FILES = Path(__file__).parent.parent / "runtime_test_files"


@pytest.fixture
def tmp_path() -> Path:
    """使用继承仓库 ACL 的临时目录，规避 Windows 0700 权限映射问题。"""
    _RUNTIME_TEST_FILES.mkdir(parents=True, exist_ok=True)
    path = _RUNTIME_TEST_FILES / f"case_{uuid.uuid4().hex[:12]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def hr_onboarding_doc() -> Path:
    """入职制度文档路径。"""
    return _HR_DOCS_DIR / "01_入职制度.md"


@pytest.fixture
def hr_probation_doc() -> Path:
    """转正制度文档路径。"""
    return _HR_DOCS_DIR / "02_转正制度.md"


@pytest.fixture
def hr_reimbursement_doc() -> Path:
    """报销制度文档路径。"""
    return _HR_DOCS_DIR / "03_报销制度.md"


@pytest.fixture
def hr_leave_doc() -> Path:
    """请假制度文档路径（含制度编号 HR-2026-04）。"""
    return _HR_DOCS_DIR / "04_请假制度.md"


@pytest.fixture
def hr_confidential_doc() -> Path:
    """高管薪酬机密文档路径（用于 ACL 测试）。"""
    return _HR_DOCS_DIR / "05_高管薪酬_机密.md"


@pytest.fixture
def hr_flexible_work_doc() -> Path:
    """弹性工作制度文档路径（含制度编号 HR-2026-04）。"""
    return _HR_DOCS_DIR / "06_弹性工作_HR-2026-04.md"


@pytest.fixture
def hr_all_docs() -> list[Path]:
    """所有标准 HR 文档路径列表。"""
    return sorted(_HR_DOCS_DIR.glob("*.md"))


@pytest.fixture
def sample_user_context() -> UserContext:
    """示例用户上下文。"""
    return UserContext(
        user_id="user_001",
        tenant_id="tenant_hr",
        department_ids=["dept_001", "dept_002"],
        role="hr",
        permissions=["hr.document.read", "hr.ticket.write"],
    )


@pytest.fixture
def sample_document() -> DocumentResponse:
    """示例文档响应。"""
    return DocumentResponse(
        id="doc_001",
        title="员工入职与转正管理制度",
        file_path="tenant_hr/2026/05/doc_001/入职转正制度.pdf",
        mime_type="application/pdf",
        status=DocumentStatus.READY,
        tenant_id="tenant_hr",
        department_id="dept_001",
        visibility=Visibility.DEPARTMENT,
        metadata={"author": "HR", "version": "1.0"},
        created_at="2026-05-28T10:00:00Z",
        updated_at="2026-05-28T10:05:00Z",
    )


@pytest.fixture
def sample_agent_run() -> AgentRunResponse:
    """示例 Agent Run 响应。"""
    return AgentRunResponse(
        id="run_001",
        user_id="user_001",
        thread_id="thread_001",
        original_query="新员工入职需要提交哪些材料？",
        status=RunStatus.COMPLETED,
        steps=[],
        tool_calls=[],
        result={
            "answer": "新员工入职需要提交身份证复印件、学历证明、离职证明等材料。",
            "citations": [],
        },
        created_at="2026-05-28T10:00:00Z",
        completed_at="2026-05-28T10:01:00Z",
    )


@pytest.fixture
def sample_tool_definition() -> ToolDefinition:
    """示例工具定义。"""
    return ToolDefinition(
        name="policy_search",
        description="检索制度证据",
        permission_scope="hr.document.read",
        risk_level=ToolRiskLevel.READ,
        requires_approval=False,
        timeout_seconds=10,
        idempotent=True,
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}},
            "required": ["query"],
        },
    )


@pytest.fixture
def sample_citation() -> Citation:
    """示例引用。"""
    return Citation(
        id=1,
        document_name="员工入职与转正管理制度",
        section="第二章 入职材料",
        page=3,
        chunk_text="新员工入职需提交以下材料：1. 身份证复印件；2. 学历证明；3. 离职证明。",
        score=0.92,
        rerank_score=0.95,
    )


@pytest.fixture
def sample_document_chunk() -> DocumentChunk:
    """示例文档分块。"""
    return DocumentChunk(
        id="chunk_001",
        document_id="doc_001",
        chunk_text="新员工入职需提交以下材料：1. 身份证复印件；2. 学历证明；3. 离职证明。",
        context_prefix="本片段来自《员工入职与转正管理制度》第二章'入职材料'，说明了新员工入职时需要提交的必备材料清单。",
        full_text="本片段来自《员工入职与转正管理制度》第二章'入职材料'，说明了新员工入职时需要提交的必备材料清单。新员工入职需提交以下材料：1. 身份证复印件；2. 学历证明；3. 离职证明。",
        parent_id=None,
        chunk_type="child",
        heading_path="员工入职与转正管理制度 > 第二章 入职材料",
        page_numbers=[3],
        token_count=45,
        tenant_id="tenant_hr",
        department_id="dept_001",
        visibility=Visibility.DEPARTMENT,
        acl_metadata={"author": "HR"},
    )
