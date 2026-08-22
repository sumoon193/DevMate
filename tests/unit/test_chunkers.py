"""
Chunker 测试。

按模块规范要求的测试：
1. test_chunker_creates_parent_child_relationship
2. test_table_chunk_keeps_header_context
3. test_contextual_prefix_uses_source_metadata
4. test_chunk_page_numbers_are_preserved
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from app.schemas.enums import Visibility
from app.services.chunker.base import ChunkConfig
from app.services.chunker.context_enricher import ContextEnricher
from app.services.chunker.hybrid import HybridChunker
from app.services.chunker.semantic import SemanticChunker
from app.services.chunker.structural import StructuralChunker
from app.services.parser.base import ParsedDocument
from app.services.parser.markdown_parser import MarkdownParser


@pytest.fixture
def sample_md_path(fixtures_dir: str) -> str:
    """示例 Markdown 文件路径。"""
    return os.path.join(fixtures_dir, "sample.md")


@pytest_asyncio.fixture
async def parsed_md_doc(sample_md_path: str) -> ParsedDocument:
    """解析后的 Markdown 文档。"""
    parser = MarkdownParser()
    return await parser.parse(
        file_path=sample_md_path,
        document_id="doc_001",
        metadata={
            "tenant_id": "tenant_hr",
            "department_id": "dept_001",
            "visibility": "department",
        },
    )


@pytest.fixture
def chunk_config() -> ChunkConfig:
    """分块配置。"""
    return ChunkConfig(
        max_parent_tokens=1000,
        max_child_tokens=300,
        min_child_tokens=50,
        overlap_tokens=30,
        preserve_tables=True,
    )


class TestHybridChunker:
    """Hybrid Chunker 测试。"""

    @pytest.mark.asyncio
    async def test_chunker_creates_parent_child_relationship(
        self, parsed_md_doc: ParsedDocument, chunk_config: ChunkConfig
    ):
        """测试 1：Chunker 创建 parent-child 关系。"""
        chunker = HybridChunker()

        chunks = await chunker.chunk(parsed_md_doc, chunk_config)

        # 分离 parent 和 child chunks
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        child_chunks = [c for c in chunks if c.chunk_type == "child"]

        # 验证存在 parent 和 child
        assert len(parent_chunks) > 0, "Should have at least one parent chunk"
        assert len(child_chunks) > 0, "Should have at least one child chunk"

        # 验证 parent chunk 的 parent_id 为 None
        for parent in parent_chunks:
            assert parent.parent_id is None, "Parent chunk should have parent_id=None"

        # 验证每个 child 都有 parent_id
        for child in child_chunks:
            assert child.parent_id is not None, "Child chunk should have parent_id"
            assert child.parent_id.startswith("chunk_"), "Parent ID should start with 'chunk_'"

        # 验证 child 数量大于等于 parent 数量
        assert len(child_chunks) >= len(parent_chunks), (
            "Should have at least as many child chunks as parent chunks"
        )

    @pytest.mark.asyncio
    async def test_table_chunk_keeps_header_context(
        self, parsed_md_doc: ParsedDocument, chunk_config: ChunkConfig
    ):
        """测试 2：表格 chunk 保留表头上下文。"""
        chunker = HybridChunker()

        chunks = await chunker.chunk(parsed_md_doc, chunk_config)

        # 查找包含表格内容的 chunk
        table_chunks = [c for c in chunks if "负责人" in c.chunk_text or "事项" in c.chunk_text]

        assert len(table_chunks) > 0, "Should have chunks containing table content"

        # 验证表格内容完整（包含表头）
        for chunk in table_chunks:
            # 表格应该包含表头（事项、负责人、时间）
            if "负责人" in chunk.chunk_text:
                assert "事项" in chunk.chunk_text, "Table chunk should include header '事项'"
                assert "时间" in chunk.chunk_text, "Table chunk should include header '时间'"

    @pytest.mark.asyncio
    async def test_contextual_prefix_uses_source_metadata(
        self, parsed_md_doc: ParsedDocument, chunk_config: ChunkConfig
    ):
        """测试 3：Contextual prefix 使用源文档元数据。"""
        chunker = HybridChunker()

        chunks = await chunker.chunk(parsed_md_doc, chunk_config)

        # 验证所有 chunk 都有 context_prefix
        for chunk in chunks:
            assert chunk.context_prefix, "Chunk should have context_prefix"
            assert len(chunk.context_prefix) > 10, "Context prefix should have meaningful length"

        # 验证 prefix 包含文档名
        for chunk in chunks:
            assert "sample.md" in chunk.context_prefix or "入职" in chunk.context_prefix, (
                "Context prefix should reference document name"
            )

        # 验证有标题路径的 chunk 包含章节信息
        chunks_with_heading = [c for c in chunks if c.heading_path]
        for chunk in chunks_with_heading[:5]:  # 检查前 5 个
            # heading_path 的内容应该出现在 prefix 中
            heading_parts = chunk.heading_path.split(" > ")
            # 至少有一个部分出现在 prefix 中
            assert any(part in chunk.context_prefix for part in heading_parts if len(part) > 2), (
                "Context prefix should include heading path content"
            )

    @pytest.mark.asyncio
    async def test_chunk_page_numbers_are_preserved(
        self, parsed_md_doc: ParsedDocument, chunk_config: ChunkConfig
    ):
        """测试 4：chunk 页码被保留。"""
        chunker = HybridChunker()

        chunks = await chunker.chunk(parsed_md_doc, chunk_config)

        # 验证所有 chunk 都有 page_numbers
        for chunk in chunks:
            assert isinstance(chunk.page_numbers, list), "page_numbers should be a list"
            assert len(chunk.page_numbers) > 0, "page_numbers should not be empty"
            assert all(isinstance(p, int) for p in chunk.page_numbers), (
                "page_numbers should contain integers"
            )

        # 验证 child chunk 继承了 parent 的页码
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        child_chunks = [c for c in chunks if c.chunk_type == "child"]

        for parent in parent_chunks:
            # 找到对应的 child chunks
            parent_children = [c for c in child_chunks if c.parent_id == parent.document_id]
            for child in parent_children:
                # child 的页码应该是 parent 页码的子集
                assert set(child.page_numbers).issubset(set(parent.page_numbers)), (
                    "Child page_numbers should be subset of parent page_numbers"
                )


class TestStructuralChunker:
    """Structural Chunker 测试。"""

    @pytest.mark.asyncio
    async def test_structural_chunker_creates_parent_chunks(
        self, parsed_md_doc: ParsedDocument, chunk_config: ChunkConfig
    ):
        """测试 Structural Chunker 创建 parent chunks。"""
        chunker = StructuralChunker()

        chunks = await chunker.chunk(parsed_md_doc, chunk_config)

        # 验证创建了 parent chunks
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.chunk_type == "parent"
            assert chunk.parent_id is None
            assert chunk.heading_path
            assert chunk.chunk_text

    @pytest.mark.asyncio
    async def test_structural_chunker_preserves_heading_hierarchy(
        self, parsed_md_doc: ParsedDocument, chunk_config: ChunkConfig
    ):
        """测试 Structural Chunker 保留标题层级。"""
        chunker = StructuralChunker()

        chunks = await chunker.chunk(parsed_md_doc, chunk_config)

        # 验证标题路径层级
        for chunk in chunks:
            if chunk.heading_path:
                # 标题路径应该用 " > " 分隔
                parts = chunk.heading_path.split(" > ")
                assert len(parts) >= 1


class TestSemanticChunker:
    """Semantic Chunker 测试。"""

    @pytest.mark.asyncio
    async def test_semantic_chunker_respects_token_limit(
        self, parsed_md_doc: ParsedDocument, chunk_config: ChunkConfig
    ):
        """测试 Semantic Chunker 遵守 token 限制。"""
        # 先用 Structural Chunker 生成 parent
        structural = StructuralChunker()
        parent_chunks = await structural.chunk(parsed_md_doc, chunk_config)

        semantic = SemanticChunker()

        for parent in parent_chunks:
            child_chunks = await semantic.chunk_parent(parent, chunk_config)

            # 验证每个 child 的 token 数不超过限制（允许 10% 误差）
            for child in child_chunks:
                assert child.token_count <= chunk_config.max_child_tokens * 1.1, (
                    f"Child chunk token count {child.token_count} exceeds limit {chunk_config.max_child_tokens}"
                )


class TestContextEnricher:
    """Context Enricher 测试。"""

    def test_context_enricher_adds_prefix(self):
        """测试 Context Enricher 添加前缀。"""
        enricher = ContextEnricher()

        # 创建测试 chunk
        from app.schemas.chunk import ChunkCreate

        chunk = ChunkCreate(
            document_id="doc_001",
            chunk_text="新员工入职需要提交身份证复印件和学历证明。",
            tenant_id="tenant_hr",
            department_id="dept_001",
            visibility=Visibility.DEPARTMENT,
            heading_path="员工入职制度 > 第二章 入职材料",
        )

        enriched = enricher.enrich([chunk], "员工入职制度.pdf")

        assert len(enriched) == 1
        assert enriched[0].context_prefix
        assert "员工入职制度.pdf" in enriched[0].context_prefix
        assert "第二章" in enriched[0].context_prefix
