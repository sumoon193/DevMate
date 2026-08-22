"""
Parser 测试。

按模块规范要求的测试：
1. test_markdown_parser_preserves_heading_path
2. test_parser_registry_routes_by_mime_type
"""

from __future__ import annotations

import os

import pytest

from app.core.exceptions import NotFoundError
from app.services.parser.base import BlockType
from app.services.parser.markdown_parser import MarkdownParser
from app.services.parser.plain_parser import PlainTextParser
from app.services.parser.registry import ParserRegistry


@pytest.fixture
def sample_md_path(fixtures_dir: str) -> str:
    """示例 Markdown 文件路径。"""
    return os.path.join(fixtures_dir, "sample.md")


@pytest.fixture
def sample_txt_path(fixtures_dir: str) -> str:
    """示例文本文件路径。"""
    return os.path.join(fixtures_dir, "sample.txt")


@pytest.fixture
def parser_registry() -> ParserRegistry:
    """Parser Registry 实例。"""
    registry = ParserRegistry()
    registry.register(MarkdownParser())
    registry.register(PlainTextParser())
    return registry


class TestMarkdownParser:
    """Markdown Parser 测试。"""

    @pytest.mark.asyncio
    async def test_markdown_parser_preserves_heading_path(self, sample_md_path: str):
        """测试 1：Markdown parser 保留标题路径。"""
        parser = MarkdownParser()

        doc = await parser.parse(
            file_path=sample_md_path, document_id="doc_001", metadata={"tenant_id": "tenant_hr"}
        )

        # 验证文档解析成功
        assert doc.document_id == "doc_001"
        assert doc.title == "员工入职与转正管理制度"
        assert doc.parser_used == "markdown"
        assert len(doc.blocks) > 0

        # 查找标题 blocks
        heading_blocks = [b for b in doc.blocks if b.block_type == BlockType.HEADING]
        assert len(heading_blocks) >= 4  # 至少有 4 个标题

        # 验证标题路径层级
        for block in heading_blocks:
            assert block.heading_path, f"Heading block {block.block_id} should have heading_path"
            assert " > " in block.heading_path or block.heading_path == block.text

        # 验证第一章的标题路径
        first_chapter = next(b for b in heading_blocks if "第一章" in b.text)
        assert "第一章" in first_chapter.heading_path

        # 验证子标题的路径包含父标题
        sub_heading = next(b for b in heading_blocks if "1.1" in b.text)
        assert "第一章" in sub_heading.heading_path
        assert "总则" in sub_heading.heading_path

    @pytest.mark.asyncio
    async def test_markdown_parser_extracts_tables(self, sample_md_path: str):
        """测试 Markdown parser 提取表格。"""
        parser = MarkdownParser()

        doc = await parser.parse(file_path=sample_md_path, document_id="doc_001", metadata={})

        # 验证表格被正确提取
        assert len(doc.tables) >= 1
        table = doc.tables[0]
        assert table.block_type == BlockType.TABLE
        assert "|" in table.text
        assert "负责人" in table.text

    @pytest.mark.asyncio
    async def test_markdown_parser_extracts_code_blocks(self, sample_md_path: str):
        """测试 Markdown parser 提取代码块。"""
        parser = MarkdownParser()

        doc = await parser.parse(file_path=sample_md_path, document_id="doc_001", metadata={})

        # 验证代码块被正确提取
        code_blocks = [b for b in doc.blocks if b.block_type == BlockType.CODE]
        assert len(code_blocks) >= 1
        assert "```" in code_blocks[0].text

    @pytest.mark.asyncio
    async def test_markdown_parser_extracts_lists(self, sample_md_path: str):
        """测试 Markdown parser 提取列表。"""
        parser = MarkdownParser()

        doc = await parser.parse(file_path=sample_md_path, document_id="doc_001", metadata={})

        # 验证列表被正确提取
        list_blocks = [b for b in doc.blocks if b.block_type == BlockType.LIST]
        assert len(list_blocks) >= 1

    @pytest.mark.asyncio
    async def test_markdown_parser_stable_output(self, sample_md_path: str):
        """测试同一文档重复解析结果稳定。"""
        parser = MarkdownParser()

        doc1 = await parser.parse(sample_md_path, "doc_001", {})
        doc2 = await parser.parse(sample_md_path, "doc_001", {})

        assert len(doc1.blocks) == len(doc2.blocks)
        for b1, b2 in zip(doc1.blocks, doc2.blocks, strict=True):
            assert b1.text == b2.text
            assert b1.heading_path == b2.heading_path


class TestPlainTextParser:
    """Plain-text Parser 测试。"""

    @pytest.mark.asyncio
    async def test_plain_parser_parses_paragraphs(self, sample_txt_path: str):
        """测试纯文本 parser 按段落切分。"""
        parser = PlainTextParser()

        doc = await parser.parse(
            file_path=sample_txt_path, document_id="doc_002", metadata={"tenant_id": "tenant_hr"}
        )

        assert doc.document_id == "doc_002"
        assert doc.parser_used == "plain_text"
        assert len(doc.blocks) > 0

        # 验证有段落和标题
        paragraph_blocks = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH]
        heading_blocks = [b for b in doc.blocks if b.block_type == BlockType.HEADING]

        assert len(paragraph_blocks) > 0
        assert len(heading_blocks) > 0

    @pytest.mark.asyncio
    async def test_plain_parser_stable_output(self, sample_txt_path: str):
        """测试同一文档重复解析结果稳定。"""
        parser = PlainTextParser()

        doc1 = await parser.parse(sample_txt_path, "doc_002", {})
        doc2 = await parser.parse(sample_txt_path, "doc_002", {})

        assert len(doc1.blocks) == len(doc2.blocks)
        for b1, b2 in zip(doc1.blocks, doc2.blocks, strict=True):
            assert b1.text == b2.text


class TestParserRegistry:
    """Parser Registry 测试。"""

    def test_parser_registry_routes_by_mime_type(self, parser_registry: ParserRegistry):
        """测试 2：Parser Registry 按 MIME 类型路由。"""
        # 验证 Markdown parser
        md_parser = parser_registry.get_parser("text/markdown")
        assert isinstance(md_parser, MarkdownParser)

        # 验证 Plain-text parser
        txt_parser = parser_registry.get_parser("text/plain")
        assert isinstance(txt_parser, PlainTextParser)

    def test_parser_registry_raises_not_found_for_unknown_type(
        self, parser_registry: ParserRegistry
    ):
        """测试 Registry 对未知 MIME 类型抛出 NotFoundError。"""
        with pytest.raises(NotFoundError):
            parser_registry.get_parser("application/pdf")

    def test_parser_registry_has_parser(self, parser_registry: ParserRegistry):
        """测试 has_parser 方法。"""
        assert parser_registry.has_parser("text/markdown") is True
        assert parser_registry.has_parser("text/plain") is True
        assert parser_registry.has_parser("application/pdf") is False

    def test_parser_registry_list_parsers(self, parser_registry: ParserRegistry):
        """测试 list_parsers 方法。"""
        parsers = parser_registry.list_parsers()
        assert "text/markdown" in parsers
        assert "text/plain" in parsers
        assert len(parsers) >= 2
