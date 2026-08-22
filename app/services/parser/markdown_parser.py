"""
Markdown Parser。

解析 Markdown 文档，提取标题、段落、代码块、列表和表格。
"""

from __future__ import annotations

import logging
import re
from typing import Any

import anyio

from app.services.parser.base import Block, BlockType, ParsedDocument

logger = logging.getLogger(__name__)


class MarkdownParser:
    """
    Markdown 文档解析器。

    支持解析：
    - 标题（# ## ### ####）
    - 段落
    - 代码块（``` ```）
    - 列表（- * 1.）
    - 表格（| --- |）
    """

    supported_types: set[str] = {"text/markdown", "text/x-markdown"}

    async def parse(
        self, file_path: str, document_id: str, metadata: dict[str, Any]
    ) -> ParsedDocument:
        """
        解析 Markdown 文档。

        Args:
            file_path: 文件路径
            document_id: 文档 ID
            metadata: 文档元数据

        Returns:
            解析后的文档结构
        """
        logger.info("parsing_markdown", extra={"file_path": file_path, "document_id": document_id})

        content = await anyio.Path(file_path).read_text(encoding="utf-8")

        # 提取标题（第一个 # 标题作为文档标题）
        title = self._extract_title(content)

        # 解析 blocks
        blocks = self._parse_blocks(content)

        # 分离表格和图片
        tables = [b for b in blocks if b.block_type == BlockType.TABLE]
        images = [b for b in blocks if b.block_type == BlockType.IMAGE_CAPTION]

        # 计算页码（Markdown 简单按行数估算）
        pages = [1]
        total_pages = 1

        source_name = file_path.split("/")[-1].split("\\")[-1]

        return ParsedDocument(
            document_id=document_id,
            source_name=source_name,
            title=title,
            blocks=blocks,
            pages=pages,
            tables=tables,
            images=images,
            metadata=metadata,
            parser_used="markdown",
            total_pages=total_pages,
        )

    def _extract_title(self, content: str) -> str:
        """提取文档标题（第一个 # 标题）。"""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        # 如果没有 # 标题，尝试第一个非空行
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:100]  # 截取前 100 个字符
        return "Untitled"

    def _parse_blocks(self, content: str) -> list[Block]:
        """解析所有 blocks。"""
        blocks: list[Block] = []
        lines = content.split("\n")

        # 维护标题路径栈
        heading_stack: list[tuple[int, str]] = []  # (level, title)
        order_index = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # 跳过空行
            if not line.strip():
                i += 1
                continue

            # 检查标题
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                # 更新标题路径栈
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))

                heading_path = " > ".join(h[1] for h in heading_stack)

                blocks.append(
                    Block(
                        block_id=f"block_{order_index}",
                        block_type=BlockType.HEADING,
                        text=heading_text,
                        page_number=1,
                        heading_path=heading_path,
                        order_index=order_index,
                        metadata={"level": level},
                    )
                )
                order_index += 1
                i += 1
                continue

            # 检查代码块
            if line.strip().startswith("```"):
                code_lines = [line]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_lines.append(lines[i])  # 结束的 ```
                    i += 1

                code_text = "\n".join(code_lines)
                heading_path = " > ".join(h[1] for h in heading_stack)

                blocks.append(
                    Block(
                        block_id=f"block_{order_index}",
                        block_type=BlockType.CODE,
                        text=code_text,
                        page_number=1,
                        heading_path=heading_path,
                        order_index=order_index,
                    )
                )
                order_index += 1
                continue

            # 检查表格
            if (
                "|" in line
                and i + 1 < len(lines)
                and re.match(r"^\s*\|[-:\s|]+\|\s*$", lines[i + 1])
            ):
                table_lines = [line, lines[i + 1]]
                i += 2
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1

                table_text = "\n".join(table_lines)
                heading_path = " > ".join(h[1] for h in heading_stack)

                blocks.append(
                    Block(
                        block_id=f"block_{order_index}",
                        block_type=BlockType.TABLE,
                        text=table_text,
                        page_number=1,
                        heading_path=heading_path,
                        order_index=order_index,
                    )
                )
                order_index += 1
                continue

            # 检查列表
            if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
                list_lines = [line]
                i += 1
                while i < len(lines) and (
                    re.match(r"^\s*[-*+]\s+", lines[i])
                    or re.match(r"^\s*\d+\.\s+", lines[i])
                    or (lines[i].startswith("  ") and lines[i].strip())
                ):
                    list_lines.append(lines[i])
                    i += 1

                list_text = "\n".join(list_lines)
                heading_path = " > ".join(h[1] for h in heading_stack)

                blocks.append(
                    Block(
                        block_id=f"block_{order_index}",
                        block_type=BlockType.LIST,
                        text=list_text,
                        page_number=1,
                        heading_path=heading_path,
                        order_index=order_index,
                    )
                )
                order_index += 1
                continue

            # 普通段落
            paragraph_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not self._is_special_line(lines[i]):
                paragraph_lines.append(lines[i])
                i += 1

            paragraph_text = "\n".join(paragraph_lines)
            heading_path = " > ".join(h[1] for h in heading_stack)

            blocks.append(
                Block(
                    block_id=f"block_{order_index}",
                    block_type=BlockType.PARAGRAPH,
                    text=paragraph_text,
                    page_number=1,
                    heading_path=heading_path,
                    order_index=order_index,
                )
            )
            order_index += 1

        return blocks

    def _is_special_line(self, line: str) -> bool:
        """检查是否是特殊行（标题、代码块、表格、列表）。"""
        line = line.strip()
        if not line:
            return True
        if re.match(r"^#{1,6}\s+", line):
            return True
        if line.startswith("```"):
            return True
        if "|" in line:
            return True
        if re.match(r"^[-*+]\s+", line) or re.match(r"^\d+\.\s+", line):
            return True
        return False
