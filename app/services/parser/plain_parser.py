"""
Plain-text Parser。

解析纯文本文档，按段落切分。
"""

from __future__ import annotations

import logging
from typing import Any

import anyio

from app.services.parser.base import Block, BlockType, ParsedDocument

logger = logging.getLogger(__name__)


class PlainTextParser:
    """
    纯文本解析器。

    支持解析：
    - 按空行分隔的段落
    - 简单的标题检测（全大写或带下划线的行）
    """

    supported_types: set[str] = {"text/plain", "text/x-plain"}

    async def parse(
        self, file_path: str, document_id: str, metadata: dict[str, Any]
    ) -> ParsedDocument:
        """
        解析纯文本文档。

        Args:
            file_path: 文件路径
            document_id: 文档 ID
            metadata: 文档元数据

        Returns:
            解析后的文档结构
        """
        logger.info(
            "parsing_plain_text", extra={"file_path": file_path, "document_id": document_id}
        )

        content = await anyio.Path(file_path).read_text(encoding="utf-8")

        # 提取标题（第一个非空行）
        title = self._extract_title(content)

        # 解析 blocks
        blocks = self._parse_blocks(content)

        # 计算页码（纯文本简单按行数估算）
        pages = [1]
        total_pages = 1

        source_name = file_path.split("/")[-1].split("\\")[-1]

        return ParsedDocument(
            document_id=document_id,
            source_name=source_name,
            title=title,
            blocks=blocks,
            pages=pages,
            tables=[],
            images=[],
            metadata=metadata,
            parser_used="plain_text",
            total_pages=total_pages,
        )

    def _extract_title(self, content: str) -> str:
        """提取文档标题（第一个非空行）。"""
        for line in content.split("\n"):
            line = line.strip()
            if line:
                return line[:100]  # 截取前 100 个字符
        return "Untitled"

    def _parse_blocks(self, content: str) -> list[Block]:
        """解析所有 blocks（按段落切分）。"""
        blocks: list[Block] = []
        paragraphs = content.split("\n\n")

        order_index = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检测是否是标题（全大写或带下划线）
            if self._is_heading(para):
                blocks.append(
                    Block(
                        block_id=f"block_{order_index}",
                        block_type=BlockType.HEADING,
                        text=para,
                        page_number=1,
                        heading_path=para,
                        order_index=order_index,
                    )
                )
            else:
                blocks.append(
                    Block(
                        block_id=f"block_{order_index}",
                        block_type=BlockType.PARAGRAPH,
                        text=para,
                        page_number=1,
                        heading_path="",
                        order_index=order_index,
                    )
                )

            order_index += 1

        return blocks

    def _is_heading(self, text: str) -> bool:
        """检测是否是标题。"""
        # 全大写（至少 3 个字符）
        if text.isupper() and len(text) >= 3:
            return True
        # 带下划线的标题（如 "第一章 总则"）
        if len(text) < 50 and not text.endswith(("。", "，", "；", "：", ".", ",", ";", ":")):
            return True
        return False
