"""
PII 脱敏器。

脱敏敏感个人信息：手机号、身份证号、邮箱、银行卡号等。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class PIIRedactor:
    """
    PII 脱敏器。

    检测并脱敏文本中的敏感个人信息。
    """

    # PII 模式定义（注意顺序：身份证号必须在手机号前面，因为18位身份证号以1开头）
    PII_PATTERNS: list[tuple[str, str, str]] = [
        # 身份证号（18位）- 必须在手机号前面
        (r"\d{17}[\dXx]", "id_card", "***身份证号***"),
        # 中国大陆手机号
        (r"1[3-9]\d{9}", "phone", "***手机号***"),
        # 身份证号（15位）
        (r"\b\d{15}\b", "id_card", "***身份证号***"),
        # 邮箱
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "email", "***邮箱***"),
        # 银行卡号（16-19位）
        (r"\b\d{16,19}\b", "bank_card", "***银行卡号***"),
        # 固定电话
        (r"(?:0\d{2,3}[-\s]?)?\d{7,8}", "landline", "***电话***"),
    ]

    def __init__(self, enabled: bool = True) -> None:
        """
        初始化 PII 脱敏器。

        Args:
            enabled: 是否启用脱敏
        """
        self._enabled = enabled

    def redact(self, text: str) -> str:
        """
        脱敏文本中的 PII。

        Args:
            text: 输入文本

        Returns:
            脱敏后的文本
        """
        if not self._enabled:
            return text

        redacted = text
        redacted_count = 0

        for pattern, _pii_type, replacement in self.PII_PATTERNS:
            matches = re.findall(pattern, redacted)
            if matches:
                redacted = re.sub(pattern, replacement, redacted)
                redacted_count += len(matches)

        if redacted_count > 0:
            logger.info("pii_redacted", extra={"count": redacted_count, "text_length": len(text)})

        return redacted

    def detect_pii(self, text: str) -> list[dict[str, str]]:
        """
        检测文本中的 PII（不脱敏）。

        Args:
            text: 输入文本

        Returns:
            检测到的 PII 列表
        """
        detected = []

        for pattern, pii_type, _ in self.PII_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                detected.append(
                    {
                        "type": pii_type,
                        "value": match[:3] + "***",  # 只显示前3个字符
                        "position": text.find(match),
                    }
                )

        return detected

    def contains_pii(self, text: str) -> bool:
        """
        检查文本是否包含 PII。

        Args:
            text: 输入文本

        Returns:
            是否包含 PII
        """
        for pattern, _, _ in self.PII_PATTERNS:
            if re.search(pattern, text):
                return True
        return False
