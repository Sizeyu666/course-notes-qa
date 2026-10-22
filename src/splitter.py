# -*- coding: utf-8 -*-
"""
文本切分模块。

使用递归字符切分器（RecursiveCharacterTextSplitter）把加载后的长文档切成适合检索的
短文本块（chunk）：优先在段落、换行、中文句读处断开，尽量保持语义完整。
块长 chunk_size 与重叠 chunk_overlap 统一从 Settings 读取。
"""
from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings

logger = logging.getLogger(__name__)

# 中文场景分隔符：按优先级从“段落边界”逐级回退到“单字符”
# 必须放在最后的空字符串 ""，作为兜底切分（超长硬切）
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]


def get_splitter() -> RecursiveCharacterTextSplitter:
    """根据 Settings 中的 chunk_size / chunk_overlap 构造递归字符切分器。"""
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        separators=CHINESE_SEPARATORS,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # 长度按字符数计算（中文一个字即一个字符），不使用 token 计数
        length_function=len,
        # 切分后保留每个 chunk 在原文档中的起始位置，便于溯源（不影响检索）
        add_start_index=True,
    )


def split_documents(documents: Sequence[Document]) -> list[Document]:
    """
    切分文档列表，返回 chunk 列表。

    - 每个 chunk 完整保留来源 Document 的 metadata（source / file_type / page 等）；
    - 为每个 chunk 追加全局递增的 metadata["chunk_index"]，从 0 开始、跨文件连续编号，
      用于排查切分顺序与重建时去重定位。
    """
    if not documents:
        logger.warning("传入切分的文档列表为空，返回空 chunk 列表。")
        return []

    splitter = get_splitter()
    chunks = splitter.split_documents(list(documents))

    for index, chunk in enumerate(chunks):
        # split_documents 已复制原 metadata，这里只做增量标注
        chunk.metadata["chunk_index"] = index

    logger.info(
        "切分完成：输入 %d 个文档，输出 %d 个 chunk（chunk_size=%d, overlap=%d）。",
        len(documents),
        len(chunks),
        get_settings().chunk_size,
        get_settings().chunk_overlap,
    )
    return chunks
