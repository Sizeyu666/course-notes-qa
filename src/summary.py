# -*- coding: utf-8 -*-
"""
划重点与复习提纲模块。

- _build_full_text()：把切分后的 chunk 按“来源文件 + chunk 顺序”拼接成供大模型阅读的
  完整课件文本；超长时在全部 chunk 范围内均匀节选（而非简单砍掉后半部分），并标注节选说明；
- generate_key_points()：基于课件生成 10~20 条核心知识点（一行一条，以「·」开头）；
- generate_outline()：基于课件生成“章节名 → 核心概念 → 要点”的分层 Markdown 复习提纲。

两个生成函数都只允许使用课件内信息；课件无有效内容时返回明确的中文说明，不调用/不编造。
LLM 客户端通过 src.config.build_chat_llm 统一初始化，与 RAG 问答链共用同一套 DeepSeek 配置。
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import build_chat_llm

logger = logging.getLogger(__name__)

# 超长节选时在文本开头标注的说明
TRUNCATION_NOTICE = "（内容较长，已节选）"

# 课件中没有任何可用内容时两个函数统一返回的说明
NO_CONTENT_TEXT = "根据现有课件资料，未找到可提炼的内容"

# ---------- 提示词 ----------
KEY_POINTS_SYSTEM_PROMPT = """你是一名课程复习助手，请只依据用户提供的「课件内容」提炼核心知识点。必须遵守：

1. 只能使用课件中出现的信息，严禁补充课件之外的知识，严禁推测。
2. 输出 10~20 条核心知识点（最多不得超过 20 条），尽量覆盖课件的各个章节；如果课件内容较少，按实际数量输出即可，不要凑数。
3. 每条知识点独占一行，以「· 」开头，用一句简洁准确的话概括一个完整知识点。
4. 只输出知识点列表本身，不要输出标题、开场白、编号说明或总结性语句。
5. 如果课件没有任何有效内容，只能输出这句话：“根据现有课件资料，未找到可提炼的内容”。"""

OUTLINE_SYSTEM_PROMPT = """你是一名课程复习助手，请只依据用户提供的「课件内容」整理 Markdown 格式的复习提纲。必须遵守：

1. 只能使用课件中出现的信息，严禁补充课件之外的知识，严禁推测。
2. 提纲按“章节名 → 核心概念 → 要点”三级组织：
   - 用 Markdown 二级标题「## 章节名」表示章节/主题；
   - 用三级标题「### 核心概念」表示该章节下的概念或主题；
   - 概念下面用无序列表「- 要点」列出具体要点。
3. 层级要清晰，要点简洁准确，覆盖课件的主要内容。
4. 只输出 Markdown 提纲本体，不要输出代码块包裹、开场白或结束语。
5. 如果课件没有任何有效内容，只能输出这句话：“根据现有课件资料，未找到可提炼的内容”。"""

# 行首可能出现的项目符号或数字编号，用于把模型输出归一化为「· 」开头
_LEADING_BULLET_PATTERN = re.compile(r"^\s*(?:[·•●*\-–]|\d+[.、)）])\s*")


def _chunk_sort_key(doc: Document) -> tuple[str, int]:
    """chunk 排序键：先按来源文件名，再按 chunk_index；元数据缺失时给出安全兜底。"""
    metadata = getattr(doc, "metadata", {}) or {}
    source = str(metadata.get("source", ""))
    chunk_index = metadata.get("chunk_index")
    return source, chunk_index if isinstance(chunk_index, int) else 0


def _group_by_source(docs: list[Document]) -> "dict[str, list[Document]]":
    """按 source 分组，组内保持 chunk_index 升序；分组顺序遵循各文件首次出现的顺序。"""
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        source = str((doc.metadata or {}).get("source", "未知来源"))
        grouped[source].append(doc)
    return grouped


def _render_sections(selected: list[Document]) -> str:
    """把选出的 chunk 按来源分节渲染为文本，每节带【文件名】标题。"""
    grouped = _group_by_source(selected)
    sections: list[str] = []
    for source, chunks in grouped.items():
        body = "\n".join((c.page_content or "").strip() for c in chunks)
        sections.append(f"【{source}】\n{body}")
    return "\n\n".join(sections)


def _build_full_text(docs: list[Document], max_chars: int = 12000) -> str:
    """
    把切分后的文档拼接为完整课件文本。

    - 按 source 分节、节内按 chunk_index 排序，保证同一文件的内容连贯；
    - 总长度不超过 max_chars 时返回全文；
    - 超长时在全部 chunk 范围内等间距抽样（开头、中间、结尾都能覆盖到），
      并在开头注明“（内容较长，已节选）”，最后一个入选 chunk 按剩余预算截断。

    :param docs: 切分后的 Document（chunk）列表
    :param max_chars: 允许的最大字符数（含节选说明），默认 12000
    :return: 拼接（或均匀节选）后的课件文本；无有效内容时返回空字符串
    """
    # 去空白、过滤空文本后按 source / chunk_index 排序
    chunks = [doc for doc in docs if (getattr(doc, "page_content", "") or "").strip()]
    chunks.sort(key=_chunk_sort_key)
    if not chunks:
        return ""

    full_text = _render_sections(chunks)
    if len(full_text) <= max_chars:
        return full_text

    # 预留节选说明的字符额度，其余额度按 chunk 均匀抽样分配
    budget = max_chars - len(TRUNCATION_NOTICE)
    if budget <= 0:
        return TRUNCATION_NOTICE

    total_len = sum(len(c.page_content.strip()) for c in chunks)
    avg_len = total_len / len(chunks)
    # 估算可容纳的 chunk 数量，再在 [0, n-1] 上取等间距下标，保证覆盖开头到结尾
    target_count = min(len(chunks), max(1, int(budget // max(avg_len, 1))))
    if target_count == 1:
        picked_indices = [0]
    else:
        picked_indices = sorted(
            {round(i * (len(chunks) - 1) / (target_count - 1)) for i in range(target_count)}
        )

    selected: list[Document] = []
    used = 0
    for idx in picked_indices:
        text = chunks[idx].page_content.strip()
        remaining = budget - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            # 最后一个入选 chunk 按剩余预算截断，保证总长不超限
            text = text[:remaining]
        # 截断后需要替换原文档内容用于渲染，这里复制一份避免修改入参对象
        clipped = Document(page_content=text, metadata=dict(chunks[idx].metadata or {}))
        selected.append(clipped)
        used += len(text)

    return TRUNCATION_NOTICE + _render_sections(selected)


def _invoke_llm(llm, system_prompt: str, course_text: str) -> str:
    """用 system + human 两条消息调用大模型，失败时抛出带中文说明的错误。"""
    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"课件内容如下：\n\n{course_text}"),
            ]
        )
    except Exception as exc:  # noqa: BLE001 外部调用统一转中文错误
        raise RuntimeError(f"调用大模型生成内容失败，请检查网络与 DeepSeek 服务。原始错误：{exc}") from exc
    return (getattr(response, "content", "") or "").strip()


def _normalize_key_points(raw_text: str, max_points: int = 20) -> str:
    """
    把模型输出归一化为“每行一条、以「· 」开头”的知识点列表。

    模型偶尔会超出条数上限，这里在输出侧做确定性兜底：超过 max_points 时只保留前 20 条，
    优先保证覆盖课件开头主干章节，格式始终合规。
    """
    if NO_CONTENT_TEXT in raw_text:
        return NO_CONTENT_TEXT

    lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # 去掉模型可能自带的符号/数字编号，统一重新加「· 」
        line = _LEADING_BULLET_PATTERN.sub("", line).strip()
        if line:
            lines.append(f"· {line}")

    if not lines:
        return NO_CONTENT_TEXT
    if len(lines) > max_points:
        logger.info("模型输出 %d 条知识点，超出上限 %d，已截取前 %d 条。", len(lines), max_points, max_points)
        lines = lines[:max_points]
    return "\n".join(lines)


def generate_key_points(docs: list[Document], llm=None) -> str:
    """
    基于课件生成核心知识点列表。

    :param docs: 切分后的 chunk 列表
    :param llm: 大模型客户端；为 None 时用 build_chat_llm() 按统一配置初始化
    :return: 每行一条、以「· 」开头的知识点文本；无内容时返回固定中文说明
    """
    course_text = _build_full_text(docs)
    if not course_text:
        logger.info("课件内容为空，划重点直接返回无内容提示，不调用大模型。")
        return NO_CONTENT_TEXT

    if llm is None:
        llm = build_chat_llm()
    raw_text = _invoke_llm(llm, KEY_POINTS_SYSTEM_PROMPT, course_text)
    return _normalize_key_points(raw_text)


def generate_outline(docs: list[Document], llm=None) -> str:
    """
    基于课件生成分层 Markdown 复习提纲（章节名 → 核心概念 → 要点）。

    :param docs: 切分后的 chunk 列表
    :param llm: 大模型客户端；为 None 时用 build_chat_llm() 按统一配置初始化
    :return: Markdown 提纲文本；无内容时返回固定中文说明
    """
    course_text = _build_full_text(docs)
    if not course_text:
        logger.info("课件内容为空，复习提纲直接返回无内容提示，不调用大模型。")
        return NO_CONTENT_TEXT

    if llm is None:
        llm = build_chat_llm()
    raw_text = _invoke_llm(llm, OUTLINE_SYSTEM_PROMPT, course_text)
    outline = raw_text.strip()
    # 剥掉模型可能误加的 Markdown 代码块包裹
    outline = re.sub(r"^```[a-zA-Z]*\n?|```$", "", outline).strip()
    if NO_CONTENT_TEXT in outline:
        return NO_CONTENT_TEXT
    return outline or NO_CONTENT_TEXT
