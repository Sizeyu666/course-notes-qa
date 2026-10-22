# -*- coding: utf-8 -*-
"""
RAG 问答链模块（LangChain 1.x / langchain-classic，LCEL 组合链）。

链路：
    用户问题 + 聊天历史
        → 历史感知检索器：把含指代词的追问改写成独立问题，再做向量检索
        → 回答链：把检索到的课件片段填入提示词，由 DeepSeek 在资料范围内作答
        → create_retrieval_chain 串联两者

输出在标准链路基础上增加两点保障：
1. 资料中无依据时必须回答固定拒答语，严禁编造（由 system prompt 强约束）；
2. 回答末尾的「参考来源：」由代码依据实际召回的 chunk 元数据确定性生成，
   格式统一为「文件名#chunk序号」并去重，避免模型漏标或乱标。

禁用 RetrievalQA / ConversationalRetrievalChain 等老接口。
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangChain 1.x 中经典 RAG 链由兼容层 langchain_classic 提供
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# LLM 初始化统一复用 config 中的公共入口，不在本模块重复编写 DeepSeek 配置
from src.config import build_chat_llm
from src.vector_store import get_retriever

logger = logging.getLogger(__name__)

# 资料中没有依据时的固定拒答语
NO_EVIDENCE_ANSWER = "根据现有课件资料，未找到相关内容"

# ---------- 提示词 1：结合聊天历史的问题改写 ----------
CONTEXTUALIZE_SYSTEM_PROMPT = (
    "你是一个问题改写助手。给定聊天历史和用户最新提出的问题，"
    "该问题可能包含“它”“这个”“上面提到的”等指代词。"
    "请结合聊天历史，把最新问题改写成一个不依赖上下文也能独立理解的完整中文问题。"
    "要求：只输出改写后的问题本身，不要回答问题，不要添加任何解释或客套话；"
    "如果最新问题本身已经完整、无需改写，则原样返回该问题。"
)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ---------- 提示词 2：基于课件资料的回答 ----------
QA_SYSTEM_PROMPT = """你是一名课程学习助手，请严格依据「检索到的课件资料」回答学生的问题。你必须遵守以下规则：

1. 只能使用下面提供的课件资料中的信息作答，严禁使用资料之外的任何知识，也不要推测。
2. 如果资料中没有与问题相关的依据，必须且只能回答：“根据现有课件资料，未找到相关内容”，严禁编造、严禁提及资料以外的内容。
3. 回答要条理清晰，内容较多时适当分点（使用 1. 2. 3. 或项目符号），语言简洁准确。
4. 不要在回答正文的开头或结尾输出“参考来源/引用出处”等来源清单，参考来源由问答系统在回答之外自动展示；你只需专注回答内容本身。

检索到的课件资料如下：
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 匹配模型自行生成的结尾“参考来源：……”段落（含其前的空行/横线），用于去重前的切除
_REFERENCE_TAIL_PATTERN = re.compile(
    r"\n*[-—]{2,}\s*\n*参考来源[：:].*$|[\n\r]+\s*参考来源[：:].*$",
    re.DOTALL,
)


def get_rag_chain(vector_store, top_k: int | None = None):
    """
    组装完整 RAG 问答链并返回。

    :param vector_store: 已加载/构建好的 Chroma 向量库实例
    :param top_k: 检索召回块数；None 时取 Settings.top_k。界面调整滑块后
                  只需用新 top_k 重新组装链（检索器随之更新），无需重建向量库。
    """
    llm = build_chat_llm(temperature=0.2)
    retriever = get_retriever(vector_store, k=top_k)

    # 第 1 段：带历史的问题改写 → 用改写后的问题检索
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=CONTEXTUALIZE_PROMPT,
    )

    # 第 2 段：检索片段 + 提示词 → 生成回答
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=QA_PROMPT)

    # 第 3 段：串成“检索 → 生成”完整链（输入 input / chat_history，输出 answer / context）
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def _convert_chat_history(chat_history: list[dict[str, str]]) -> list[BaseMessage]:
    """
    把 dict 形式的聊天历史转换为 LangChain 消息对象列表。

    入参格式：[{"role": "user" | "assistant", "content": "..."}, ...]
    - user      → HumanMessage
    - assistant → AIMessage
    其他角色或缺少 content 的条目会被跳过并记录警告。
    """
    messages: list[BaseMessage] = []
    for index, item in enumerate(chat_history or []):
        if not isinstance(item, dict):
            logger.warning("聊天历史第 %d 项不是字典，已忽略。", index)
            continue
        role = item.get("role")
        content = item.get("content", "")
        if not isinstance(content, str):
            logger.warning("聊天历史第 %d 项的 content 不是字符串，已忽略。", index)
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            logger.warning("聊天历史第 %d 项的角色 %r 不受支持，已忽略。", index, role)
    return messages


def _build_reference_line(source_documents: list[Any]) -> str:
    """
    依据召回文档的 metadata 确定性生成参考来源文本，去重并保持召回顺序。

    格式：参考来源：文件名#chunk序号、文件名#chunk序号
    """
    references: list[str] = []
    seen: set[str] = set()
    for doc in source_documents:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "未知文件")
        chunk_index = metadata.get("chunk_index")
        reference = f"{source}#{chunk_index}" if chunk_index is not None else str(source)
        if reference not in seen:
            seen.add(reference)
            references.append(reference)
    return "、".join(references)


def answer(chain, question: str, chat_history: list[dict[str, str]] | None = None) -> dict:
    """
    执行一轮问答。

    :param chain: get_rag_chain() 返回的 RAG 链
    :param question: 用户本轮问题
    :param chat_history: 历史消息，格式 [{"role": "user"/"assistant", "content": ...}]
    :return: {"answer": 回答文本（末尾含参考来源）, "source_documents": 召回的 Document 列表}
    """
    if not question or not question.strip():
        raise ValueError("问题不能为空，请输入有效的问题。")

    messages = _convert_chat_history(chat_history)
    try:
        result = chain.invoke({"input": question, "chat_history": messages})
    except Exception as exc:  # noqa: BLE001 外部调用统一转中文错误
        raise RuntimeError(f"调用 RAG 问答链失败，请检查网络与 DeepSeek 服务是否可用。原始错误：{exc}") from exc

    answer_text: str = result.get("answer", "") or ""
    # 经典检索链把召回文档放在 context 字段，对外统一改名为 source_documents
    source_documents = result.get("context", []) or []

    # 先切除模型自行生成的“参考来源”尾段，统一改由代码确定性生成，保证格式与去重一致
    cleaned_answer = _REFERENCE_TAIL_PATTERN.sub("", answer_text).rstrip()

    # 拒答场景：不附加来源；正常回答：追加确定性的参考来源行
    if NO_EVIDENCE_ANSWER in cleaned_answer:
        final_answer = NO_EVIDENCE_ANSWER
    else:
        reference_line = _build_reference_line(source_documents)
        final_answer = (
            f"{cleaned_answer}\n参考来源：{reference_line}" if reference_line else cleaned_answer
        )

    return {"answer": final_answer, "source_documents": source_documents}


def stream_answer(chain, question: str, chat_history: list[dict[str, str]] | None = None):
    """
    流式执行一轮问答：逐 token yield 回答增量文本，供界面用 st.write_stream 实时渲染。

    检索到的原文片段在流开始时一次性给出，挂在返回生成器的 ``state`` 属性上；
    流结束后还可从 ``state`` 读取拼接好的完整回答与召回文档，界面据此单独渲染
    「参考来源」标签与原文片段折叠区。

    :return: 生成器对象 g；迭代 g 得到回答增量，迭代结束后：
             g.state["answer"] 为完整回答（已剥离来源尾段、拒答时归一化），
             g.state["source_documents"] 为召回 Document 列表。
    """
    if not question or not question.strip():
        raise ValueError("问题不能为空，请输入有效的问题。")

    messages = _convert_chat_history(chat_history)
    stream = _StreamingAnswer(chain, question, messages)
    # 返回可直接交给 st.write_stream 迭代的对象，完整回答与召回文档在其 state 上读取
    return stream


class _StreamingAnswer:
    """
    流式问答的可迭代包装：逐 token 产出回答增量，同时把召回文档与完整回答暴露在
    ``self.state`` 上。原生生成器对象不支持自定义属性，故用类做“带外返回通道”。
    """

    def __init__(self, chain, question: str, messages: list[BaseMessage]):
        self._chain = chain
        self._question = question
        self._messages = messages
        self.state: dict = {"answer": "", "source_documents": []}

    def __iter__(self):
        collected: list[str] = []
        try:
            for chunk in self._chain.stream({"input": self._question, "chat_history": self._messages}):
                # 检索结果在回答开始前以 context 键一次性到达
                context = chunk.get("context")
                if context:
                    self.state["source_documents"] = context
                token = chunk.get("answer")
                if token:
                    collected.append(token)
                    yield token
        except Exception as exc:  # noqa: BLE001 外部调用统一转中文错误
            raise RuntimeError(
                f"流式调用问答链失败，请检查网络与 DeepSeek 服务是否可用。原始错误：{exc}"
            ) from exc

        # 流结束：剥离可能的来源尾段；拒答时归一化为固定拒答语（不带来源）
        full_text = _REFERENCE_TAIL_PATTERN.sub("", "".join(collected)).rstrip()
        self.state["answer"] = (
            NO_EVIDENCE_ANSWER if NO_EVIDENCE_ANSWER in full_text else full_text
        )
