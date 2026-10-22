# -*- coding: utf-8 -*-
"""
课程笔记智能问答助手 —— Streamlit 单页应用入口。

功能：上传 PDF / Word / PPT 课件 → 构建本地向量知识库 → 多轮问答（带历史改写与参考来源）
     → 一键划重点 / 生成复习提纲。

UI 规范：严格遵循 .trae/skills/streamlit-clean-ui/SKILL.md（清爽学术风），
全局样式只在 inject_css() 一处注入。
"""

# ============================================================
# SQLite 兼容处理（必须放在所有第三方库 import 之前）
# ------------------------------------------------------------
# 背景：Chroma 要求较新版本的 SQLite，而 Streamlit Cloud 等 Linux 部署环境
# 自带的系统 SQLite 可能过旧。requirements.txt 中 pysqlite3-binary 通过
# ``sys_platform == "linux"`` 标记仅在 Linux 安装；这里在任何库间接
# import sqlite3 之前，优先用 pysqlite3 顶替标准库模块。
# Windows / macOS 不安装该包，ImportError 时静默跳过即可。
try:
    import pysqlite3 as _pysqlite3
    import sys as _sys

    _sys.modules["sqlite3"] = _pysqlite3
except ImportError:
    pass
# ============================================================

import html
import logging
import os
import re
from contextlib import contextmanager

import streamlit as st

from src.config import get_settings
from src.document_loader import load_uploaded_files
from src.embeddings import get_embeddings
from src.rag_chain import NO_EVIDENCE_ANSWER, get_rag_chain, stream_answer
from src.splitter import split_documents
from src.summary import generate_key_points, generate_outline
from src.vector_store import build_vector_store, load_vector_store

# 业务模块默认日志级别，避免第三方库的调试信息干扰界面
logging.basicConfig(level=logging.WARNING)


# ============================================================
# 全局样式（设计令牌见 streamlit-clean-ui skill，仅在此处注入一次）
# ============================================================
def inject_css() -> None:
    """注入清爽学术风全局 CSS；任何样式问题都不应导致页面中断。"""
    st.markdown(
        """
        <style>
        /* ===== 全局字体与背景 ===== */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                         "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
        }
        .stApp { background-color: #F6F8FB; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1180px; }

        /* ===== 隐藏冗余 chrome，保持干净 ===== */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { display: none; }

        /* ===== 侧边栏 ===== */
        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E5E9F0;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
        section[data-testid="stSidebar"] h1 {
            font-size: 18px; color: #1E293B; margin-bottom: 2px;
        }
        .side-label {
            font-size: 12px; font-weight: 600; letter-spacing: .04em;
            color: #64748B; text-transform: uppercase; margin: 14px 0 6px;
        }

        /* ===== 主按钮 ===== */
        .stButton > button, button[data-testid="stBaseButton-primary"] {
            border-radius: 8px; font-weight: 600; padding: 8px 16px;
            transition: all .15s ease;
        }
        button[data-testid="stBaseButton-primary"] {
            background: #3B82F6; border-color: #3B82F6;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background: #2563EB; border-color: #2563EB;
        }
        .stButton > button[kind="secondary"] {
            background: #FFFFFF; border: 1px solid #E5E9F0; color: #334155;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: #3B82F6; color: #2563EB; background: #F8FAFF;
        }

        /* ===== 文件上传 ===== */
        [data-testid="stFileUploader"] section {
            border: 1.5px dashed #CBD5E1; border-radius: 12px;
            background: #F8FAFC; padding: 8px;
        }
        [data-testid="stFileUploader"] section:hover { border-color: #3B82F6; }

        /* ===== 卡片 ===== */
        .ui-card {
            background: #FFFFFF; border: 1px solid #E5E9F0;
            border-radius: 12px; padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(15,23,42,.06);
            margin-bottom: 12px;
        }
        div[data-testid="stExpander"] {
            background: #FFFFFF; border: 1px solid #E5E9F0;
            border-radius: 12px; overflow: hidden;
        }

        /* ===== Tabs ===== */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0; padding: 8px 18px; font-weight: 500;
            color: #64748B;
        }
        .stTabs [aria-selected="true"] {
            color: #2563EB; background: #FFFFFF;
        }
        .stTabs [data-baseweb="tab-highlight"] { background-color: #3B82F6; }

        /* ===== 聊天消息 ===== */
        [data-testid="stChatMessage"] {
            background: transparent; border: none; padding: 10px 0;
        }
        [data-testid="stChatMessageContent"] {
            background: #FFFFFF; border: 1px solid #E5E9F0;
            border-radius: 12px; padding: 12px 16px; line-height: 1.7;
            box-shadow: 0 1px 2px rgba(15,23,42,.04);
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
            [data-testid="stChatMessageContent"] {
            background: #3B82F6; color: #FFFFFF; border-color: #3B82F6;
        }
        [data-testid="stChatMessageAvatarAssistant"] {
            background: #8B5CF6 !important; color: #fff !important;
        }
        .src-tag {
            display: inline-block; font-size: 12px; color: #2563EB;
            background: #EFF6FF; border: 1px solid #DBEAFE; border-radius: 999px;
            padding: 2px 10px; margin: 6px 6px 0 0;
        }
        .src-title { font-size: 12px; color: #64748B; margin-top: 10px; }

        /* ===== 聊天输入框 ===== */
        [data-testid="stChatInput"] {
            border: 1px solid #E5E9F0; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(15,23,42,.05); background: #FFFFFF;
        }

        /* ===== 提示条 ===== */
        [data-testid="stAlert"] { border-radius: 10px; }

        /* ===== 状态徽标 ===== */
        .kb-badge {
            display:inline-block; font-size:12px; font-weight:600;
            border-radius:999px; padding:3px 10px;
        }
        .kb-ok { color:#15803D; background:#F0FDF4; border:1px solid #BBF7D0; }
        .kb-no { color:#B45309; background:#FFFBEB; border:1px solid #FDE68A; }

        /* ===== 空状态 ===== */
        .empty-state {
            border: 1px dashed #CBD5E1; border-radius: 12px;
            background: #F8FAFC; color: #64748B; text-align: center;
            padding: 34px 20px; font-size: 14px; line-height: 1.9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# session_state 初始化
# ============================================================
def _persisted_store_exists() -> bool:
    """判断磁盘上是否已有持久化的 Chroma 库，避免无库时白白加载 Embedding 模型（耗时数秒）。"""
    persist_dir = os.path.abspath(get_settings().vector_store_dir)
    return os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))


def init_state() -> None:
    """
    初始化会话状态，所有键给默认值，避免跨 rerun 出现 KeyError。

    首次启动时若磁盘上已有持久化的向量库（vector_store/chroma.sqlite3 存在），
    自动加载并恢复 session_state，用户重启应用后无需重新上传构建。
    """
    defaults = {
        "rag_chain": None,        # 构建好的 RAG 问答链
        "vector_store": None,     # Chroma 向量库实例
        "chat_history": [],       # [{"role": user/assistant, "content": ..., "sources": ...}]
        "key_points": "",         # 划重点结果（Markdown 文本）
        "outline": "",            # 复习提纲结果（Markdown 文本）
        "kb_ready": False,        # 知识库是否就绪
        "chunk_count": 0,         # 当前知识库的文本块数量
        "kb_notice": "",          # 一次性成功提示（rerun 后在主区展示，展示后清除）
        "top_k": get_settings().top_k,  # 检索召回数量（滑块可调，仅重建检索器）
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 首次运行时尝试自动恢复磁盘上已持久化的知识库
    if not st.session_state.kb_ready and not st.session_state.vector_store:
        # 没有库文件时直接跳过：不加载 Embedding 模型，避免首屏额外等待
        if not _persisted_store_exists():
            return
        try:
            embeddings = get_embeddings()
            vs = load_vector_store(embeddings)
            chunk_count = vs._collection.count() if vs is not None else 0
            if chunk_count == 0:
                # 空集合（例如上次构建中途失败）不视为可用的知识库
                return
            st.session_state.vector_store = vs
            st.session_state.rag_chain = get_rag_chain(vs, top_k=st.session_state.top_k)
            st.session_state.chunk_count = chunk_count
            st.session_state.kb_ready = True
            st.session_state.kb_notice = (
                f"检测到已有的本地知识库，已自动恢复。当前共 {chunk_count} 个文本块。"
                f"若要更新，可重新上传课件并点击「构建知识库」。"
            )
        except Exception:  # noqa: BLE001 自动恢复失败不阻塞应用，保持未就绪状态即可
            pass


# ============================================================
# 知识库构建（加载 → 切分 → embedding → 建库 → 构建问答链）
# ============================================================
def _dedupe_uploads(uploaded_files) -> tuple[list, list[str]]:
    """
    按“文件名 + 文件大小”对上传文件去重，返回 (去重后的文件列表, 被去重的文件名列表)。

    同名且同大小视为重复上传（多文件选择或多次添加时可能出现），每份只构建一次。
    """
    unique: list = []
    seen: set[tuple[str, int]] = set()
    duplicated_names: list[str] = []
    for up_file in uploaded_files:
        signature = (up_file.name, up_file.size)
        if signature in seen:
            duplicated_names.append(up_file.name)
        else:
            seen.add(signature)
            unique.append(up_file)
    return unique, duplicated_names


@contextmanager
def _capture_loader_warnings():
    """
    临时挂接日志处理器，收集文档加载阶段的中文警告。

    文档加载模块对“不支持格式 / 空文件 / 损坏文件 / 扫描件”等只记录日志，
    这里把日志转成界面提示，避免用户上传多个文件时部分文件被静默跳过。
    """
    collected: list[str] = []

    class _CollectHandler(logging.Handler):
        def emit(self, record):  # noqa: D401 仅收集消息文本
            collected.append(record.getMessage())

    handler = _CollectHandler(level=logging.WARNING)
    loader_logger = logging.getLogger("src.document_loader")
    loader_logger.addHandler(handler)
    try:
        yield collected
    finally:
        loader_logger.removeHandler(handler)


def build_knowledge_base(uploaded_files) -> None:
    """点击「构建知识库」后的完整流水线，分阶段输出状态，异常统一转为中文错误。"""
    if not uploaded_files:
        st.error("尚未选择任何文件。请先上传 PDF、Word 或 PPT 课件，再点击「构建知识库」。")
        return

    # 同名/重复文件去重，并明确提示用户哪些文件被合并
    unique_files, duplicated_names = _dedupe_uploads(uploaded_files)
    if duplicated_names:
        st.warning(
            "检测到重复上传的文件："
            + "、".join(sorted(set(duplicated_names)))
            + "。同名文件已自动去重，每份课件只构建一次。"
        )

    try:
        with st.status("正在构建知识库…首次加载本地向量模型可能需要几分钟", expanded=True) as status:
            st.write("① 正在加载文件：解析 PDF / Word / PPT 课件…")
            with _capture_loader_warnings() as skip_notes:
                documents = load_uploaded_files(unique_files)
            # 逐条展示被跳过的文件及原因（去重，避免同一原因重复刷屏）
            for note in dict.fromkeys(skip_notes):
                st.warning(note)
            if not documents:
                status.update(label="课件解析失败", state="error", expanded=True)
                st.error(
                    "未能从上传的文件中解析出任何文本。请确认文件不是空文件、加密 PDF 或"
                    "纯图片扫描件（扫描件需先做 OCR），然后重试。"
                )
                return

            st.write("② 正在进行文本切分…")
            chunks = split_documents(documents)

            st.write("③ 正在向量化：加载 Embedding 模型并写入向量库…")
            embeddings = get_embeddings()
            vector_store = build_vector_store(chunks, embeddings)

            st.write("④ 正在构建检索链（按当前 top_k 组装）…")
            rag_chain = get_rag_chain(vector_store, top_k=st.session_state.top_k)

            status.update(label="知识库构建完成", state="complete", expanded=False)

        # 重建知识库后，旧对话与旧的重点/提纲全部失效，需要重置
        st.session_state.vector_store = vector_store
        st.session_state.rag_chain = rag_chain
        st.session_state.chunk_count = len(chunks)
        st.session_state.kb_ready = True
        st.session_state.chat_history = []
        st.session_state.key_points = ""
        st.session_state.outline = ""
        # 成功提示写入一次性 notice，rerun 后由主区稳定展示，避免一闪而过
        st.session_state.kb_notice = f"知识库构建完成，共 {len(chunks)} 个文本块。现在可以开始提问。"
        st.rerun()
    except Exception as exc:  # noqa: BLE001 任何异常都兜住，禁止白屏/英文堆栈
        st.error(f"知识库构建失败：{exc}。请检查文件内容、网络或 .env 中的模型配置后重试。")


# ============================================================
# 划重点 / 复习提纲
# ============================================================
def make_key_points() -> None:
    """侧边栏「一键划重点」：结果存入 session_state，供重点 tab 渲染。"""
    try:
        with st.spinner("正在通读课件并提炼核心知识点…"):
            # 直接从已构建的向量库取出全部入库 chunk，保证重点覆盖整份课件
            raw_chunks = st.session_state.vector_store._collection.get(
                include=["documents", "metadatas"]
            )
            docs = _records_to_documents(raw_chunks)
            st.session_state.key_points = generate_key_points(docs)
        st.success("划重点完成，已更新到「重点」标签页。")
    except Exception as exc:  # noqa: BLE001
        st.error(f"划重点失败：{exc}。请稍后重试。")


def make_outline() -> None:
    """侧边栏「生成复习提纲」：结果存入 session_state，供提纲 tab 渲染。"""
    try:
        with st.spinner("正在按章节整理复习提纲…"):
            raw_chunks = st.session_state.vector_store._collection.get(
                include=["documents", "metadatas"]
            )
            docs = _records_to_documents(raw_chunks)
            st.session_state.outline = generate_outline(docs)
        st.success("复习提纲已生成，已更新到「复习提纲」标签页。")
    except Exception as exc:  # noqa: BLE001
        st.error(f"生成复习提纲失败：{exc}。请稍后重试。")


def _records_to_documents(records: dict) -> list:
    """把 Chroma collection.get() 返回的记录重新组装成 LangChain Document。"""
    from langchain_core.documents import Document

    docs: list[Document] = []
    contents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    for content, metadata in zip(contents, metadatas):
        docs.append(Document(page_content=content, metadata=metadata or {}))
    # 按 chunk_index 排序，保证送给大模型的内容顺序与课件一致
    docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    return docs


# ============================================================
# 侧边栏
# ============================================================
def render_sidebar() -> None:
    """渲染侧边栏：品牌、上传、构建、状态徽标、重点/提纲、清空、模型信息。"""
    settings = get_settings()
    with st.sidebar:
        st.markdown("# 课程笔记问答")
        st.caption("上传课件，构建你的专属课程知识库")

        st.markdown('<div class="side-label">知识库</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "上传课件（支持多文件）",
            type=["pdf", "docx", "pptx"],
            accept_multiple_files=True,
            help="支持 PDF、Word（.docx）、PPT（.pptx），单文件不超过 50MB",
        )

        if st.button("构建知识库", type="primary", use_container_width=True):
            build_knowledge_base(uploaded_files)

        # 知识库状态徽标
        if st.session_state.kb_ready:
            st.markdown(
                f'<span class="kb-badge kb-ok">● 知识库就绪 · {st.session_state.chunk_count} 个文本块</span>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="side-label">检索设置</div>', unsafe_allow_html=True)
            new_top_k = st.slider(
                "检索片段数量 top_k",
                min_value=2,
                max_value=8,
                value=st.session_state.top_k,
                step=1,
                help="每次问答从知识库召回的片段数量。调大覆盖更全但可能引入无关内容，调小更聚焦。",
            )
            st.caption(
                "说明：top_k 控制每次召回的片段数量（按相似度从高到低返回）；"
                "相似度阈值用于过滤低相关片段，本应用当前采用“固定返回最相关的 top_k 个片段”策略。"
                "调整后仅重建检索器，无需重新构建向量库。"
            )
            # 滑块变化时用新的 k 重新组装链（检索器随之更新），控件交互本身会触发一次 rerun，
            # 这里不额外调用 st.rerun，避免布局跳变。
            if new_top_k != st.session_state.top_k:
                try:
                    with st.spinner(f"正在按 top_k={new_top_k} 重建检索器…"):
                        st.session_state.rag_chain = get_rag_chain(
                            st.session_state.vector_store, top_k=new_top_k
                        )
                    st.session_state.top_k = new_top_k
                    st.success(f"检索器已更新，当前每次召回 {new_top_k} 个片段。")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"检索器更新失败：{exc}")

            st.markdown('<div class="side-label">学习工具</div>', unsafe_allow_html=True)
            if st.button("一键划重点", use_container_width=True):
                make_key_points()
            if st.button("生成复习提纲", use_container_width=True):
                make_outline()
        else:
            st.markdown(
                '<span class="kb-badge kb-no">○ 知识库未就绪</span>',
                unsafe_allow_html=True,
            )

        if st.session_state.chat_history:
            st.markdown('<div class="side-label">会话</div>', unsafe_allow_html=True)
            if st.button("清空对话", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.kb_notice = "对话已清空。"
                st.rerun()

        with st.expander("使用说明"):
            st.markdown(
                "1. 在上方上传 PDF / Word / PPT 课件（可多选）；\n\n"
                "2. 点击「构建知识库」，等待解析、切分与向量化完成；\n\n"
                "3. 在「问答」页连续提问，支持“它 / 这个”等指代追问；\n\n"
                "4. 可随时一键划重点、生成复习提纲；\n\n"
                "5. 回答仅依据课件内容，资料中没有时会明确说明。"
            )

        st.divider()
        st.caption(
            f"对话模型：{settings.deepseek_model}  \n"
            f"嵌入模型：{settings.embedding_model}  \n"
            f"当前召回 top_k：{st.session_state.top_k}"
        )


# ============================================================
# 问答页
# ============================================================
def _split_answer_and_references(text: str) -> str:
    """去掉回答末尾由问答链追加的“参考来源：”文本行（来源改用标签胶囊展示）。"""
    return re.sub(r"\n*参考来源：.*$", "", text, flags=re.DOTALL).strip()


def render_source_tags(source_documents: list) -> None:
    """把召回文档渲染为去重后的来源标签胶囊，最多展示 5 条。"""
    tags: list[str] = []
    seen: set[tuple] = set()
    for doc in source_documents:
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", "未知文件"))
        page = metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        safe_source = html.escape(source)
        label = f"📄 第{page}页 · {safe_source}" if page is not None else f"📄 {safe_source}"
        tags.append(f'<span class="src-tag">{label}</span>')

    tags = tags[:5]
    if tags:
        st.markdown('<div class="src-title">参考来源：</div>', unsafe_allow_html=True)
        st.markdown("".join(tags), unsafe_allow_html=True)


def render_evidence_expander(source_documents: list) -> None:
    """在回答下渲染可折叠的「检索到的原文片段」，展示文件名、chunk_index 与原文前 200 字。"""
    if not source_documents:
        return
    with st.expander(f"查看检索到的原文片段（共 {len(source_documents)} 条）"):
        for index, doc in enumerate(source_documents, start=1):
            metadata = getattr(doc, "metadata", {}) or {}
            source = html.escape(str(metadata.get("source", "未知文件")))
            chunk_index = metadata.get("chunk_index", "-")
            page = metadata.get("page")
            location = f"第{page}页" if page is not None else "段落/页未知"
            st.markdown(f"**片段 {index}　📄 {source}　|　{location}　|　chunk #{chunk_index}**")
            preview = (doc.page_content or "")[:200].strip()
            st.caption(preview if preview else "（该片段无文本）")
            if index < len(source_documents):
                st.divider()


def render_chat_tab(tab) -> None:
    """渲染「问答」标签页：历史消息流、输入框、来源标签。"""
    with tab:
        if not st.session_state.kb_ready:
            st.warning("知识库尚未就绪。请先在左侧上传 PDF / Word / PPT 课件，然后点击「构建知识库」。")

        # 历史消息
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(_split_answer_and_references(message["content"]))
                if message["role"] == "assistant" and message.get("sources"):
                    has_evidence = NO_EVIDENCE_ANSWER not in message["content"]
                    if has_evidence:
                        render_source_tags(message["sources"])
                        render_evidence_expander(message["sources"])

        # 输入框：知识库未就绪时禁用
        prompt = st.chat_input(
            "请输入关于课件的问题，例如：什么是 RAG？",
            disabled=not st.session_state.kb_ready,
        )
        if prompt and st.session_state.kb_ready:
            # 先渲染用户消息
            with st.chat_message("user"):
                st.markdown(prompt)

            # 传给问答链的历史不含本轮问题；仅保留 role/content 两个字段
            history_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]

            with st.chat_message("assistant"):
                try:
                    # 流式输出：st.write_stream 逐 token 渲染，返回拼接后的完整回答
                    token_stream = stream_answer(
                        st.session_state.rag_chain, prompt, history_for_llm
                    )
                    full_answer = st.write_stream(token_stream)
                    # 召回文档在流开始前已挂到生成器 state 上
                    source_documents = token_stream.state.get("source_documents", [])
                    is_refusal = NO_EVIDENCE_ANSWER in (full_answer or "")

                    # 参考来源在流结束后单独输出；拒答时不展示来源与原文片段
                    if source_documents and not is_refusal:
                        render_source_tags(source_documents)
                        render_evidence_expander(source_documents)

                    assistant_content = token_stream.state.get("answer") or full_answer or ""
                    assistant_sources = [] if is_refusal else source_documents
                except Exception as exc:  # noqa: BLE001 问答失败也只显示中文错误，不白屏
                    fallback = f"回答失败：{exc}。请检查网络后重试。"
                    st.error(fallback)
                    assistant_content = fallback
                    assistant_sources = []

            # 本轮问答入历史（放在调用之后，确保历史不含当前问题）
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_content, "sources": assistant_sources}
            )


# ============================================================
# 重点页 / 提纲页
# ============================================================
def render_text_tab(tab, state_key: str, empty_tip: str) -> None:
    """渲染纯文本结果页（重点 / 提纲）：卡片展示 Markdown，无结果时显示空状态。"""
    with tab:
        content = st.session_state.get(state_key, "")
        if content:
            # border 容器自带白色卡片样式（与 .ui-card 令牌一致），且能正确包裹子组件
            with st.container(border=True):
                if state_key == "key_points":
                    # 用 Markdown 硬换行保证每个知识点独占一行，保留「· 」前缀
                    st.markdown(content.replace("\n", "  \n"))
                else:
                    st.markdown(content)
        else:
            st.markdown(
                f'<div class="empty-state">{empty_tip}</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# 主入口
# ============================================================
def main() -> None:
    """页面主渲染流程：配置 → 样式 → 状态 → 侧栏 → 标题 → 三个标签页。"""
    st.set_page_config(page_title="课程笔记智能问答助手", layout="wide")
    inject_css()
    init_state()

    render_sidebar()

    # 展示并消费一次性成功提示（同一运行内先渲染 success 再清标志，不会触发额外重跑）
    if st.session_state.kb_notice:
        st.success(st.session_state.kb_notice)
        st.session_state.kb_notice = ""

    st.markdown("# 📚 课程笔记智能问答助手")
    st.caption("上传课程课件构建知识库，围绕课件内容进行多轮智能问答、划重点与生成复习提纲。")

    tab_chat, tab_points, tab_outline = st.tabs(["💬 问答", "✨ 重点", "📝 复习提纲"])
    render_chat_tab(tab_chat)
    render_text_tab(
        tab_points,
        "key_points",
        "还没有划重点内容。<br>请先在左侧构建知识库，再点击「一键划重点」。",
    )
    render_text_tab(
        tab_outline,
        "outline",
        "还没有复习提纲。<br>请先在左侧构建知识库，再点击「生成复习提纲」。",
    )


if __name__ == "__main__":
    main()
