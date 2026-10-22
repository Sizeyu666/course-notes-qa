# -*- coding: utf-8 -*-
"""
文档加载模块。

支持三种课程课件格式，统一输出为 LangChain 的 Document 列表：
- .pdf ：基于 pypdf 逐页提取文本，每页生成一个 Document（可识别加密与无文本扫描件）；
- .docx：基于 python-docx 遍历段落，每个非空段落生成一个 Document，并保留标题层级；
- .pptx：基于 python-pptx 遍历每张幻灯片的文本框，每页幻灯片拼接为一个 Document。

另外提供 load_uploaded_files()，兼容 Streamlit 上传的字节流：先落盘到系统临时目录，
再复用 load_files() 加载，临时文件用完自动清理。

所有单文件异常都只记录中文警告并跳过，不会中断整批加载。
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence, Union

from docx import Document as DocxDocument
from langchain_core.documents import Document
from pptx import Presentation
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

logger = logging.getLogger(__name__)

# 支持的文件扩展名（小写，含点）
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx"}

# 标题样式名匹配：英文 “Heading 1” / 中文 “标题 1”
_HEADING_PATTERN = re.compile(r"(?:heading|标题)\s*(\d+)", re.IGNORECASE)

# Streamlit UploadedFile 或显式二元组均可作为上传文件入参
UploadedFile = Union[BinaryIO, tuple[str, Union[bytes, BinaryIO]]]


def _base_metadata(source: str, file_type: str, page: int) -> dict:
    """构造统一的基础元数据：source 为不含目录的文件名，file_type 为格式标识，page 为页码/序号。"""
    return {"source": source, "file_type": file_type, "page": page}


def _load_pdf(file_path: str, source: str) -> list[Document]:
    """
    使用 pypdf 逐页提取 PDF 文本，每页返回一个 Document。

    异常处理策略（均转为中文提示，不向外崩溃）：
    - 文件不是合法 PDF：记录警告并返回空列表；
    - 加密 PDF：先尝试空密码，失败则警告跳过；
    - 全部页面均提取不到文本（疑似扫描件）：警告并返回空列表；
    - 个别空白页：警告并跳过该页。
    """
    try:
        reader = PdfReader(file_path)
    except (PdfReadError, OSError) as exc:
        logger.warning("文件 %s 无法作为 PDF 解析，已跳过。原因：%s", source, exc)
        return []

    # 加密 PDF：pypdf 必须先 decrypt 才能读页面
    if reader.is_encrypted:
        decrypt_result = 0
        try:
            decrypt_result = reader.decrypt("")  # 先尝试空密码（常见于“权限加密”的 PDF）
        except Exception as exc:  # noqa: BLE001
            logger.warning("文件 %s 为加密 PDF，空密码解密失败，已跳过。原因：%s", source, exc)
            return []
        if decrypt_result == 0:
            logger.warning("文件 %s 为加密 PDF 且需要密码，本系统暂不支持，已跳过。", source)
            return []
        logger.info("文件 %s 为加密 PDF，已通过空密码解密。", source)

    documents: list[Document] = []
    empty_pages = 0
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except FileNotDecryptedError as exc:
            logger.warning("文件 %s 第 %d 页文本提取失败，已跳过该页。原因：%s", source, index, exc)
            empty_pages += 1
            continue
        if not text:
            empty_pages += 1
            continue
        metadata = _base_metadata(source, "pdf", index)
        metadata["total_pages"] = len(reader.pages)
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        if empty_pages == len(reader.pages) and len(reader.pages) > 0:
            logger.warning(
                "文件 %s 的 %d 个页面均未提取到文本，疑似无文字层的扫描件 PDF，"
                "需先经过 OCR 处理，已跳过。",
                source,
                len(reader.pages),
            )
        else:
            logger.warning("文件 %s 未提取到任何有效文本，已跳过。", source)
    return documents


def _load_docx(file_path: str, source: str) -> list[Document]:
    """
    使用 python-docx 遍历 Word 段落，每个非空段落返回一个 Document。

    - metadata.page 为非空段落的累计序号（从 1 开始）；
    - 标题层级保留在 metadata.heading_level（1~N 表示标题级别，0 表示正文段落）。
    """
    try:
        docx = DocxDocument(file_path)
    except Exception as exc:  # noqa: BLE001 损坏的 docx（非 zip 结构等）统一兜底
        logger.warning("文件 %s 无法作为 Word 文档解析，已跳过。原因：%s", source, exc)
        return []

    documents: list[Document] = []
    paragraph_no = 0  # 非空段落累计序号
    for paragraph in docx.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        paragraph_no += 1

        # 从样式名识别标题级别，识别不到时按正文处理
        heading_level = 0
        style_name = paragraph.style.name if paragraph.style is not None else ""
        matched = _HEADING_PATTERN.search(style_name or "")
        if matched:
            heading_level = int(matched.group(1))

        metadata = _base_metadata(source, "docx", paragraph_no)
        metadata["heading_level"] = heading_level
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        logger.warning("Word 文件 %s 中没有任何非空段落（疑似空白或纯图片文档），已跳过。", source)
    return documents


def _extract_slide_text(slide, source: str, slide_no: int) -> str:
    """遍历一张幻灯片内的全部形状，拼接所有文本框中的非空文本行。"""
    lines: list[str] = []
    for shape in slide.shapes:
        if not getattr(shape, "has_text_frame", False):
            continue  # 图片、表格等非文本框形状本次不提取
        for paragraph in shape.text_frame.paragraphs:
            line = "".join(run.text for run in paragraph.runs).strip()
            if not line:
                # 部分文本不通过 run 组织时退回整段文本
                line = paragraph.text.strip()
            if line:
                lines.append(line)
    return "\n".join(lines).strip()


def _load_pptx(file_path: str, source: str) -> list[Document]:
    """使用 python-pptx 遍历每张幻灯片，将每页全部文本框内容拼接为一个 Document。"""
    try:
        presentation = Presentation(file_path)
    except Exception as exc:  # noqa: BLE001 损坏的 pptx 统一兜底
        logger.warning("文件 %s 无法作为 PowerPoint 文档解析，已跳过。原因：%s", source, exc)
        return []

    documents: list[Document] = []
    empty_slides = 0
    for index, slide in enumerate(presentation.slides, start=1):
        text = _extract_slide_text(slide, source, index)
        if not text:
            empty_slides += 1
            logger.info("文件 %s 第 %d 页幻灯片不含文本，已跳过该页。", source, index)
            continue
        metadata = _base_metadata(source, "pptx", index)
        metadata["total_slides"] = len(presentation.slides)
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        logger.warning(
            "PPT 文件 %s 的 %d 页幻灯片均未提取到文本（疑似纯图片课件），已跳过。",
            source,
            empty_slides,
        )
    return documents


def load_files(file_paths: Sequence[str]) -> list[Document]:
    """
    按扩展名分派加载多个文件，返回拍平后的 Document 列表。

    - 不支持的扩展名：记录中文警告后跳过；
    - 空文件：记录中文警告后跳过；
    - 单个文件解析失败不影响其余文件。
    """
    all_documents: list[Document] = []
    for raw_path in file_paths:
        file_path = os.fspath(raw_path)
        source = os.path.basename(file_path)

        if not os.path.exists(file_path):
            logger.warning("文件 %s 不存在，已跳过。", file_path)
            continue
        if os.path.getsize(file_path) == 0:
            logger.warning("文件 %s 大小为 0（空文件），已跳过。", source)
            continue

        extension = Path(file_path).suffix.lower()
        if extension == ".pdf":
            all_documents.extend(_load_pdf(file_path, source))
        elif extension == ".docx":
            all_documents.extend(_load_docx(file_path, source))
        elif extension == ".pptx":
            all_documents.extend(_load_pptx(file_path, source))
        else:
            logger.warning(
                "文件 %s 的格式 %s 暂不支持（仅支持 .pdf/.docx/.pptx），已跳过。",
                source,
                extension or "无扩展名",
            )
    return all_documents


def _read_uploaded_bytes(item: UploadedFile) -> tuple[str, bytes]:
    """
    从上传项中取出 (原始文件名, 文件字节)。

    兼容两种入参：
    - Streamlit UploadedFile 等文件对象：具备 name 属性与 read/getvalue 方法；
    - (file_name, bytes 或文件对象) 二元组。
    """
    if isinstance(item, tuple):
        file_name, payload = item
    else:
        file_name = getattr(item, "name", None)
        payload = item
        if not file_name:
            raise ValueError("上传的文件对象缺少 name 属性，无法确定原始文件名。")

    if isinstance(payload, bytes):
        return file_name, payload

    # 文件对象：优先 getvalue（BytesIO/UploadedFile），退回 read
    try:
        getter = getattr(payload, "getvalue", None)
        content = getter() if callable(getter) else payload.read()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"读取上传文件 {file_name} 的字节流失败：{exc}") from exc
    if not isinstance(content, bytes):
        raise RuntimeError(f"上传文件 {file_name} 的内容不是字节数据，无法落盘加载。")
    return file_name, content


def load_uploaded_files(uploaded_files: Iterable[UploadedFile]) -> list[Document]:
    """
    加载 Streamlit 上传的文件（字节流 + 文件名）。

    内部把每个文件以原始文件名写入系统临时目录，再复用 load_files() 加载，
    从而保证 Document.metadata.source 仍是用户看到的原始文件名；临时目录退出时自动清理。
    """
    temp_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="course_notes_upload_") as temp_dir:
        for index, item in enumerate(uploaded_files):
            try:
                file_name, content = _read_uploaded_bytes(item)
            except (ValueError, RuntimeError) as exc:
                logger.warning("%s，已跳过该上传项。", exc)
                continue

            # 空字节直接给出中文警告，避免落盘后再走“空文件”分支造成重复提示
            if not content:
                logger.warning("上传文件 %s 内容为空（0 字节），已跳过。", file_name)
                continue

            safe_name = os.path.basename(file_name)  # 防止文件名中夹带路径分隔符
            # 每个文件落到独立子目录：不同内容但同名的文件不会互相覆盖，
            # 且 basename 仍是原始文件名，保证 metadata.source 与用户看到的一致。
            file_dir = os.path.join(temp_dir, str(index))
            try:
                os.makedirs(file_dir, exist_ok=True)
            except OSError as exc:
                logger.warning("为上传文件 %s 创建临时目录失败，已跳过。原因：%s", safe_name, exc)
                continue
            temp_path = os.path.join(file_dir, safe_name)
            try:
                with open(temp_path, "wb") as f:
                    f.write(content)
            except OSError as exc:
                logger.warning("上传文件 %s 写入临时目录失败，已跳过。原因：%s", safe_name, exc)
                continue
            temp_paths.append(temp_path)

        return load_files(temp_paths)
