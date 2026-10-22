# -*- coding: utf-8 -*-
"""
向量库模块（Chroma 持久化）。

- build_vector_store()：用切好的 chunks 全量构建（重建）本地 Chroma 向量库，
  构建前先删除同名旧 collection，保证不与上一次的索引串数据；
- load_vector_store()：加载已持久化的向量库，库不存在时返回 None；
- get_retriever()：基于向量库生成相似度检索器，召回数量取 Settings.top_k。

说明：langchain-chroma 新版通过 persist_directory / PersistentClient 自动持久化，
无需手动调用 persist()。重建时采用“删除旧 collection”而非“删除磁盘目录”，
可规避 Windows 上同进程旧客户端仍持有索引文件句柄导致的文件占用错误。
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
import sqlite3
import uuid
from typing import Optional

import chromadb
from langchain_chroma import Chroma

from src.config import get_settings

logger = logging.getLogger(__name__)

# 固定的集合名，整个课程知识库统一使用
COLLECTION_NAME = "course_notes"


def _persist_dir() -> str:
    """返回向量库持久化目录的绝对路径（相对路径基于当前工作目录解析）。"""
    settings = get_settings()
    return os.path.abspath(settings.vector_store_dir)


def _get_client() -> chromadb.api.ClientAPI:
    """创建（或打开）持久化目录对应的 Chroma 本地客户端，目录不存在时自动创建。"""
    persist_dir = _persist_dir()
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def _delete_collection_if_exists(client) -> None:
    """删除同名旧 collection（不存在则静默跳过），用于重建前清空旧数据。"""
    try:
        existing_names = {collection.name for collection in client.list_collections()}
        if COLLECTION_NAME in existing_names:
            client.delete_collection(COLLECTION_NAME)
            logger.info("已删除旧 collection：%s", COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001 不同版本异常类型不一，统一转中文错误
        raise RuntimeError(f"清理旧向量集合 {COLLECTION_NAME} 失败。原始错误：{exc}") from exc


def _cleanup_orphan_segments(persist_dir: str) -> None:
    """
    清扫“已删除 collection 遗留的孤儿索引目录”（尽力而为，任何失败都静默降级）。

    背景：在 Windows 上删除 collection 时，若同进程仍持有旧索引文件句柄，
    HNSW 段目录会残留在磁盘上；chromadb 重启后也不会自动回收，反复重建会累积垃圾。
    判定方式：Chroma 的每个向量段以 UUID 命名为根目录下的文件夹，并登记在
    chroma.sqlite3 的 segments 表中；目录名是 UUID 但不在该表中，即为孤儿。
    """
    db_file = os.path.join(persist_dir, "chroma.sqlite3")
    if not os.path.exists(db_file):
        return

    active_segment_ids: set[str] = set()
    try:
        # 以只读方式打开，避免与 chromadb 自身的数据库连接互相干扰
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        try:
            active_segment_ids = {row[0] for row in conn.execute("SELECT id FROM segments")}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.debug("读取 segments 表失败，跳过孤儿目录清扫：%s", exc)
        return

    for entry in os.listdir(persist_dir):
        entry_path = os.path.join(persist_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        # 只处理 UUID 命名的段目录，避免误删 chroma 的其他内部文件
        try:
            uuid.UUID(entry)
        except ValueError:
            continue
        if entry in active_segment_ids:
            continue
        try:
            shutil.rmtree(entry_path)
            logger.info("已清扫孤儿索引目录：%s", entry)
        except OSError as exc:
            # 文件仍被占用（如同进程旧实例未释放）时放弃本次清扫，下次重建再清
            logger.debug("孤儿目录 %s 暂时无法删除，将在下次重建时重试：%s", entry, exc)


def build_vector_store(chunks, embeddings) -> Chroma:
    """
    用 chunks 全量构建（重建）Chroma 向量库并返回实例。

    :param chunks: 切分后的 Document 列表（由 splitter.split_documents 产生）
    :param embeddings: Embedding 模型（由 embeddings.get_embeddings 产生）
    """
    if not chunks:
        raise ValueError("没有可入库的文本块（chunks 为空），请先加载并切分文档后再构建向量库。")

    persist_dir = _persist_dir()

    # 先清扫历史重建遗留的孤儿索引目录（跨进程时可正常删除）
    _cleanup_orphan_segments(persist_dir)

    client = _get_client()

    # 先删同名旧集合，避免新文档被追加进旧索引造成数据翻倍/串数据
    _delete_collection_if_exists(client)
    # 触发垃圾回收，尽量释放同进程内旧向量库对象持有的索引文件句柄（Windows）
    gc.collect()
    # 删除后再清扫一次：刚删掉的旧段若句柄已释放即可立即回收
    _cleanup_orphan_segments(persist_dir)

    try:
        # 复用同一 client 创建新集合并写入；数据随 PersistentClient 自动持久化到磁盘
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            client=client,
        )
    except Exception as exc:  # noqa: BLE001 统一转中文错误
        raise RuntimeError(f"构建 Chroma 向量库失败，请检查 Embedding 模型与磁盘权限。原始错误：{exc}") from exc

    logger.info(
        "向量库构建完成：%d 个 chunk 已写入集合 %s（目录：%s）。",
        len(chunks),
        COLLECTION_NAME,
        persist_dir,
    )
    return vector_store


def load_vector_store(embeddings) -> Optional[Chroma]:
    """
    从持久化目录加载已有向量库；集合不存在（从未构建或已被清空）时返回 None。
    """
    client = _get_client()
    try:
        existing_names = {collection.name for collection in client.list_collections()}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"读取 Chroma 集合列表失败，目录：{_persist_dir()}。原始错误：{exc}") from exc

    if COLLECTION_NAME not in existing_names:
        logger.info("向量库中不存在集合 %s，请先构建知识库。", COLLECTION_NAME)
        return None

    try:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            client=client,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"加载 Chroma 向量库失败，目录：{_persist_dir()}。原始错误：{exc}") from exc

    logger.info("已加载向量库：集合 %s（目录：%s）。", COLLECTION_NAME, _persist_dir())
    return vector_store


def get_retriever(vector_store: Chroma, k: Optional[int] = None):
    """
    基于向量库返回相似度检索器。

    :param k: 召回块数；为 None 时取 Settings.top_k。界面滑块调整 k 后，
              只需用新的 k 重建检索器，无需重建向量库。
    """
    settings = get_settings()
    retrieve_k = k if isinstance(k, int) and k > 0 else settings.top_k
    return vector_store.as_retriever(search_kwargs={"k": retrieve_k})
