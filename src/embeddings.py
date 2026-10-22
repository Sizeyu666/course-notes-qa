# -*- coding: utf-8 -*-
"""
Embedding 模型封装模块。

根据 Settings.embedding_provider 在两种向量化方案间分派：
- local（默认）：本地 BAAI/bge-small-zh-v1.5，基于 langchain-huggingface +
  sentence-transformers，完全离线运行；首次使用自动从 HuggingFace（国内镜像）
  下载模型（约 100MB），之后优先使用本地缓存；
- dashscope：通义千问 text-embedding-v3 在线接口（OpenAI 兼容协议），
  需要配置 DASHSCOPE_API_KEY。

get_embeddings() 返回进程内单例，避免重复加载模型。
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from src.config import LOCAL_PROVIDER, get_settings

logger = logging.getLogger(__name__)

# HuggingFace 国内镜像端点；仅当用户未自行设置 HF_ENDPOINT 时才注入，不覆盖已有配置
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def _build_local_embeddings():
    """构造本地 HuggingFace BGE 嵌入模型；缺少依赖或加载失败时抛出带中文说明的错误。"""
    settings = get_settings()

    # 仅在未设置时启用国内镜像，保证与用户/CI 的自定义网络配置不冲突。
    # 必须在初始化模型前设置，sentence-transformers 底层通过 huggingface_hub 读取该变量。
    if not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
        logger.info("未检测到 HF_ENDPOINT，已自动设置为国内镜像 %s。", HF_MIRROR_ENDPOINT)

    # 延迟导入：仅在真正使用本地方案时才要求安装 torch / sentence-transformers
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "本地 Embedding 依赖未安装。请先执行：pip install -r requirements-local-embed.txt"
            "（该文件会安装 langchain-huggingface、sentence-transformers 及 torch，体积较大）。"
            "若不想安装本地依赖，也可在 .env 中设置 EMBEDDING_PROVIDER=dashscope 改用在线接口。"
        ) from exc

    try:
        # model_kwargs 指定 CPU 推理，教学场景稳定且无需 CUDA 配置
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # bge 检索建议归一化向量
        )
    except Exception as exc:  # noqa: BLE001 统一转成中文错误，便于用户定位网络/模型问题
        raise RuntimeError(
            f"本地 Embedding 模型 {settings.embedding_model} 加载失败。"
            f"请检查网络（首次需从 {HF_MIRROR_ENDPOINT} 下载约 100MB 模型）或本地缓存是否完整。"
            f"原始错误：{exc}"
        ) from exc

    logger.info("本地 Embedding 模型已就绪：%s（优先使用本地缓存）。", settings.embedding_model)
    return embeddings


def _build_dashscope_embeddings():
    """构造通义千问在线 Embedding（OpenAI 兼容接口）；缺少 key 时给出中文报错。"""
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise ValueError(
            "使用通义在线 Embedding（EMBEDDING_PROVIDER=dashscope）必须在 .env 中配置 "
            "DASHSCOPE_API_KEY。请补全密钥后重试；若无 DashScope 密钥，可将 "
            "EMBEDDING_PROVIDER 改为 local 使用免费的本地 bge 模型。"
        )

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "未安装 langchain-openai，无法使用通义在线 Embedding，请先执行 "
            "pip install -r requirements.txt。"
        ) from exc

    try:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            # DashScope 兼容接口不支持基于 tiktoken 的上下文长度校验，关闭以避免报错
            check_embedding_ctx_length=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"通义 Embedding 客户端初始化失败（模型：{settings.embedding_model}）。"
            f"请检查 DASHSCOPE_API_KEY 与 DASHSCOPE_BASE_URL 是否正确。原始错误：{exc}"
        ) from exc

    logger.info("通义在线 Embedding 已就绪：%s。", settings.embedding_model)
    return embeddings


@lru_cache(maxsize=1)
def get_embeddings():
    """
    获取 Embedding 模型单例。

    按 Settings.embedding_provider 分派：
    - "local"（默认）→ 本地 bge 模型；
    - 其他取值（如 "dashscope"）→ 通义在线接口。
    """
    settings = get_settings()
    provider = settings.embedding_provider.lower().strip()
    if provider == LOCAL_PROVIDER:
        return _build_local_embeddings()
    if provider == "dashscope":
        return _build_dashscope_embeddings()
    raise ValueError(
        f"不支持的 EMBEDDING_PROVIDER 取值：{settings.embedding_provider!r}，"
        f"仅支持 'local'（本地 bge）或 'dashscope'（通义在线接口）。"
    )
