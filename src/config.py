# -*- coding: utf-8 -*-
"""
统一配置模块。

职责：
1. 启动时从项目根目录的 .env 加载环境变量（文件不存在时静默跳过，继续使用系统环境变量）；
2. 用 dataclass 集中定义全部配置项（大模型、Embedding、向量库、切分与检索参数）；
3. 通过 get_settings() 提供全局单例；
4. 通过 check_keys() 检查必填密钥是否缺失。

安全约定：密钥只允许从环境变量读取，禁止在代码中硬编码；打印配置时密钥自动打码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（src/ 的上一级），保证无论从哪个工作目录启动都能定位到根目录的 .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

# 加载根目录 .env。
# - 文件不存在时 python-dotenv 默认静默返回 False，不抛异常；
# - override=False：已存在的系统环境变量优先级更高，.env 只负责补缺；
# - 显式指定 utf-8，避免 Windows 终端默认 GBK 编码读取中文注释时出错。
load_dotenv(dotenv_path=ENV_FILE_PATH, override=False, encoding="utf-8")

# 本地 Embedding 提供商标识：EMBEDDING_PROVIDER 取该值时使用本地 bge 模型，
# 其余取值（如 dashscope）视为使用通义在线 Embedding。
LOCAL_PROVIDER = "local"


def _get_streamlit_secret(key: str) -> str:
    """
    从 Streamlit secrets 读取配置（部署 Streamlit Cloud 时的密钥来源）。

    非 Streamlit 环境（命令行脚本、单元测试）import streamlit 或访问 st.secrets
    都可能失败，统一静默返回空串，由调用方回退到默认值。
    """
    try:
        import streamlit as st

        value = st.secrets.get(key, "")
        return str(value).strip() if value else ""
    except Exception:  # noqa: BLE001 非 Streamlit 环境或未配置 secrets 均视为缺失
        return ""


def _get_str_env(key: str, default: str = "") -> str:
    """读取字符串型配置：系统环境变量 > .env > Streamlit secrets > default。"""
    value = os.getenv(key)
    if value and value.strip():
        return value.strip()
    secret = _get_streamlit_secret(key)
    if secret:
        return secret
    return default


def _get_int_env(key: str, default: int) -> int:
    """读取整数型配置（环境变量 > Streamlit secrets > default）；值非法时抛出中文错误。"""
    raw = os.getenv(key) or _get_streamlit_secret(key)
    if not raw or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"配置项 {key} 的值 {raw!r} 不是合法整数，请检查 .env 或 Streamlit secrets（当前值应为整数）"
        ) from exc


def _mask_secret(value: str) -> str:
    """对密钥打码：空值显示“未配置”，短值全打码，长值仅保留前 3 位与后 4 位。"""
    if not value:
        return "<未配置>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}****{value[-4:]}"


@dataclass(frozen=True)
class Settings:
    """全局配置对象（不可变），所有字段由环境变量注入，缺省时使用代码内默认值。"""

    # DeepSeek 大模型（OpenAI 兼容接口）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # 通义千问 DashScope（仅在使用通义在线 Embedding 时需要 key）
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Embedding 配置：provider=local 时走本地 bge，其他取值走通义在线接口
    embedding_provider: str = LOCAL_PROVIDER
    embedding_model: str = "text-embedding-v3"

    # 向量库持久化目录、文本切分与检索参数
    vector_store_dir: str = "./vector_store"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    @staticmethod
    def from_env() -> "Settings":
        """从环境变量构造配置对象；非法的整数配置会在此处抛出中文错误。"""
        return Settings(
            deepseek_api_key=_get_str_env("DEEPSEEK_API_KEY"),
            deepseek_base_url=_get_str_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=_get_str_env("DEEPSEEK_MODEL", "deepseek-chat"),
            dashscope_api_key=_get_str_env("DASHSCOPE_API_KEY"),
            dashscope_base_url=_get_str_env(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            embedding_provider=_get_str_env("EMBEDDING_PROVIDER", LOCAL_PROVIDER),
            embedding_model=_get_str_env("EMBEDDING_MODEL", "text-embedding-v3"),
            vector_store_dir=_get_str_env("VECTOR_STORE_DIR", "./vector_store"),
            chunk_size=_get_int_env("CHUNK_SIZE", 500),
            chunk_overlap=_get_int_env("CHUNK_OVERLAP", 50),
            top_k=_get_int_env("TOP_K", 4),
        )

    def check_keys(self) -> list[str]:
        """
        检查必填密钥是否缺失，返回缺失项的环境变量名列表；返回空列表表示配置齐全。

        规则：
        - DEEPSEEK_API_KEY：始终必填（问答与摘要都依赖 DeepSeek）；
        - DASHSCOPE_API_KEY：仅当 Embedding 使用通义在线服务（provider 非 local）时必填，
          使用本地 bge 时不需要。
        """
        missing: list[str] = []
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if self.embedding_provider.lower() != LOCAL_PROVIDER and not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY")
        return missing

    def __repr__(self) -> str:
        """打印配置时对所有密钥字段自动打码，避免明文泄露到日志或终端。"""
        parts: list[str] = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("_api_key"):
                value = _mask_secret(value)
            parts.append(f"{field.name}={value!r}")
        return f"Settings({', '.join(parts)})"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一的配置单例（首次调用时从环境变量加载，之后直接缓存）。"""
    return Settings.from_env()


def build_chat_llm(temperature: float = 0.2):
    """
    统一的大模型客户端初始化入口（DeepSeek，OpenAI 兼容接口）。

    供 RAG 问答链、划重点、复习提纲等所有需要大模型的模块复用，
    避免在多处重复编写密钥 / 地址 / 模型名配置。langchain_openai 采用延迟导入，
    使仅使用配置项的模块不必承担该依赖的导入开销。
    """
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise ValueError(
            "未配置 DEEPSEEK_API_KEY，无法初始化大模型。请在 .env 中填入 DeepSeek 密钥后重试。"
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "未安装 langchain-openai，无法初始化 DeepSeek 聊天模型，"
            "请先执行 pip install -r requirements.txt。"
        ) from exc
    try:
        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001 初始化失败统一转中文错误
        raise RuntimeError(
            f"DeepSeek 聊天模型初始化失败，请检查密钥与接口地址。原始错误：{exc}"
        ) from exc
