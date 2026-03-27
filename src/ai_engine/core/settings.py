# src/ai_engine/core/settings.py
import os

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ACTIVE_ENV = os.getenv("ENV", "dev").lower()


def project_root() -> str:
    """动态计算项目根目录 (寻找 .env 基础文件作为锚点)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir:
        if os.path.exists(os.path.join(current_dir, ".env")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        current_dir = parent_dir
    return current_dir


class Settings(BaseSettings):
    # ===============================
    # 项目与运行配置
    # ===============================
    ENV: str = Field(default=ACTIVE_ENV, description="当前激活的运行环境 (dev/prod)")
    PROJECT_NAME: str = Field(default="AI-Engine", description="项目名称")
    PROJECT_DESCRIPTION: str = Field(default="基于 LangServe 与 RAG 架构的底层能力支撑 API",
                                     description="项目描述，用于 OpenAPI 展示")
    PROJECT_VERSION: str = Field(default="0.1.0", description="项目版本号")
    PROJECT_RELOAD: bool = Field(default=False, description="是否开启 Uvicorn 热重载")
    PROJECT_HOST: str = Field(default="127.0.0.1", description="服务监听绑定的 IP 地址")
    PROJECT_PORT: int = Field(default=8000, description="服务监听的端口")
    MAX_HISTORY_MESSAGES: int = Field(default=20, description="对话历史截断限制，防止上下文 Token 溢出")
    ENABLE_LANGSERVE_EXTRAS: bool = Field(default=False, description="是否开启 LangServe 辅助端点(playground等)")

    # ===============================
    # 路径与目录配置
    # ===============================
    LOG_LEVEL: str = Field(default="INFO", description="日志输出级别")
    LOG_DIR: str = Field(default="logs", description="日志文件存储的相对目录")
    CHROMA_DATA_DIR: str = Field(default="data/chroma_data", description="ChromaDB 本地持久化目录")
    PROMPTS_DATA_DIR: str = Field(default="resource/prompts", description="Prompt 提示词模板统一存放目录")
    KNOWLEDGE_DATA_DIR: str = Field(default="resource/knowledge", description="RAG 业务知识库源文件存放目录")

    # ===============================
    # LLM 与大模型配置
    # ===============================
    QWEN_API_KEY: SecretStr | None = Field(default=None, description="Qwen AI API Key (敏感信息)")
    QWEN_API_BASE: str | None = Field(default=None, description="Qwen API 代理/请求地址")
    QWEN_MODEL_LLM: str = Field(default="qwen-plus", description="用于主干对话与 RAG 生成的大模型名称")
    QWEN_MODEL_EMBEDDING: str = Field(default="text-embedding-v3", description="文本向量化 Embedding 模型名称")
    QWEN_EMBEDDING_DIM: int = Field(default=1024, description="向量维度大小")
    TEMPERATURE: float = Field(default=0.0, description="模型生成温度，0.0 适合 RAG 与严格分类")

    # ===============================
    # Rerank 与检索配置
    # ===============================
    QWEN_MODEL_RERANK: str = Field(default="gte-rerank-v2", description="二次高精度排序重排模型名称")
    RERANK_THRESHOLD: float = Field(default=0.1, description="重排分数过滤阈值，低于此分数视为无关")
    RERANK_TOP_N: int = Field(default=3, description="重排后最终喂给大模型的精华片段数")
    VECTOR_SEARCH_TOP_K: int = Field(default=10, description="向量检索初筛抓取的片段数 (应大于 TOP_N)")

    # ===============================
    # 数据库配置
    # ===============================
    PG_USER: str = Field(default="postgres", description="PostgreSQL 数据库用户名")
    PG_PASSWORD: SecretStr = Field(default="password", description="PostgreSQL 数据库密码")
    PG_HOST: str = Field(default="127.0.0.1", description="PostgreSQL 主机地址")
    PG_PORT: int = Field(default=5432, description="PostgreSQL 端口")
    PG_DB: str = Field(default="ai_engine", description="PostgreSQL 数据库名称")

    # ===============================
    # 数据库连接池高级配置
    # ===============================
    DB_POOL_SIZE: int = Field(default=20, description="数据库连接池基础容量")
    DB_MAX_OVERFLOW: int = Field(default=30, description="连接池最大溢出容量")
    DB_ECHO: bool = Field(default=False, description="是否在控制台打印底层 SQL 语句")

    # ===============================
    # 向量数据库引擎切换
    # ===============================
    VECTOR_STORE_TYPE: str = Field(default="postgresql", description="向量数据库引擎: 'chroma' 或 'postgresql'")
    INIT_KNOWLEDGE_BASE: bool = Field(default=False, description="是否在服务启动时强制重新初始化知识库")

    # ===============================
    # 动态路径与连接串计算
    # ===============================
    @property
    def project_root_dir(self) -> str:
        return project_root()

    @property
    def chroma_persist_dir(self) -> str:
        """计算向量数据库存储的绝对路径"""
        if os.path.isabs(self.CHROMA_DATA_DIR):
            return self.CHROMA_DATA_DIR
        return os.path.join(self.project_root_dir, self.CHROMA_DATA_DIR)

    @property
    def log_save_path(self) -> str:
        """计算日志存储的绝对路径"""
        if os.path.isabs(self.LOG_DIR):
            return self.LOG_DIR
        return os.path.join(self.project_root_dir, self.LOG_DIR)

    @property
    def prompt_dir(self) -> str:
        """计算 Prompt 模板存储的绝对路径"""
        if os.path.isabs(self.PROMPTS_DATA_DIR):
            return self.PROMPTS_DATA_DIR
        return os.path.join(self.project_root_dir, self.PROMPTS_DATA_DIR)

    @property
    def knowledge_dir(self) -> str:
        """计算 Knowledge 模板存储的绝对路径"""
        if os.path.isabs(self.KNOWLEDGE_DATA_DIR):
            return self.KNOWLEDGE_DATA_DIR
        return os.path.join(self.project_root_dir, self.KNOWLEDGE_DATA_DIR)

    def get_prompt_path(self, filename: str) -> str:
        """获取具体某个 Prompt 文件的路径"""
        return os.path.join(self.prompt_dir, filename)

    @property
    def sync_postgres_url(self):
        return f"postgresql+psycopg://{self.PG_USER}:{self.PG_PASSWORD.get_secret_value()}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"

    # ===============================
    # 多环境级联加载
    # ===============================
    model_config = SettingsConfigDict(
        env_file=(
            os.path.join(project_root(), ".env"),
            os.path.join(project_root(), f".env.{ACTIVE_ENV}")
        ),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# 实例化单例
try:
    settings = Settings()

    print(f"AIOS Configuration Loaded! Active Profile: [{settings.ENV.upper()}]")

    if not settings.QWEN_API_KEY:
        raise ValueError(
            f"无法读取 QWEN_API_KEY，请检查 .env 或 .env.{ACTIVE_ENV} 中是否配置正确！")

except Exception as e:
    print(f"配置文件加载失败！错误详情: {e}")
    raise e
