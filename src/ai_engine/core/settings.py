# src/ai_engine/core/sittings.py
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> str:
    """动态计算项目根目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir:
        if os.path.exists(os.path.join(current_dir, ".env")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # 根据目录结构向上推算
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        current_dir = parent_dir
    return current_dir


class Settings(BaseSettings):
    # --- 项目基本信息 ---
    PROJECT_NAME: str = Field(default="AI-Engine")
    PROJECT_VERSION: str = Field(default="0.1.0")
    PROJECT_RELOAD: bool = Field(default=False, description="是否开启 Uvicorn 热重载")
    PROJECT_HOST: str = Field(default="127.0.0.1", description="服务监听地址")
    PROJECT_PORT: int = Field(default=8000, description="服务监听端口")

    # --- 日志配置 ---
    LOG_LEVEL: str = Field(default="INFO")
    LOG_DIR: str = Field(default="logs")

    # --- Chroma配置 ---
    CHROMA_DATA_DIR: str = Field(default="data/chroma_data")

    # --- Prompts配置 ---
    PROMPTS_DATA_DIR: str = Field(default="resource/prompts")

    # --- Knowledge配置 ---
    KNOWLEDGE_DATA_DIR: str = Field(default="resource/knowledge")

    # --- LLM 配置 ---
    QWEN_API_KEY: str | None = Field(default=None, description="Qwen AI API Key")
    QWEN_API_BASE: str | None = Field(default=None, description="Qwen API 代理地址")
    QWEN_MODEL_LLM: str = Field(default="qwen-plus")
    QWEN_MODEL_EMBEDDING: str = Field(default="text-embedding-v3")
    QWEN_EMBEDDING_DIM: int = Field(default=1024)
    TEMPERATURE: float = Field(default=0.0)

    # --- ✨ Rerank 增强配置 ---
    # 模型选择：gte-rerank-v2, qwen3-rerank, qwen3-vl-rerank
    QWEN_MODEL_RERANK: str = Field(default="gte-rerank-v2", description="重排模型名称")
    RERANK_THRESHOLD: float = Field(default=0.1, description="重排分数过滤阈值")
    RERANK_TOP_N: int = Field(default=3, description="重排后保留的最终片段数")
    VECTOR_SEARCH_TOP_K: int = Field(default=10, description="向量检索初筛抓取的片段数")

    # --- PostgreSQL 数据库配置 ---
    PG_USER: str = Field(default="postgres", description="PostgreSQL 数据库用户名")
    PG_PASSWORD: str = Field(default="password", description="PostgreSQL 数据库密码")
    PG_HOST: str = Field(default="127.0.0.1", description="PostgreSQL 数据库主机地址 (例如 127.0.0.1 或 localhost)")
    PG_PORT: int = Field(default=5432, description="PostgreSQL 数据库连接端口 (默认 5432)")
    PG_DB: str = Field(default="ai_engine", description="PostgreSQL 数据库名称")

    # --- 数据库连接池高级配置 ---
    DB_POOL_SIZE: int = Field(default=20, description="数据库连接池的基础容量 (系统常驻的空闲连接数)")
    DB_MAX_OVERFLOW: int = Field(default=30, description="连接池满时的最大溢出容量 (高并发峰值时允许临时多建的连接数)")
    DB_ECHO: bool = Field(default=False, description="是否在控制台打印底层执行的 SQL 语句 (建议仅在 Debug 时开启)")

    # --- 智能路径寻址 ---
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
    def postgres_url(self) -> str:
        """生成异步 PostgreSQL 连接字符串 (使用顶级性能的 asyncpg 驱动)"""
        return f"postgresql+asyncpg://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"

    # ===============================
    # Settings 行为配置
    # ===============================
    model_config = SettingsConfigDict(
        env_file=os.path.join(project_root(), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# 实例化单例
try:
    settings = Settings()

    # 终极防御：如果加载完发现关键配置还是 None，说明 .env 内容不全
    if not settings.QWEN_API_KEY:
        raise ValueError(f"无法从环境变量或 .env 中读取 QWEN_API_KEY，加载路径: {os.path.join(project_root(), '.env')}")

except Exception as e:
    print(f"❌ 配置文件加载失败！")
    print(f"错误详情: {e}")
    raise e
