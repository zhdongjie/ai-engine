# src/ai_engine/core/settings.py
import os

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> str:
    """动态计算项目根目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir:
        if os.path.exists(os.path.join(current_dir, ".env.prod")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # 根据目录结构向上推算
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        current_dir = parent_dir
    return current_dir


class Settings(BaseSettings):
    # ===============================
    # 项目基本信息
    # ===============================
    # 项目名称
    PROJECT_NAME: str = Field(default="AI-Engine")
    # 项目描述，用于 OpenAPI 文档展示
    PROJECT_DESCRIPTION: str = Field(default="基于 LangServe 与 RAG 架构的底层能力支撑 API")
    # 项目版本号
    PROJECT_VERSION: str = Field(default="0.1.0")
    # 是否开启 Uvicorn 热重载 (开发环境建议为 True)
    PROJECT_RELOAD: bool = Field(default=False, description="是否开启 Uvicorn 热重载")
    # 服务监听绑定的 IP 地址
    PROJECT_HOST: str = Field(default="127.0.0.1", description="服务监听地址")
    # 服务监听的端口
    PROJECT_PORT: int = Field(default=8000, description="服务监听端口")
    # 对话历史截断限制，防止上下文 Token 溢出
    MAX_HISTORY_MESSAGES: int = Field(default=20, description="对话历史截断限制")
    # 是否开启 LangServe 的辅助端点 (如 playground, feedback 等)，生产环境建议设为 False
    ENABLE_LANGSERVE_EXTRAS: bool = Field(default=False, description="是否开启 LangServe 的辅助端点")

    # ===============================
    # 路径与目录配置
    # ===============================
    # 日志输出级别 (DEBUG, INFO, WARNING, ERROR)
    LOG_LEVEL: str = Field(default="INFO")
    # 日志文件存储的相对目录
    LOG_DIR: str = Field(default="logs")
    # ChromaDB 向量数据库本地持久化的目录
    CHROMA_DATA_DIR: str = Field(default="data/chroma_data")
    # Prompt 提示词模板统一存放目录
    PROMPTS_DATA_DIR: str = Field(default="resource/prompts")
    # RAG 业务知识库源文件存放目录
    KNOWLEDGE_DATA_DIR: str = Field(default="resource/knowledge")

    # ===============================
    # LLM 大模型配置
    # ===============================
    # Qwen AI API Key (敏感信息，使用 SecretStr 脱敏，防止日志泄露)
    QWEN_API_KEY: SecretStr | None = Field(default=None, description="Qwen AI API Key")
    # Qwen API 代理/请求地址
    QWEN_API_BASE: str | None = Field(default=None, description="Qwen API 代理地址")
    # 用于主流程对话和 RAG 生成的大模型名称 (主干大模型)
    QWEN_MODEL_LLM: str = Field(default="qwen-plus")
    # 用于将文本转化为向量的 Embedding 模型名称
    QWEN_MODEL_EMBEDDING: str = Field(default="text-embedding-v3")
    # 向量维度大小 (必须与 Embedding 模型输出的维度一致)
    QWEN_EMBEDDING_DIM: int = Field(default=1024)
    # 模型生成温度，0.0 表示最严谨稳定，适合 RAG 与意图分类
    TEMPERATURE: float = Field(default=0.0)

    # ===============================
    # Rerank 增强检索配置
    # ===============================
    # 重排模型名称 (对初筛结果进行二次高精度排序)
    QWEN_MODEL_RERANK: str = Field(default="gte-rerank-v2", description="重排模型名称")
    # 重排分数过滤阈值，低于此分数的文档片段会被判定为无关并丢弃
    RERANK_THRESHOLD: float = Field(default=0.1, description="重排分数过滤阈值")
    # 重排后最终保留、并喂给大模型的精华上下文片段数
    RERANK_TOP_N: int = Field(default=3, description="重排后保留的最终片段数")
    # 向量检索初筛时抓取的粗略片段数 (数量通常大于 TOP_N，然后再送去重排)
    VECTOR_SEARCH_TOP_K: int = Field(default=10, description="向量检索初筛抓取的片段数")

    # ===============================
    # PostgreSQL 关系型数据库配置
    # ===============================
    # PostgreSQL 数据库用户名
    PG_USER: str = Field(default="postgres", description="PostgreSQL 数据库用户名")
    # PostgreSQL 数据库密码 (敏感信息，使用 SecretStr 脱敏)
    PG_PASSWORD: SecretStr = Field(default="password", description="PostgreSQL 数据库密码")
    # PostgreSQL 数据库主机地址 (例如 127.0.0.1 或线上云库 IP)
    PG_HOST: str = Field(default="127.0.0.1", description="PostgreSQL 数据库主机地址")
    # PostgreSQL 数据库连接端口
    PG_PORT: int = Field(default=5432, description="PostgreSQL 数据库连接端口")
    # PostgreSQL 数据库名称
    PG_DB: str = Field(default="ai_engine", description="PostgreSQL 数据库名称")

    # ===============================
    # 数据库连接池高级配置
    # ===============================
    # 数据库连接池的基础容量 (系统常驻的空闲连接数，随时待命)
    DB_POOL_SIZE: int = Field(default=20, description="数据库连接池的基础容量")
    # 连接池满时的最大溢出容量 (高并发峰值时允许临时多建的连接数)
    DB_MAX_OVERFLOW: int = Field(default=30, description="连接池满时的最大溢出容量")
    # 是否在控制台打印底层执行的 SQL 语句 (建议仅在本地 Debug 时开启)
    DB_ECHO: bool = Field(default=False, description="是否在控制台打印底层执行的 SQL 语句")

    # ===============================
    # 向量数据库引擎切换
    # ===============================
    VECTOR_STORE_TYPE: str = Field(default="chroma", description="向量数据库引擎选择: 'chroma' 或 'postgresql'")

    # ===============================
    # 环境与初始化控制
    # ===============================
    # 是否在服务启动时强制重新初始化知识库
    # 开发环境建议 True，生产环境务必 False
    INIT_KNOWLEDGE_BASE: bool = Field(default=False, description="是否启动时初始化向量库")

    # ===============================
    # 智能路径与 URL 寻址
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
    def postgres_url(self) -> str:
        """生成异步 PostgreSQL 连接字符串"""
        return f"postgresql+asyncpg://{self.PG_USER}:{self.PG_PASSWORD.get_secret_value()}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"

    @property
    def sync_postgres_url(self):
        """生成同步 PostgreSQL 连接字符串"""
        return f"postgresql+psycopg://{self.PG_USER}:{self.PG_PASSWORD.get_secret_value()}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"

    # ===============================
    # Settings 行为配置
    # ===============================
    model_config = SettingsConfigDict(
        env_file=os.path.join(project_root(), ".env.prod"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# 实例化单例
try:
    settings = Settings()

    # 终极防御：如果加载完发现关键配置还是 None，说明 .env.prod 内容不全
    if not settings.QWEN_API_KEY:
        raise ValueError(f"无法从环境变量或 .env.prod 中读取 QWEN_API_KEY，加载路径: {os.path.join(project_root(), '.env.prod')}")

except Exception as e:
    print(f"❌ 配置文件加载失败！")
    print(f"错误详情: {e}")
    raise e
