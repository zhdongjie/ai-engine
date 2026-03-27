# --- 阶段 1: 构建阶段 (Builder) ---
FROM python:3.11-slim AS builder

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.3.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# 将 Poetry 加入 PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖 (psycopg 编译可能需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装 (利用 Docker 层缓存)
COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --only main

# --- 阶段 2: 运行阶段 (Final) ---
FROM python:3.11-slim AS final

WORKDIR /app

# 从构建阶段复制安装好的依赖 (Python 库通常在 site-packages)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装运行时的基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 复制源代码和资源
COPY src ./src
COPY resource ./resource
COPY scripts ./scripts
COPY main.py .
COPY .env* .

# 设置 PYTHONPATH 确保模块能被正确导入
ENV PYTHONPATH=/app/src

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "ai_engine.server:app", "--host", "0.0.0.0", "--port", "8000"]