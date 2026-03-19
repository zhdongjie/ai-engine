# app/core/logger.py
import logging
import os
import sys

from loguru import logger

from ai_engine.core.settings import settings


class InterceptHandler(logging.Handler):
    """
    拦截标准 logging 的日志，并将其路由给 loguru
    这样可以把 Uvicorn 和 FastAPI 的底层日志统一接管
    """

    def emit(self, record: logging.LogRecord):
        # 尝试获取对应 loguru 的 level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 寻找调用栈中最初触发日志的位置
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """初始化全局日志配置"""

    # 1. 移除 loguru 默认的控制台输出（防止重复打日志）
    logger.remove()

    # 2. 重新添加控制台输出，并自定义格式和颜色
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
               "| <level>{level: <8}</level> "
               "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        enqueue=True,  # 开启异步队列，保证高并发下日志不阻塞
    )

    # 3. 处理日志基础路径
    log_path_prefix = os.path.join(settings.log_save_path, "{time:YYYY-MM-DD}")

    # 4. 常规业务日志文件 (info.log)
    # filter 作用：只允许 INFO 和 WARNING 级别的日志写入此文件
    logger.add(
        os.path.join(log_path_prefix, "info.log"),
        level="INFO",
        filter=lambda record: record["level"].name in ["INFO", "WARNING"],
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",  # 每天午夜 0 点轮转
        retention="30 days",  # 保留 30 天
        encoding="utf-8",
        enqueue=True,
    )

    # 5. 错误告警日志文件 (error.log)
    # 专门记录 ERROR 及以上级别 (ERROR, CRITICAL)
    logger.add(
        os.path.join(log_path_prefix, "error.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
        enqueue=True,
    )

    # 6. 接管 Python 标准库和 Uvicorn 的日志
    logging.getLogger().handlers = [InterceptHandler()]

    # 将一些常见的三方 Web 库日志也挂载过来
    for log_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        logging_logger = logging.getLogger(log_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False  # 防止日志向上传递导致重复输出

    logger.info(f"🚀 {settings.PROJECT_NAME} 日志系统初始化完成! 日志根目录: {settings.log_save_path}")
