"""日志工具。

基于 loguru 配置控制台和文件双输出，所有模块通过 get_logger(name) 获取带
模块名绑定的 logger，便于定位流水线阶段问题。
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache

from config.app_config import settings

try:
    from loguru import logger as _loguru_logger
except ModuleNotFoundError:
    _loguru_logger = None


@lru_cache(maxsize=1)
def configure_logger():
    """初始化全局日志输出，只执行一次。"""

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    if _loguru_logger is not None:
        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level=settings.log_level, enqueue=False)
        _loguru_logger.add(
            settings.log_dir / "mul_tree_mmkg_{time:YYYY-MM-DD}.log",
            level=settings.log_level,
            rotation="00:00",
            retention="14 days",
            encoding="utf-8",
            enqueue=False,
        )
        return _loguru_logger

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    file_handler = logging.FileHandler(settings.log_dir / "mul_tree_mmkg.log", encoding="utf-8")
    file_handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root_logger = logging.getLogger("mul_tree_mmkg")
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
    return root_logger


def get_logger(name: str):
    """获取绑定模块名的 logger 实例。"""

    configured = configure_logger()
    if hasattr(configured, "bind"):
        return configured.bind(module=name)
    return logging.getLogger(f"mul_tree_mmkg.{name}")
