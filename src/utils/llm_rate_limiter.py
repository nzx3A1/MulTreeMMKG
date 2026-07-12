"""大模型请求限流工具。

所有文本和多模态客户端共用同一个限流器，避免不同客户端同时把请求压到同一服务端。
"""
from __future__ import annotations

import threading
import time


def qps_to_min_interval(qps: float) -> float:
    """把每秒请求数转换为相邻两次请求的最小间隔；非正数表示不限流。"""

    return 1.0 / qps if qps > 0 else 0.0


class LLMRateLimiter:
    """以请求开始时间为基准的进程内限流器。"""

    def __init__(self) -> None:
        """初始化线程安全的请求时间记录。"""

        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def wait(self, min_interval: float) -> None:
        """等待到可发起下一次请求的时间；间隔为零时直接返回。"""

        if min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            remaining = self._last_request_at + min_interval - now
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()


# 中文说明：文本 LLM 与视觉 LLM 共享限流状态，避免两个客户端分别绕过 QPS 限制。
GLOBAL_LLM_RATE_LIMITER = LLMRateLimiter()
