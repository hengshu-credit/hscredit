"""并行任务参数工具."""

import os
from typing import Optional


def resolve_n_jobs(n_jobs: Optional[int]) -> Optional[int]:
    """解析并行任务数.

    ``-1`` 在 hscredit 中表示保留一个逻辑 CPU 给系统，其余 CPU 用于计算。
    单核环境或无法读取 CPU 数量时至少返回 1。

    :param n_jobs: 用户传入的并行任务数
    :return: 解析后的任务数
    """
    if n_jobs == -1:
        return max((os.cpu_count() or 1) - 1, 1)
    return n_jobs
