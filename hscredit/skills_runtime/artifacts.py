"""Agent Skills 制品协议适配，底层发布事务由通用工具层维护。"""

# 保留历史模块引用及异常类型，业务报告无需依赖 skills_runtime。
from ..utils.artifacts import ArtifactTransaction as _ArtifactTransaction
from ..utils.artifacts import summarize_dataframe, _json_value, _column_label, _preview_key
from ..utils.artifacts import os, shutil, tempfile, socket, logger
from .errors import SkillExecutionError


class ArtifactTransaction(_ArtifactTransaction):
    """使用 Skills 稳定错误码的通用制品事务适配器。"""

    error_type = SkillExecutionError
