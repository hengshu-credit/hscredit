"""Agent Skills 稳定错误协议。"""

from typing import Any, Dict, Optional


class SkillExecutionError(RuntimeError):
    """携带稳定错误码和中文消息的 Skill 执行异常。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.cause = cause

    def to_dict(self, *, debug: bool = False) -> Dict[str, Any]:
        """转换为不包含明细数据的 JSON 兼容错误信封。"""
        error: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        if self.cause is not None:
            error["cause_type"] = type(self.cause).__name__
            if debug:
                error["cause"] = str(self.cause)
        return {"status": "error", "error": error}
