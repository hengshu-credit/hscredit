"""Agent Skills 稳定错误协议。"""

from typing import Any, Dict, Optional


class SkillExecutionError(RuntimeError):
    """携带稳定错误码和中文消息的 Skill 执行异常。

    :param code: 稳定的机器可读错误码。
    :param message: 面向用户的中文错误详情。
    :param field: 错误涉及的请求字段路径；None 表示非单一字段错误。 默认 ``None``。
    :param cause: 底层异常对象；默认不将其文本写入错误信封。
    """

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
        """转换为不包含明细数据的 JSON 兼容错误信封。

        :param debug: 是否附加底层异常文本；False 仅输出底层异常类型。 默认 ``False``。
        :return: 包含 status 和 error 的 JSON 兼容字典。
        """
        error: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        if self.cause is not None:
            error["cause_type"] = type(self.cause).__name__
            if debug:
                error["cause"] = str(self.cause)
        return {"status": "error", "error": error}
