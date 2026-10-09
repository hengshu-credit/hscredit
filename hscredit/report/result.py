"""报告章节状态与发布清单，不依赖具体计算或渲染器。"""

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Optional

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class ReportSectionResult:
    """一个报告章节的执行状态。"""

    name: str
    status: str = "未执行"
    required: bool = True
    reason: str = ""
    input_rows: Optional[int] = None
    output_rows: Optional[int] = None
    elapsed_seconds: float = 0.0
    sampling: Dict[str, Any] = field(default_factory=dict)


class ReportGenerationError(ValueError):
    """报告未完整生成；result 保留已完成和失败章节以供诊断。"""

    def __init__(self, message: str, result: "ReportResult"):
        super().__init__(message)
        self.result = result


class ReportResult(dict):
    """兼容 ``dict[str, DataFrame]`` 的结构化报告。

    **参数**

    mode : str
        ``strict`` 拒绝必需章节失败，``best_effort`` 保留成功表并记录失败。
    metadata : dict, 可选
        本次运行的目标、字段及样本口径，不保存逐行个人数据。

    **属性**

    sections : dict
        包含成功、不适用、失败和未执行状态的完整章节清单。
    artifacts : list
        仅在成功发布后追加的产物信息。
    """

    def __init__(self, *, mode: str = "best_effort", metadata: Optional[Dict[str, Any]] = None):
        if mode not in {"strict", "best_effort"}:
            raise ValueError("报告 mode 必须为 'strict' 或 'best_effort'")
        super().__init__()
        self.mode = mode
        self.metadata = dict(metadata or {})
        self.sections: Dict[str, ReportSectionResult] = {}
        self.artifacts = []
        self.sheet_mapping: Dict[str, str] = {}

    def plan(self, name: str, *, required: bool = True, input_rows: Optional[int] = None) -> None:
        self.pop(name, None)
        self.sheet_mapping.pop(name, None)
        self.sections[name] = ReportSectionResult(name, required=required, input_rows=input_rows)

    def _reset(self, name: str) -> ReportSectionResult:
        """开始一次新尝试；不将旧数据、行数或错误理由混入本次状态。"""
        if name not in self.sections:
            self.plan(name)
        previous = self.sections[name]
        self.plan(name, required=previous.required, input_rows=previous.input_rows)
        return self.sections[name]

    def skip(self, name: str, reason: str) -> None:
        section = self._reset(name)
        section.status = "不适用"
        section.reason = reason

    def success(self, name: str, table: Optional[pd.DataFrame] = None) -> None:
        section = self._reset(name)
        section.status = "成功"
        if table is not None:
            self[name] = table
            section.output_rows = len(table)

    def failure(self, name: str, reason: str, *, required: bool = True) -> None:
        if name not in self.sections:
            self.plan(name, required=required)
        self.pop(name, None)
        self.sheet_mapping.pop(name, None)
        self.sections[name].output_rows = None
        self.sections[name].status = "失败"
        previous = self.sections[name].reason
        self.sections[name].reason = f"{previous}\n{reason}" if previous else reason

    def run(self, name: str, function: Callable[[], pd.DataFrame]) -> None:
        section = self._reset(name)
        start = perf_counter()
        try:
            table = function()
            if not isinstance(table, pd.DataFrame):
                raise TypeError("报告章节必须返回 DataFrame")
            self[name] = table
            section.status = "成功"
            section.output_rows = len(table)
        except Exception as exc:
            section.status = "失败"
            section.reason = f"{type(exc).__name__}: {exc}"
            logger.warning("报告章节失败 [%s]: %s", name, section.reason)
            if self.mode == "strict" and section.required:
                raise ReportGenerationError(f"必需报告章节失败: {name}，{section.reason}", self) from exc
        finally:
            section.elapsed_seconds = perf_counter() - start

    @property
    def complete(self) -> bool:
        return all(section.status in {"成功", "不适用"} for section in self.sections.values())

    def ensure_publishable(self) -> None:
        failed = [s.name for s in self.sections.values() if s.required and s.status in {"失败", "未执行"}]
        if self.mode == "strict" and failed:
            raise ReportGenerationError(f"报告包含未完成的必需章节: {', '.join(failed)}", self)

    def status_table(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"章节": s.name, "状态": s.status, "必需章节": s.required, "说明": s.reason,
             "输入行数": s.input_rows, "输出行数": s.output_rows, "耗时(秒)": s.elapsed_seconds}
            for s in self.sections.values()
        ])


__all__ = ["ReportResult", "ReportSectionResult", "ReportGenerationError"]
