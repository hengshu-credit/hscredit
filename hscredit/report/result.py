"""报告章节状态与发布清单，不依赖具体计算或渲染器。"""

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReportSectionResult:
    """一个报告章节的执行状态。

    **参数**

    :param name: 章节名称，与 ``ReportResult.sections`` 的键及结果表名称对应。
    :param status: 执行状态，默认 ``'未执行'``；可取 ``'成功'``、``'不适用'``、``'失败'``。
    :param required: 是否为必需章节，默认 ``True``；严格模式下失败或未执行会阻止发布。
    :param reason: 失败详情或不适用原因，默认空字符串。
    :param input_rows: 输入样本行数，默认 ``None`` 表示未记录，不等于零行。
    :param output_rows: 输出表行数，默认 ``None``；不代表输入样本量。
    :param elapsed_seconds: 执行耗时，单位秒，默认 ``0.0``；由 ``ReportResult.run`` 计时。
    :param sampling: 抽样口径等补充信息，每个实例默认拥有独立的空字典；不会自动执行抽样。

    **属性**

    上述同名字段均可读取和更新；通常由 :class:`ReportResult` 的章节方法维护。

    **参考样例**

    >>> section = ReportSectionResult('分箱统计', input_rows=100)
    >>> section.status
    '未执行'
    """

    name: str
    status: str = "未执行"
    required: bool = True
    reason: str = ""
    input_rows: Optional[int] = None
    output_rows: Optional[int] = None
    elapsed_seconds: float = 0.0
    sampling: Dict[str, Any] = field(default_factory=dict)


class ReportGenerationError(ValueError):
    """报告生成或发布未满足严格模式要求时抛出的异常。

    :param message: 中文错误详情，可通过 ``str(error)`` 读取。
    :param result: 本次报告结果；保留已经完成的表和全部章节状态，供排查或重试。
    :ivar result: 传入的 :class:`ReportResult` 原对象，可调用 ``status_table()`` 查看失败章节。
    """

    def __init__(self, message: str, result: "ReportResult"):
        super().__init__(message)
        self.result = result


class ReportResult(dict):
    """兼容 ``dict[str, DataFrame]`` 的结构化报告。

    **参数**

    :param mode: 执行模式，默认 ``'best_effort'``。``'strict'`` 在 ``run`` 的必需章节
        失败时抛出 :class:`ReportGenerationError`，并在发布检查时拒绝未完成的必需章节；
        ``'best_effort'`` 保留成功表并记录失败。直接调用 ``failure`` 仅记录状态。
    :param metadata: 本次运行的目标、字段及样本口径，默认 ``None``。
        初始化时浅拷贝为独立字典；建议仅存口径描述，不传入逐行个人数据。
    :raises ValueError: ``mode`` 不是上述两种模式。

    **属性**

    :ivar mode: 当前执行模式。
    :ivar metadata: 报告元数据字典，默认空字典。
    :ivar sections: ``{章节名: ReportSectionResult}``，包含成功、不适用、失败和未执行
        章节，按登记顺序保存。与仅存储成功结果表的字典主体不同。
    :ivar artifacts: 发布产物清单，默认空列表；导出器成功发布后追加信息，例如
        ``{'类型': 'Excel', '路径': '报告.xlsx', '完整': True}``。本对象自身不写文件。
    :ivar sheet_mapping: 章节名到工作表名的映射，默认空字典，供导出器维护；
        重新计划、重试或标记失败时清除对应章节的旧映射。

    通过 ``result[章节名]`` 读取成功的 DataFrame；继承的字典赋值不会自动更新章节状态，
    应使用 ``success`` 或 ``run`` 登记结果。``complete`` 检查所有章节是否完成，
    ``ensure_publishable`` 则按模式检查必需章节是否允许发布。

    **参考样例**

    >>> import pandas as pd
    >>> result = ReportResult(mode='strict', metadata={'目标': 'FPD'})
    >>> result.plan('样本概览', input_rows=2)
    >>> result.run('样本概览', lambda: pd.DataFrame({'样本总数': [2]}))
    >>> result.ensure_publishable()
    >>> result['样本概览']['样本总数'].iloc[0]
    2
    >>> result.complete
    True
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
        """登记待执行章节，并清除同名章节的旧结果、状态和工作表映射。

        :param name: 章节名称，也是成功结果表的字典键。
        :param required: 是否为必需章节，默认 ``True``，用于严格模式的失败和发布检查。
        :param input_rows: 输入样本行数，默认 ``None`` 表示未记录。
        :return: ``None``；新章节状态为 ``'未执行'``。
        """
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
        """将章节标记为不适用，清除旧表并保留原登记的必需标记和输入行数。

        :param name: 章节名称；未登记时按默认必需章节自动登记。
        :param reason: 当前数据或模型不适用该章节的原因，不应代替执行失败的记录。
        :return: ``None``；不适用章节视为已完成，不阻止严格模式发布。
        """
        section = self._reset(name)
        section.status = "不适用"
        section.reason = reason

    def success(self, name: str, table: Optional[pd.DataFrame] = None) -> None:
        """登记章节成功结果，重置旧的错误信息、输出行数和耗时。

        :param name: 章节名称；未登记时自动登记，已有章节保留必需标记和输入行数。
        :param table: 成功结果表，默认 ``None``，用于仅记录渲染或发布等无表步骤。
            传表时保存为 ``self[name]``，并将输出行数记为 ``len(table)``。
        :return: ``None``；此方法不负责计时，执行函数并计时请使用 ``run``。
        """
        section = self._reset(name)
        section.status = "成功"
        if table is not None:
            self[name] = table
            section.output_rows = len(table)

    def failure(self, name: str, reason: str, *, required: bool = True) -> None:
        """记录章节失败，移除旧结果表和工作表映射，连续失败的原因按换行追加。

        :param name: 章节名称。
        :param reason: 错误类型、详情等诊断信息。
        :param required: 默认 ``True``，仅在首次登记章节时使用；不会覆盖已登记的必需标记。
        :return: ``None``；即使严格模式也只记录失败，发布前调用 ``ensure_publishable`` 检查。
        """
        if name not in self.sections:
            self.plan(name, required=required)
        self.pop(name, None)
        self.sheet_mapping.pop(name, None)
        self.sections[name].output_rows = None
        self.sections[name].status = "失败"
        previous = self.sections[name].reason
        self.sections[name].reason = f"{previous}\n{reason}" if previous else reason

    def run(self, name: str, function: Callable[[], pd.DataFrame]) -> None:
        """执行一个章节，保存结果表、行数、耗时及失败原因。

        :param name: 章节名称；未登记时自动登记为必需章节。重试前清除该章节旧结果。
        :param function: 无参数可调用对象，必须返回 DataFrame；需要传参时使用闭包或
            ``functools.partial``。返回其他类型按章节失败处理。
        :return: ``None``；成功表从 ``self[name]`` 读取，执行状态从 ``sections[name]`` 读取。
        :raises ReportGenerationError: 严格模式下必需章节执行失败；异常的 ``result`` 指向当前对象。
            尽力模式或非必需章节失败仅记录状态并输出日志。
        """
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
        """是否所有已登记章节均为成功或不适用。

        :return: 布尔值；非必需章节失败也返回 ``False``，没有登记章节时返回 ``True``。
            此属性不检查文件是否存在，也不等同于严格模式的发布检查。
        """
        return all(section.status in {"成功", "不适用"} for section in self.sections.values())

    def ensure_publishable(self) -> None:
        """按当前模式检查必需章节，供导出器在发布文件前调用。

        :return: ``None``；尽力模式不阻止发布，严格模式允许不适用或非必需章节失败。
        :raises ReportGenerationError: 严格模式下存在失败或未执行的必需章节。
            异常保留当前结果对象，本方法自身不保存文件。
        """
        failed = [s.name for s in self.sections.values() if s.required and s.status in {"失败", "未执行"}]
        if self.mode == "strict" and failed:
            raise ReportGenerationError(f"报告包含未完成的必需章节: {', '.join(failed)}", self)

    def status_table(self) -> pd.DataFrame:
        """按登记顺序生成章节执行状态表。

        :return: DataFrame，包含 ``章节``、``状态``、``必需章节``、``说明``、``输入行数``、
            ``输出行数``、``耗时(秒)`` 七列。未记录的行数为空值，抽样信息保留在
            ``sections[name].sampling`` 中。尚未登记章节时返回无行、无列的空表。
        """
        return pd.DataFrame(
            [
                {
                    "章节": s.name,
                    "状态": s.status,
                    "必需章节": s.required,
                    "说明": s.reason,
                    "输入行数": s.input_rows,
                    "输出行数": s.output_rows,
                    "耗时(秒)": s.elapsed_seconds,
                }
                for s in self.sections.values()
            ]
        )


__all__ = ["ReportResult", "ReportSectionResult", "ReportGenerationError"]
