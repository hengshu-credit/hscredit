"""hsbin 分箱可视化操作适配器。"""

from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ...core.viz import bin_2d_plot, bin_overdues_plot, bin_plot, bin_trend_plot
from ..errors import SkillExecutionError
from ..registry import OperationSpec
from .binning import _data, _parameters


_PLOTS = OrderedDict(
    [
        ("bin_plot", bin_plot),
        ("bin_trend_plot", bin_trend_plot),
        ("bin_overdues_plot", bin_overdues_plot),
        ("bin_2d_plot", bin_2d_plot),
    ]
)


def _figure(value):
    if hasattr(value, "savefig"):
        return value
    if isinstance(value, tuple):
        for item in value:
            if hasattr(item, "savefig"):
                return item
    return None


def _plot_handler(function):
    def handler(context) -> dict:
        output_format = (context.request.output.format or "png").lower()
        if output_format not in {"png", "svg"}:
            raise SkillExecutionError(
                code="SCHEMA_INVALID",
                message="绘图输出格式只支持 png 或 svg",
                field="output.format",
            )
        params = _parameters(
            function,
            context.request.parameters,
            reserved=("data", "save", "bin_table", "binner"),
        )
        staged = context.artifacts.stage_path(f"{context.request.output.name}.{output_format}")

        if function is bin_overdues_plot and "bin_table" in context.request.inputs:
            table = context.resolver.resolve(context.request.inputs["bin_table"])
            if not isinstance(table, pd.DataFrame):
                raise SkillExecutionError(
                    code="SCHEMA_INVALID",
                    message="inputs.bin_table 必须解析为 DataFrame",
                    field="inputs.bin_table",
                )
            result = function(table, bin_table=table, save=str(staged), **params)
        elif function is bin_2d_plot and "binner" in context.request.inputs:
            binner = context.resolver.resolve(context.request.inputs["binner"])
            result = function(binner, binner=binner, save=str(staged), **params)
        else:
            result = function(_data(context), save=str(staged), **params)

        figure = _figure(result)
        if not staged.is_file():
            if figure is None:
                raise SkillExecutionError(
                    code="ARTIFACT_WRITE_FAILED",
                    message=f"绘图操作“{function.__name__}”没有生成图片",
                )
            figure.savefig(staged, format=output_format, bbox_inches="tight")
        context.artifacts.publish(
            staged,
            f"{context.request.output.name}.{output_format}",
            artifact_type="image",
        )
        if figure is not None:
            width, height = figure.get_size_inches()
            summary = {
                "format": output_format,
                "width_inches": float(width),
                "height_inches": float(height),
            }
            plt.close(figure)
        else:
            summary = {"format": output_format}
        return {"summary": summary}

    return handler


def register_visualization_operations(registry) -> None:
    """登记 hsbin 的四个分箱绘图操作。"""
    for name, function in _PLOTS.items():
        registry.register(
            OperationSpec("hsbin", name, _plot_handler(function), extras=("skills",))
        )
