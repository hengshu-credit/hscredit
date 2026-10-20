"""Agent Skills 内置操作登记。"""

from .binning import register_binning_operations
from .models import register_model_operations
from .reports import register_report_operations
from .visualization import register_visualization_operations
from .extended import (
    register_excel_operations,
    register_hscredit_operations,
    register_rule_operations,
    register_selector_operations,
    register_viz_operations,
)


def register_operations(registry) -> None:
    """登记当前实现阶段的所有操作。"""
    register_binning_operations(registry)
    register_visualization_operations(registry)
    register_report_operations(registry)
    register_model_operations(registry)
    register_rule_operations(registry)
    register_selector_operations(registry)
    register_viz_operations(registry)
    register_excel_operations(registry)
    register_hscredit_operations(registry)
