"""Agent Skills 内置操作登记。"""

from .binning import register_binning_operations
from .reports import register_report_operations
from .visualization import register_visualization_operations


def register_operations(registry) -> None:
    """登记当前实现阶段的所有操作。"""
    register_binning_operations(registry)
    register_visualization_operations(registry)
    register_report_operations(registry)
