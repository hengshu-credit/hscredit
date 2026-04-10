# -*- coding: utf-8 -*-
"""向后兼容: 旧路径 hscredit.core.models.logistic_regression → classical.logistic_regression."""

from .classical.logistic_regression import LogisticRegression  # noqa: F401

__all__ = ["LogisticRegression"]
