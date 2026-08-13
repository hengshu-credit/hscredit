"""风险模型默认概率评分卡支持。"""

from typing import Any, Dict, Optional

import numpy as np

from ...exceptions import NotFittedError


class _ProbabilityScoreCardMixin:
    """为风险模型提供统一的标准概率评分卡。"""

    DEFAULT_SCORECARD_PARAMS: Dict[str, Any] = {
        "method": "standard",
        "pdo": 50,
        "base_score": 600,
        "lower": 0,
        "upper": 1000,
        "direction": "descending",
        "rate": 2,
        "decimal": 0,
        "clip": True,
    }

    def _initialize_scorecard_params(self, scorecard_params: Optional[Dict[str, Any]]) -> None:
        """校验评分卡参数，并以用户配置部分覆盖默认值。"""
        if scorecard_params is not None and not isinstance(scorecard_params, dict):
            raise TypeError("scorecard_params 必须是字典或 None")

        supplied = dict(scorecard_params or {})
        invalid = sorted(set(supplied) - set(self.DEFAULT_SCORECARD_PARAMS))
        if invalid:
            raise ValueError(f"不支持的评分卡参数: {invalid}")

        self.scorecard_params = scorecard_params
        self.scorecard_config_ = {**self.DEFAULT_SCORECARD_PARAMS, **supplied}

    def _fit_probability_scorecard(self, y: np.ndarray) -> None:
        """使用完整训练标签计算坏好比并拟合默认概率评分卡。"""
        # sklearn 的 set_params/clone 会直接更新构造参数；训练时重新合并以保持契约。
        self._initialize_scorecard_params(self.scorecard_params)
        labels = np.asarray(y).reshape(-1)
        unique_labels = set(np.unique(labels).tolist())
        if unique_labels != {0, 1}:
            raise ValueError("训练标签必须同时包含 0 和 1，且 1 表示坏样本")

        self.bad_rate_ = float(np.mean(labels == 1))
        self.base_odds_ = self.bad_rate_ / (1.0 - self.bad_rate_)

        # 延迟导入，避免评分卡包初始化期间与模型基类形成循环依赖。
        from .scorecard.model_scorecard import ProbabilityScoreCard

        config = dict(self.scorecard_config_)
        self.scorecard_ = ProbabilityScoreCard(
            model=None,
            base_odds=self.base_odds_,
            **config,
        ).fit(proba=np.asarray([self.bad_rate_], dtype=float))

    def _predict_probability_score(self, X: Any) -> np.ndarray:
        """把模型正类概率转换为已拟合的标准风险评分。"""
        if not hasattr(self, "scorecard_"):
            raise NotFittedError("模型评分卡尚未拟合，请先调用 fit()")
        proba = np.asarray(self.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba 必须返回至少两列概率，第二列表示坏样本概率")
        return self.scorecard_.predict_score(proba=proba[:, 1])
