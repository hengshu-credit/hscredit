"""风险模型默认概率评分卡支持。"""

from typing import Any, Dict, Optional

import numpy as np

from ...exceptions import NotFittedError, SerializationError


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
    SCORE_TRANSFORMER_OPTION_KEYS = {"n_quantiles", "lmbda", "shift"}

    def _initialize_scorecard_params(self, scorecard_params: Optional[Dict[str, Any]]) -> None:
        """校验评分卡参数，并以用户配置部分覆盖默认值。"""
        if scorecard_params is not None and not isinstance(scorecard_params, dict):
            raise TypeError("scorecard_params 必须是字典或 None")

        supplied = dict(scorecard_params or {})
        allowed = set(self.DEFAULT_SCORECARD_PARAMS) | self.SCORE_TRANSFORMER_OPTION_KEYS
        invalid = sorted(set(supplied) - allowed)
        if invalid:
            raise ValueError(f"不支持的评分卡参数: {invalid}")

        self.scorecard_params = scorecard_params
        self.scorecard_config_ = {**self.DEFAULT_SCORECARD_PARAMS, **supplied}

    @staticmethod
    def _validate_probability_scorecard_labels(y: np.ndarray) -> np.ndarray:
        """在原生模型训练前校验统一二分类标签。"""
        labels = np.asarray(y).reshape(-1)
        unique_labels = set(np.unique(labels).tolist())
        if unique_labels != {0, 1}:
            raise ValueError("训练标签必须同时包含 0 和 1，且 1 表示坏样本")
        return labels

    def _positive_probability_values(self, proba: Any) -> np.ndarray:
        """按 classes_ 从概率结果提取类别 1。"""
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("predict_proba 必须返回至少两列概率，类别 1 表示坏样本")
        positive_index = 1
        classes = getattr(self, "classes_", None)
        if classes is not None:
            positive = np.flatnonzero(np.asarray(classes) == 1)
            if len(positive) == 1:
                positive_index = int(positive[0])
        return proba[:, positive_index]

    def _positive_probability(self, X: Any) -> np.ndarray:
        """调用模型概率方法并提取类别 1。"""
        return self._positive_probability_values(self.predict_proba(X))

    def _fit_probability_scorecard(self, X: Any, y: np.ndarray, proba: Any = None) -> None:
        """使用完整训练概率拟合模型自己的概率评分转换器。"""
        # sklearn 的 set_params/clone 会直接更新构造参数；训练时重新合并以保持契约。
        self._initialize_scorecard_params(self.scorecard_params)
        labels = self._validate_probability_scorecard_labels(y)

        self.bad_rate_ = float(np.mean(labels == 1))
        self.base_odds_ = self.bad_rate_ / (1.0 - self.bad_rate_)

        # 延迟导入，避免评分卡包初始化期间与模型基类形成循环依赖。
        from .scorecard.model_scorecard import ProbabilityScoreCard

        config = dict(self.scorecard_config_)
        train_probability = (
            self._positive_probability(X)
            if proba is None
            else self._positive_probability_values(proba)
        )
        self.scorecard_ = ProbabilityScoreCard(
            model=None,
            base_odds=self.base_odds_,
            **config,
        ).fit(proba=train_probability)
        self.score_transformer_ = self.scorecard_.transformer_

    def _probability_scorecard_state(self) -> Dict[str, Any]:
        """返回原生模型序列化时需要保留的评分刻度状态。"""
        if not hasattr(self, "scorecard_"):
            raise NotFittedError("模型评分卡尚未拟合，无法保存完整模型")
        return {
            "bad_rate": float(self.bad_rate_),
            "base_odds": float(self.base_odds_),
            "scorecard_params": dict(self.scorecard_params or {}),
        }

    @staticmethod
    def _score_transformer_sidecar_path(path) -> str:
        return f"{path}.score_transformer.joblib"

    def _attach_score_transformer(self, transformer) -> None:
        """恢复直接属性，并让兼容 scorecard_ 共享同一转换器对象。"""
        from .scorecard.model_scorecard import ProbabilityScoreCard

        self.score_transformer_ = transformer
        self.scorecard_ = ProbabilityScoreCard(
            model=None,
            base_odds=self.base_odds_,
            **dict(self.scorecard_config_),
        )
        self.scorecard_.model_ = None
        self.scorecard_.transformer_ = transformer
        self.scorecard_.A_ = getattr(transformer.transformer_, "A_", None)
        self.scorecard_.B_ = getattr(transformer.transformer_, "B_", None)
        self.scorecard_.direction_ = transformer.direction_
        self.scorecard_._is_fitted = True

    def _save_score_transformer_sidecar(self, path) -> str:
        """保存原生模型之外的完整概率评分转换器状态。"""
        if not hasattr(self, "score_transformer_"):
            raise NotFittedError("模型评分转换器尚未拟合，无法保存")
        from ...utils import save_pickle

        sidecar = self._score_transformer_sidecar_path(path)
        feature_names = getattr(self, "feature_names_in_", None)
        payload = {
            "score_transformer": self.score_transformer_,
            "bad_rate": float(self.bad_rate_),
            "base_odds": float(self.base_odds_),
            "scorecard_params": dict(self.scorecard_params or {}),
            "feature_names_in": list(feature_names) if feature_names is not None else [],
            "n_features_in": getattr(self, "n_features_in_", None),
            "classes": np.asarray(getattr(self, "classes_", [0, 1])).tolist(),
        }
        save_pickle(payload, sidecar, engine="joblib")
        return sidecar

    def _load_score_transformer_sidecar(self, path, *, required: bool = False) -> bool:
        """从 sidecar 恢复转换器；旧原生模型可只恢复概率能力。"""
        from pathlib import Path
        from ...utils import load_pickle

        sidecar = Path(self._score_transformer_sidecar_path(path))
        if not sidecar.exists():
            if required:
                raise SerializationError(f"评分转换器制品不存在: {sidecar}")
            self.__dict__.pop("score_transformer_", None)
            self.__dict__.pop("scorecard_", None)
            return False
        payload = load_pickle(sidecar, engine="joblib")
        if not isinstance(payload, dict) or "score_transformer" not in payload:
            raise SerializationError(f"评分转换器制品格式无效: {sidecar}")
        self._initialize_scorecard_params(payload.get("scorecard_params"))
        self.bad_rate_ = float(payload["bad_rate"])
        self.base_odds_ = float(payload["base_odds"])
        feature_names = payload.get("feature_names_in")
        if feature_names:
            self.feature_names_in_ = list(feature_names)
        if payload.get("n_features_in") is not None:
            self.n_features_in_ = int(payload["n_features_in"])
        self.classes_ = np.asarray(payload.get("classes", [0, 1]))
        self._attach_score_transformer(payload["score_transformer"])
        return True

    def _restore_probability_scorecard(self, state: Dict[str, Any]) -> None:
        """从原生模型元数据恢复概率评分卡。"""
        if not isinstance(state, dict) or "bad_rate" not in state or "base_odds" not in state:
            raise ValueError("JSON模型元数据缺少概率评分卡状态，无法完整恢复模型")

        self._initialize_scorecard_params(state.get("scorecard_params"))
        self.bad_rate_ = float(state["bad_rate"])
        self.base_odds_ = float(state["base_odds"])

        from .scorecard.model_scorecard import ProbabilityScoreCard

        scorecard = ProbabilityScoreCard(
            model=None,
            base_odds=self.base_odds_,
            **dict(self.scorecard_config_),
        ).fit(proba=np.asarray([self.bad_rate_], dtype=float))
        self._attach_score_transformer(scorecard.transformer_)

    def _predict_probability_score(self, X: Any) -> np.ndarray:
        """把模型正类概率转换为已拟合的标准风险评分。"""
        if not hasattr(self, "score_transformer_"):
            raise NotFittedError("模型评分卡尚未拟合，请先调用 fit()")
        return self.score_transformer_.predict(self._positive_probability(X))
