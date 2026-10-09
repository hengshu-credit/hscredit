"""风险模型默认概率评分卡支持。"""

from typing import Any, Dict, Optional

import numpy as np

from ...exceptions import NotFittedError, SerializationError
from ._contracts import positive_probability, validate_labels, validate_sample_weight


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
        allowed = set(self.DEFAULT_SCORECARD_PARAMS) | self.SCORE_TRANSFORMER_OPTION_KEYS | {"base_bad_rate"}
        invalid = sorted(set(supplied) - allowed)
        if invalid:
            raise ValueError(f"不支持的评分卡参数: {invalid}")
        prior = supplied.get("base_bad_rate")
        if prior is not None and (isinstance(prior, (bool, np.bool_)) or not isinstance(prior, (int, float, np.integer, np.floating)) or not np.isfinite(prior) or not 0 < prior < 1):
            raise ValueError("base_bad_rate 必须为 (0, 1) 内的有限业务先验坏率")

        self.scorecard_params = scorecard_params
        self.scorecard_config_ = {**self.DEFAULT_SCORECARD_PARAMS, **supplied}

    @staticmethod
    def _validate_probability_scorecard_labels(y: np.ndarray) -> np.ndarray:
        """在原生模型训练前校验统一二分类标签。"""
        return validate_labels(y)

    def _positive_probability_values(self, proba: Any) -> np.ndarray:
        """按 classes_ 从概率结果提取类别 1。"""
        return positive_probability(proba, getattr(self, "classes_", [0, 1]), 1)

    def _positive_probability(self, X: Any) -> np.ndarray:
        """调用模型概率方法并提取类别 1。"""
        return self._positive_probability_values(self.predict_proba(X))

    def _fit_probability_scorecard(self, X: Any, y: np.ndarray, proba: Any = None, *, sample_weight=None) -> None:
        """使用完整训练概率拟合模型自己的概率评分转换器。"""
        # sklearn 的 set_params/clone 会直接更新构造参数；训练时重新合并以保持契约。
        self._initialize_scorecard_params(self.scorecard_params)
        labels = self._validate_probability_scorecard_labels(y)

        weight_type = getattr(self, "weight_type", "cost")
        if weight_type not in ("frequency", "cost"):
            raise ValueError("weight_type 必须为 frequency 或 cost")
        weights = validate_sample_weight(sample_weight, len(labels))
        explicit_prior = self.scorecard_config_.get("base_bad_rate")
        if explicit_prior is not None:
            self.bad_rate_ = float(explicit_prior)
            self.score_prior_source_ = "显式业务先验"
        elif weight_type == "frequency" and weights is not None:
            self.bad_rate_ = float(np.average(labels == 1, weights=weights))
            self.score_prior_source_ = "频数加权坏率"
        else:
            self.bad_rate_ = float(np.mean(labels == 1))
            self.score_prior_source_ = "原始样本坏率"
        if not 0 < self.bad_rate_ < 1:
            raise ValueError("评分刻度需要同时有正权重的好坏样本，或显式配置 base_bad_rate")
        self.base_odds_ = self.bad_rate_ / (1.0 - self.bad_rate_)

        # 延迟导入，避免评分卡包初始化期间与模型基类形成循环依赖。
        from .scorecard.model_scorecard import ProbabilityScoreCard

        config = dict(self.scorecard_config_)
        config.pop("base_bad_rate", None)
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
            "prior_source": getattr(self, "score_prior_source_", "旧版刻度"),
        }

    @staticmethod
    def _score_transformer_sidecar_path(path) -> str:
        return f"{path}.score_transformer.joblib"

    def _attach_score_transformer(self, transformer) -> None:
        """恢复直接属性，并让兼容 scorecard_ 共享同一转换器对象。"""
        from .scorecard.model_scorecard import ProbabilityScoreCard

        self.score_transformer_ = transformer
        config = dict(self.scorecard_config_)
        config.pop("base_bad_rate", None)
        self.scorecard_ = ProbabilityScoreCard(
            model=None,
            base_odds=self.base_odds_,
            **config,
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
            "prior_source": getattr(self, "score_prior_source_", "旧版刻度"),
            "scorecard_params": dict(self.scorecard_params or {}),
            "feature_names_in": list(feature_names) if feature_names is not None else [],
            "feature_names_known": getattr(self, "_feature_names_known_", True),
            "n_features_in": getattr(self, "n_features_in_", None),
            "classes": np.asarray(getattr(self, "classes_", [0, 1])).tolist(),
        }
        payload["model_state"] = {
            name: getattr(self, name)
            for name in (
                "training_history_",
                "training_summary_",
                "native_params_",
                "_evals_result",
                "_best_iteration",
                "_best_score",
                "scale_pos_weight_",
                "tuner",
                "feature_schema_",
                "fit_revision_",
            )
            if hasattr(self, name)
        }
        if hasattr(self, "get_params"):
            payload["model_params"] = self.get_params(deep=False)
        save_pickle(payload, sidecar, engine="cloudpickle")
        return sidecar

    def _load_score_transformer_sidecar(self, path, *, required: bool = False) -> bool:
        """从 sidecar 恢复转换器；旧原生模型可只恢复概率能力。"""
        from pathlib import Path
        from ...utils import load_pickle

        self.__dict__.pop("feature_schema_", None)
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
        self.score_prior_source_ = payload.get("prior_source", "旧版刻度")
        feature_names = payload.get("feature_names_in")
        if feature_names:
            self.feature_names_in_ = list(feature_names)
        if payload.get("n_features_in") is not None:
            self.n_features_in_ = int(payload["n_features_in"])
        self.classes_ = np.asarray(payload.get("classes", [0, 1]))
        self._feature_names_known_ = payload.get("feature_names_known", True)
        self._attach_score_transformer(payload["score_transformer"])
        if payload.get("model_params") and hasattr(self, "set_params"):
            self.set_params(**payload["model_params"])
        self.__dict__.update(payload.get("model_state", {}))
        return True

    def _restore_probability_scorecard(self, state: Dict[str, Any]) -> None:
        """从原生模型元数据恢复概率评分卡。"""
        if not isinstance(state, dict) or "bad_rate" not in state or "base_odds" not in state:
            raise ValueError("JSON模型元数据缺少概率评分卡状态，无法完整恢复模型")

        self._initialize_scorecard_params(state.get("scorecard_params"))
        self.bad_rate_ = float(state["bad_rate"])
        self.base_odds_ = float(state["base_odds"])
        self.score_prior_source_ = state.get("prior_source", "旧版刻度")

        from .scorecard.model_scorecard import ProbabilityScoreCard

        scorecard = ProbabilityScoreCard(
            model=None,
            base_odds=self.base_odds_,
            **{key: value for key, value in self.scorecard_config_.items() if key != "base_bad_rate"},
        ).fit(proba=np.asarray([self.bad_rate_], dtype=float))
        self._attach_score_transformer(scorecard.transformer_)

    def _predict_probability_score(self, X: Any) -> np.ndarray:
        """把模型正类概率转换为已拟合的标准风险评分。"""
        if not hasattr(self, "score_transformer_"):
            raise NotFittedError("模型评分卡尚未拟合，请先调用 fit()")
        return self.score_transformer_.predict(self._positive_probability(X))
