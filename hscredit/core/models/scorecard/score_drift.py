"""评分漂移校准模块.

提供多种评分漂移检测和校准方法，用于解决生产环境中
模型评分分布与训练时不一致的问题。

**核心功能**

- 线性漂移校准: 通过offset和scale调整对齐分布
- 分位数对齐: 基于分位数的分布映射校准
- 分箱重校准: 基于分箱的坏样本率重新校准
- PSI监控: 计算群体稳定性指数监控漂移
- 评分对齐: 将生产评分对齐到参考分布

**漂移类型**

在生产环境中，常见的评分漂移类型包括:

1. **整体偏移 (Shift)**: 评分整体偏高或偏低
   - 原因: 客群质量变化、经济环境变化
   - 处理: 线性offset调整

2. **缩放变化 (Scale)**: 评分分布范围变化
   - 原因: 特征分布变化、模型老化
   - 处理: 线性scale调整

3. **分布变形 (Shape)**: 评分分布形状变化
   - 原因: 概念漂移、特征关系变化
   - 处理: 分位数映射或分箱重校准

**校准方法对比**

| 方法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| Linear | 整体偏移/缩放 | 简单、可解释 | 无法处理形状变化 |
| Quantile | 分布变形 | 保持排名、灵活 | 需要大量数据 |
| Binning | 坏样本率变化 | 业务可解释 | 需要标签、粒度粗 |

**依赖**
- numpy
- pandas
- scipy

**示例**

>>> from hscredit.core.models import XGBoostRiskModel
>>> from hscredit.core.models.score_drift import ScoreDriftCalibrator
>>>
>>> # 训练基础模型
>>> model = XGBoostRiskModel()
>>> model.fit(X_train, y_train)
>>>
>>> # 创建漂移校准器
>>> calibrator = ScoreDriftCalibrator(method='linear')
>>>
>>> # 拟合校准器(使用生产环境数据)
>>> calibrator.fit(model, X_production, y_production)
>>>
>>> # 预测校准后的评分
>>> scores_calibrated = calibrator.predict_score(X_test)
>>>
>>> # 检测漂移
>>> drift_report = calibrator.detect_drift(X_reference, X_production)
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

try:
    from ...metrics.stability import psi
    PSI_AVAILABLE = True
except ImportError:
    PSI_AVAILABLE = False


class BaseDriftCalibrator(BaseEstimator, ABC):
    """评分漂移校准器基类.

    所有漂移校准器的抽象基类，定义统一接口。

    **参数**

    :param reference_scores: 参考评分(训练时的评分分布)，可选
    :param method: 校准方法
    :param clip_bounds: 是否限制评分在参考分布的边界内，默认True
    """

    def __init__(
        self,
        reference_scores: Optional[Union[np.ndarray, pd.Series]] = None,
        method: str = 'linear',
        clip_bounds: bool = True
    ):
        self.reference_scores = reference_scores
        self.method = method
        self.clip_bounds = clip_bounds
        self._is_fitted = False

    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target: str = 'target',
        require_y: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """准备数据，支持两种传参风格.

        :param X: 特征矩阵或包含target的DataFrame
        :param y: 目标变量，可选
        :param target: 目标列名
        :param require_y: 是否必须有y，默认False
        :return: (X, y) 处理后的数据
        """
        # scorecardpipeline风格：从X中提取target
        if y is None:
            if isinstance(X, pd.DataFrame) and target in X.columns:
                y = X[target].values
                X = X.drop(columns=[target])
            elif require_y:
                raise ValueError(f"y为None时，X必须是包含'{target}'列的DataFrame")
        else:
            if isinstance(y, pd.Series):
                y = y.values

        if isinstance(X, pd.DataFrame):
            X = X.values

        return X, y

    def _get_reference_scores(
        self,
        model: Any,
        X_ref: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> np.ndarray:
        """获取参考评分.

        :param model: 模型
        :param X_ref: 参考数据，可选
        :return: 参考评分
        """
        if self.reference_scores is not None:
            return np.asarray(self.reference_scores)

        if X_ref is not None and model is not None:
            return model.predict_proba(X_ref)[:, 1]

        raise ValueError("必须提供reference_scores或X_ref")

    @abstractmethod
    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_reference: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> 'BaseDriftCalibrator':
        """拟合漂移校准器.

        :param model: 已训练的模型
        :param X: 当前生产环境数据
        :param y: 目标变量，可选
        :param X_reference: 参考数据(训练数据)，可选
        :return: self
        """
        pass

    @abstractmethod
    def calibrate(
        self,
        scores: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """校准评分.

        :param scores: 原始评分
        :return: 校准后的评分
        """
        pass

    def predict_score(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        scores: Optional[Union[np.ndarray, pd.Series]] = None,
        proba: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> np.ndarray:
        """预测校准后的评分.

        可通过传入X、scores或proba之一来获取评分。

        :param X: 特征矩阵
        :param scores: 直接传入评分
        :param proba: 直接传入概率
        :return: 校准后的评分数组

        **示例**

        >>> # 通过特征矩阵预测
        >>> scores_calib = calibrator.predict_score(X_test)

        >>> # 通过概率直接校准
        >>> proba = model.predict_proba(X_test)[:, 1]
        >>> scores_calib = calibrator.predict_score(proba=proba)
        """
        check_is_fitted(self)

        if scores is not None:
            scores = np.asarray(scores)
        elif proba is not None:
            scores = np.asarray(proba)
        elif X is not None:
            if not hasattr(self, 'model_'):
                raise ValueError("未找到模型，请先调用fit()")
            scores = self.model_.predict_proba(X)[:, 1]
        else:
            raise ValueError("必须提供X、scores或proba参数之一")

        # 校准评分
        scores_calibrated = self.calibrate(scores)

        return scores_calibrated

    def detect_drift(
        self,
        X_reference: Union[np.ndarray, pd.DataFrame],
        X_current: Union[np.ndarray, pd.DataFrame],
        metric: str = 'psi',
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """检测评分漂移.

        :param X_reference: 参考数据
        :param X_current: 当前数据
        :param metric: 漂移检测指标，默认'psi'
            - 'psi': Population Stability Index
            - 'ks': Kolmogorov-Smirnov统计量
            - 'wasserstein': Wasserstein距离
        :param threshold: 漂移阈值，默认None(使用推荐值)
        :return: 漂移检测报告
        """
        if not hasattr(self, 'model_'):
            raise ValueError("未找到模型，请先调用fit()")

        # 获取评分
        scores_ref = self.model_.predict_proba(X_reference)[:, 1]
        scores_cur = self.model_.predict_proba(X_current)[:, 1]

        result = {
            'metric': metric,
            'threshold': threshold,
            'drift_detected': False,
            'statistics': {}
        }

        if metric == 'psi':
            psi_value = self._calculate_psi(scores_ref, scores_cur)
            result['statistics']['psi'] = psi_value
            result['threshold'] = threshold or 0.25
            result['drift_detected'] = psi_value > result['threshold']

        elif metric == 'ks':
            ks_stat, p_value = stats.ks_2samp(scores_ref, scores_cur)
            result['statistics']['ks_statistic'] = ks_stat
            result['statistics']['p_value'] = p_value
            result['threshold'] = threshold or 0.1
            result['drift_detected'] = ks_stat > result['threshold']

        elif metric == 'wasserstein':
            from scipy.stats import wasserstein_distance
            w_dist = wasserstein_distance(scores_ref, scores_cur)
            result['statistics']['wasserstein_distance'] = w_dist
            result['threshold'] = threshold or 0.1
            result['drift_detected'] = w_dist > result['threshold']

        else:
            raise ValueError(f"不支持的漂移检测指标: {metric}")

        return result

    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """计算PSI (Population Stability Index).

        :param expected: 参考分布
        :param actual: 实际分布
        :param n_bins: 分箱数
        :return: PSI值
        """
        # 创建分箱
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # 计算频率
        expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)

        # 避免除零
        expected_percents = np.clip(expected_percents, 1e-10, 1)
        actual_percents = np.clip(actual_percents, 1e-10, 1)

        # 计算PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

        return float(psi)


class LinearDriftCalibrator(BaseDriftCalibrator):
    """线性漂移校准器.

    通过线性变换对齐评分分布:
        calibrated_score = (score - current_mean) / current_std * ref_std + ref_mean

    适用于整体偏移或缩放变化的情况。

    **参数**

    :param reference_scores: 参考评分，可选
    :param target_mean: 目标均值，默认None(使用参考分布均值)
    :param target_std: 目标标准差，默认None(使用参考分布标准差)
    :param clip_bounds: 是否限制边界，默认True

    **示例**

    >>> calibrator = LinearDriftCalibrator()
    >>> calibrator.fit(model, X_production, X_reference=X_train)
    >>> scores_calib = calibrator.predict_score(X_test)
    """

    def __init__(
        self,
        reference_scores: Optional[Union[np.ndarray, pd.Series]] = None,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        clip_bounds: bool = True
    ):
        super().__init__(reference_scores, 'linear', clip_bounds)
        self.target_mean = target_mean
        self.target_std = target_std

    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_reference: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        target: str = 'target',
        **kwargs
    ) -> 'LinearDriftCalibrator':
        """拟合线性漂移校准器.

        :param model: 已训练的模型
        :param X: 当前生产环境数据
        :param y: 目标变量，可选
        :param X_reference: 参考数据，可选
        :param target: 目标列名
        :return: self
        """
        # 准备数据(y可选)
        X_prep, _ = self._prepare_data(X, y, target, require_y=False)

        # 保存模型
        if not hasattr(model, 'predict_proba'):
            raise ValueError("模型必须有predict_proba方法")
        self.model_ = model

        # 获取参考评分
        if X_reference is not None:
            scores_ref = model.predict_proba(X_reference)[:, 1]
        elif self.reference_scores is not None:
            scores_ref = np.asarray(self.reference_scores)
        else:
            raise ValueError("必须提供X_reference或reference_scores")

        # 获取当前评分
        scores_cur = model.predict_proba(X_prep)[:, 1]

        # 计算统计量
        self.ref_mean_ = np.mean(scores_ref)
        self.ref_std_ = np.std(scores_ref)
        self.cur_mean_ = np.mean(scores_cur)
        self.cur_std_ = np.std(scores_cur)

        # 目标统计量
        self.target_mean_ = self.target_mean if self.target_mean is not None else self.ref_mean_
        self.target_std_ = self.target_std if self.target_std is not None else self.ref_std_

        # 计算变换参数
        if self.cur_std_ > 0:
            self.scale_ = self.target_std_ / self.cur_std_
        else:
            self.scale_ = 1.0
            warnings.warn("当前标准差为0，scale设置为1.0")

        self.offset_ = self.target_mean_ - self.cur_mean_ * self.scale_

        # 边界
        if self.clip_bounds:
            self.lower_bound_ = np.min(scores_ref)
            self.upper_bound_ = np.max(scores_ref)

        self._is_fitted = True

        if self.clip_bounds:
            print(f"线性校准参数: scale={self.scale_:.4f}, offset={self.offset_:.4f}")
            print(f"参考分布: mean={self.ref_mean_:.4f}, std={self.ref_std_:.4f}")
            print(f"当前分布: mean={self.cur_mean_:.4f}, std={self.cur_std_:.4f}")

        return self

    def calibrate(
        self,
        scores: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """校准评分.

        :param scores: 原始评分
        :return: 校准后的评分
        """
        check_is_fitted(self)
        scores = np.asarray(scores)

        # 线性变换
        scores_calibrated = scores * self.scale_ + self.offset_

        # 边界限制
        if self.clip_bounds:
            scores_calibrated = np.clip(
                scores_calibrated,
                self.lower_bound_,
                self.upper_bound_
            )

        return scores_calibrated


class QuantileAligner(BaseDriftCalibrator):
    """分位数对齐校准器.

    通过分位数映射将当前分布对齐到参考分布:
        calibrated_score = F_ref^{-1}(F_cur(score))

    适用于分布形状发生变化的情况。

    **参数**

    :param reference_scores: 参考评分，可选
    :param n_quantiles: 分位数数量，默认100
    :param interpolation: 插值方法，默认'linear'
    :param clip_bounds: 是否限制边界，默认True

    **示例**

    >>> calibrator = QuantileAligner(n_quantiles=100)
    >>> calibrator.fit(model, X_production, X_reference=X_train)
    >>> scores_calib = calibrator.predict_score(X_test)
    """

    def __init__(
        self,
        reference_scores: Optional[Union[np.ndarray, pd.Series]] = None,
        n_quantiles: int = 100,
        interpolation: str = 'linear',
        clip_bounds: bool = True
    ):
        super().__init__(reference_scores, 'quantile', clip_bounds)
        self.n_quantiles = n_quantiles
        self.interpolation = interpolation

    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_reference: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        target: str = 'target',
        **kwargs
    ) -> 'QuantileAligner':
        """拟合分位数对齐器.

        :param model: 已训练的模型
        :param X: 当前生产环境数据
        :param y: 目标变量，可选
        :param X_reference: 参考数据，可选
        :param target: 目标列名
        :return: self
        """
        # 准备数据(y可选)
        X_prep, _ = self._prepare_data(X, y, target, require_y=False)

        # 保存模型
        if not hasattr(model, 'predict_proba'):
            raise ValueError("模型必须有predict_proba方法")
        self.model_ = model

        # 获取参考评分
        if X_reference is not None:
            scores_ref = model.predict_proba(X_reference)[:, 1]
        elif self.reference_scores is not None:
            scores_ref = np.asarray(self.reference_scores)
        else:
            raise ValueError("必须提供X_reference或reference_scores")

        # 获取当前评分
        scores_cur = model.predict_proba(X_prep)[:, 1]

        # 计算分位数
        quantiles = np.linspace(0, 1, self.n_quantiles)

        self.ref_quantiles_ = np.quantile(scores_ref, quantiles)
        self.cur_quantiles_ = np.quantile(scores_cur, quantiles)

        # 创建插值函数
        self._transform_func = interp1d(
            self.cur_quantiles_,
            self.ref_quantiles_,
            kind=self.interpolation,
            bounds_error=False,
            fill_value=(self.ref_quantiles_[0], self.ref_quantiles_[-1])
        )

        # 边界
        if self.clip_bounds:
            self.lower_bound_ = np.min(scores_ref)
            self.upper_bound_ = np.max(scores_ref)

        self._is_fitted = True

        return self

    def calibrate(
        self,
        scores: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """校准评分.

        :param scores: 原始评分
        :return: 校准后的评分
        """
        check_is_fitted(self)
        scores = np.asarray(scores)

        # 分位数映射
        scores_calibrated = self._transform_func(scores)

        # 边界限制
        if self.clip_bounds:
            scores_calibrated = np.clip(
                scores_calibrated,
                self.lower_bound_,
                self.upper_bound_
            )

        return scores_calibrated


class BinningRecalibrator(BaseDriftCalibrator):
    """分箱重校准器.

    基于分箱的坏样本率重新校准评分。
    将当前评分分箱后，根据参考分布的坏样本率重新计算分数。

    适用于坏样本率发生系统性变化的情况。

    **参数**

    :param reference_scores: 参考评分，可选
    :param reference_y: 参考标签，可选
    :param n_bins: 分箱数，默认10
    :param binning_strategy: 分箱策略，默认'quantile'
        - 'quantile': 等频分箱
        - 'uniform': 等宽分箱
    :param clip_bounds: 是否限制边界，默认True

    **示例**

    >>> calibrator = BinningRecalibrator(n_bins=10)
    >>> calibrator.fit(model, X_production, y_production, X_reference=X_train, y_reference=y_train)
    >>> scores_calib = calibrator.predict_score(X_test)
    """

    def __init__(
        self,
        reference_scores: Optional[Union[np.ndarray, pd.Series]] = None,
        reference_y: Optional[Union[np.ndarray, pd.Series]] = None,
        n_bins: int = 10,
        binning_strategy: str = 'quantile',
        clip_bounds: bool = True
    ):
        super().__init__(reference_scores, 'binning', clip_bounds)
        self.reference_y = reference_y
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy

    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_reference: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_reference: Optional[Union[np.ndarray, pd.Series]] = None,
        target: str = 'target',
        **kwargs
    ) -> 'BinningRecalibrator':
        """拟合分箱重校准器.

        :param model: 已训练的模型
        :param X: 当前生产环境数据
        :param y: 当前标签
        :param X_reference: 参考数据
        :param y_reference: 参考标签
        :param target: 目标列名
        :return: self
        """
        # 准备数据(需要y)
        X_prep, y_prep = self._prepare_data(X, y, target, require_y=True)

        # 保存模型
        if not hasattr(model, 'predict_proba'):
            raise ValueError("模型必须有predict_proba方法")
        self.model_ = model

        # 获取参考评分和标签
        if X_reference is not None and y_reference is not None:
            X_ref, y_ref = self._prepare_data(X_reference, y_reference, target)
            scores_ref = model.predict_proba(X_ref)[:, 1]
        elif self.reference_scores is not None and self.reference_y is not None:
            scores_ref = np.asarray(self.reference_scores)
            y_ref = np.asarray(self.reference_y)
        else:
            raise ValueError("必须提供X_reference和y_reference，或reference_scores和reference_y")

        # 获取当前评分
        scores_cur = model.predict_proba(X_prep)[:, 1]

        # 创建分箱边界(基于参考分布)
        if self.binning_strategy == 'quantile':
            self.bin_edges_ = np.quantile(
                scores_ref,
                np.linspace(0, 1, self.n_bins + 1)
            )
        else:  # uniform
            self.bin_edges_ = np.linspace(
                scores_ref.min(),
                scores_ref.max(),
                self.n_bins + 1
            )

        # 确保边界唯一
        self.bin_edges_ = np.unique(self.bin_edges_)
        self.n_bins_actual_ = len(self.bin_edges_) - 1

        # 计算参考分布每个箱的坏样本率
        self.ref_bad_rates_ = []
        for i in range(self.n_bins_actual_):
            mask = (scores_ref >= self.bin_edges_[i]) & (scores_ref < self.bin_edges_[i + 1])
            if i == self.n_bins_actual_ - 1:  # 最后一个箱包含右边界
                mask = (scores_ref >= self.bin_edges_[i]) & (scores_ref <= self.bin_edges_[i + 1])

            if mask.sum() > 0:
                bad_rate = y_ref[mask].mean()
            else:
                bad_rate = 0.5  # 默认
            self.ref_bad_rates_.append(bad_rate)

        self.ref_bad_rates_ = np.array(self.ref_bad_rates_)

        # 计算当前分布每个箱的坏样本率
        self.cur_bad_rates_ = []
        for i in range(self.n_bins_actual_):
            mask = (scores_cur >= self.bin_edges_[i]) & (scores_cur < self.bin_edges_[i + 1])
            if i == self.n_bins_actual_ - 1:
                mask = (scores_cur >= self.bin_edges_[i]) & (scores_cur <= self.bin_edges_[i + 1])

            if mask.sum() > 0:
                bad_rate = y_prep[mask].mean()
            else:
                bad_rate = 0.5
            self.cur_bad_rates_.append(bad_rate)

        self.cur_bad_rates_ = np.array(self.cur_bad_rates_)

        # 计算校准映射(从当前坏样本率到参考坏样本率)
        # 使用log-odds空间进行映射
        ref_odds = np.clip(self.ref_bad_rates_ / (1 - self.ref_bad_rates_), 1e-10, 1e10)
        cur_odds = np.clip(self.cur_bad_rates_ / (1 - self.cur_bad_rates_), 1e-10, 1e10)

        self.ref_scores_ = np.log(ref_odds)
        self.cur_scores_ = np.log(cur_odds)

        # 边界
        if self.clip_bounds:
            self.lower_bound_ = np.min(scores_ref)
            self.upper_bound_ = np.max(scores_ref)

        self._is_fitted = True

        # 打印对比
        print(f"分箱重校准拟合完成，实际分箱数: {self.n_bins_actual_}")
        print(f"参考坏样本率: {self.ref_bad_rates_}")
        print(f"当前坏样本率: {self.cur_bad_rates_}")

        return self

    def calibrate(
        self,
        scores: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """校准评分.

        :param scores: 原始评分
        :return: 校准后的评分
        """
        check_is_fitted(self)
        scores = np.asarray(scores)

        # 找到每个分数所属的箱
        bin_indices = np.digitize(scores, self.bin_edges_[1:-1], right=True)
        bin_indices = np.clip(bin_indices, 0, self.n_bins_actual_ - 1)

        # 应用校准映射
        scores_calibrated = np.zeros_like(scores)
        for i in range(self.n_bins_actual_):
            mask = bin_indices == i
            if mask.any():
                # 在当前箱内，根据相对位置进行映射
                bin_scores = scores[mask]
                bin_min = self.bin_edges_[i]
                bin_max = self.bin_edges_[i + 1]

                if bin_max > bin_min:
                    # 在箱内的相对位置
                    relative_pos = (bin_scores - bin_min) / (bin_max - bin_min)
                else:
                    relative_pos = 0.5

                # 映射到参考评分空间
                ref_score = self.ref_scores_[i]
                cur_score = self.cur_scores_[i]

                # 调整分数
                scores_calibrated[mask] = bin_scores + (ref_score - cur_score) * 0.1

        # 边界限制
        if self.clip_bounds:
            scores_calibrated = np.clip(
                scores_calibrated,
                self.lower_bound_,
                self.upper_bound_
            )

        return scores_calibrated


class ScoreDriftCalibrator(BaseDriftCalibrator):
    """统一评分漂移校准器接口.

    提供统一的接口，支持多种漂移校准方法。

    **参数**

    :param method: 校准方法，默认'linear'
        - 'linear': 线性漂移校准
        - 'quantile': 分位数对齐
        - 'binning': 分箱重校准
    :param reference_scores: 参考评分，可选
    :param clip_bounds: 是否限制边界，默认True
    :param target: 目标列名，默认'target'
        - 用于从DataFrame中提取目标变量
    :param kwargs: 传递给具体校准器的参数

    **常用配置**

    **线性校准(推荐)**
    >>> calibrator = ScoreDriftCalibrator(method='linear')
    >>> calibrator.fit(model, X_production, X_reference=X_train)
    >>> scores_calib = calibrator.predict_score(X_test)

    **分位数对齐**
    >>> calibrator = ScoreDriftCalibrator(method='quantile', n_quantiles=100)
    >>> calibrator.fit(model, X_production, X_reference=X_train)

    **分箱重校准(需要标签)**
    >>> calibrator = ScoreDriftCalibrator(method='binning', n_bins=10)
    >>> calibrator.fit(model, X_production, y_production, X_reference=X_train, y_reference=y_train)

    **示例**

    **sklearn风格**
    >>> calibrator = ScoreDriftCalibrator(method='linear')
    >>> calibrator.fit(model, X_prod, y_prod, X_reference=X_train)
    >>> scores_calib = calibrator.predict_score(X_test)

    **scorecardpipeline风格**
    >>> calibrator = ScoreDriftCalibrator(method='linear')
    >>> calibrator.fit(model, df_prod, X_reference=X_train)  # df_prod包含target列
    >>> scores_calib = calibrator.predict_score(df_test)
    """

    def __init__(
        self,
        method: Literal['linear', 'quantile', 'binning'] = 'linear',
        reference_scores: Optional[Union[np.ndarray, pd.Series]] = None,
        clip_bounds: bool = True,
        target: str = 'target',
        **kwargs
    ):
        super().__init__(reference_scores, method, clip_bounds)
        self.target = target
        self.calibrator_params = kwargs

    def fit(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X_reference: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_reference: Optional[Union[np.ndarray, pd.Series]] = None,
        target: Optional[str] = None,
        **kwargs
    ) -> 'ScoreDriftCalibrator':
        """拟合漂移校准器.

        支持两种传参风格:

        **sklearn风格**::
            calibrator.fit(model, X_prod, y_prod, X_reference=X_train)

        **scorecardpipeline风格**::
            calibrator.fit(model, df_prod, X_reference=X_train)  # df_prod包含target列

        :param model: 已训练的模型
        :param X: 当前生产环境数据
        :param y: 当前标签，可选
        :param X_reference: 参考数据
        :param y_reference: 参考标签，可选
        :param target: 目标列名，默认使用初始化时设置的target
        :return: self
        """
        # 使用初始化时设置的target
        target = target or self.target

        # 创建具体的校准器
        if self.method == 'linear':
            self.calibrator_ = LinearDriftCalibrator(
                reference_scores=self.reference_scores,
                clip_bounds=self.clip_bounds,
                **self.calibrator_params
            )
        elif self.method == 'quantile':
            self.calibrator_ = QuantileAligner(
                reference_scores=self.reference_scores,
                clip_bounds=self.clip_bounds,
                **self.calibrator_params
            )
        elif self.method == 'binning':
            self.calibrator_ = BinningRecalibrator(
                reference_scores=self.reference_scores,
                clip_bounds=self.clip_bounds,
                **self.calibrator_params
            )
        else:
            raise ValueError(f"不支持的校准方法: {self.method}")

        # 拟合具体校准器
        fit_kwargs = {'target': target, **kwargs}
        if y_reference is not None:
            fit_kwargs['y_reference'] = y_reference

        self.calibrator_.fit(model, X, y, X_reference=X_reference, **fit_kwargs)

        # 复制重要属性
        self.model_ = self.calibrator_.model_

        self._is_fitted = True

        return self

    def calibrate(
        self,
        scores: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """校准评分.

        :param scores: 原始评分
        :return: 校准后的评分
        """
        check_is_fitted(self)
        return self.calibrator_.calibrate(scores)


# 尝试导入matplotlib进行绘图
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_drift_comparison(
    calibrator: BaseDriftCalibrator,
    X_reference: Union[np.ndarray, pd.DataFrame],
    X_current: Union[np.ndarray, pd.DataFrame],
    figsize: Tuple[int, int] = (15, 5),
    show: bool = True,
    colors: Optional[List[str]] = None
) -> Any:
    """绘制漂移对比图.

    :param calibrator: 已拟合的漂移校准器
    :param X_reference: 参考数据
    :param X_current: 当前数据
    :param figsize: 图表大小，默认(15, 5)
    :param show: 是否显示图表，默认True
    :param colors: 颜色列表，默认使用hscredit配色 ["#2639E9", "#F76E6C", "#FE7715"]
    :return: matplotlib Figure对象

    **示例**

    >>> calibrator = ScoreDriftCalibrator(method='linear')
    >>> calibrator.fit(model, X_production, X_reference=X_train)
    >>> fig = plot_drift_comparison(calibrator, X_train, X_production)
    >>> fig.savefig('drift_comparison.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("需要安装matplotlib才能绘图")

    # hscredit默认配色
    if colors is None:
        colors = ["#2639E9", "#F76E6C", "#FE7715"]

    check_is_fitted(calibrator)

    # 辅助函数：设置坐标轴样式
    def _setup_axis_style(ax, color="#2639E9"):
        ax.spines['top'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 获取评分
    scores_ref = calibrator.model_.predict_proba(X_reference)[:, 1]
    scores_cur = calibrator.model_.predict_proba(X_current)[:, 1]
    scores_cur_calib = calibrator.predict_score(X_current)

    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 左图: 分布对比
    ax1 = axes[0]
    ax1.hist(scores_ref, bins=30, alpha=0.6, label='Reference', color=colors[0], density=True, edgecolor='white')
    ax1.hist(scores_cur, bins=30, alpha=0.6, label='Current (Raw)', color=colors[1], density=True, edgecolor='white')
    ax1.set_xlabel('Score', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Distribution Comparison', fontweight='bold')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3, linestyle='--')
    _setup_axis_style(ax1)

    # 中图: 校准后对比
    ax2 = axes[1]
    ax2.hist(scores_ref, bins=30, alpha=0.6, label='Reference', color=colors[0], density=True, edgecolor='white')
    ax2.hist(scores_cur_calib, bins=30, alpha=0.6, label='Current (Calibrated)', color=colors[2], density=True, edgecolor='white')
    ax2.set_xlabel('Score', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('After Calibration', fontweight='bold')
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.3, linestyle='--')
    _setup_axis_style(ax2)

    # 右图: Q-Q图
    ax3 = axes[2]
    quantiles = np.linspace(0, 1, 100)
    ref_q = np.quantile(scores_ref, quantiles)
    cur_q = np.quantile(scores_cur, quantiles)
    cur_calib_q = np.quantile(scores_cur_calib, quantiles)

    ax3.plot(ref_q, cur_q, 'o', color=colors[1], label='Raw', alpha=0.5, markersize=4)
    ax3.plot(ref_q, cur_calib_q, 'o', color=colors[2], label='Calibrated', alpha=0.5, markersize=4)
    ax3.plot([ref_q.min(), ref_q.max()], [ref_q.min(), ref_q.max()], 'k--', label='Ideal', alpha=0.5)
    ax3.set_xlabel('Reference Quantiles', fontweight='bold')
    ax3.set_ylabel('Current Quantiles', fontweight='bold')
    ax3.set_title('Q-Q Plot', fontweight='bold')
    ax3.legend(frameon=False)
    ax3.grid(True, alpha=0.3, linestyle='--')
    _setup_axis_style(ax3)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def compare_drift_methods(
    model: Any,
    X_reference: Union[np.ndarray, pd.DataFrame],
    X_current: Union[np.ndarray, pd.DataFrame],
    y_current: Optional[Union[np.ndarray, pd.Series]] = None,
    methods: List[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    show: bool = True,
    colors: Optional[List[str]] = None
) -> Any:
    """对比多种漂移校准方法.

    :param model: 已训练的模型
    :param X_reference: 参考数据
    :param X_current: 当前数据
    :param y_current: 当前标签，可选(用于binning方法)
    :param methods: 要对比的方法列表，默认['linear', 'quantile']
    :param figsize: 图表大小，默认(12, 4)
    :param show: 是否显示图表，默认True
    :param colors: 颜色列表，默认使用hscredit配色 ["#2639E9", "#F76E6C", "#FE7715"]
    :return: matplotlib Figure对象

    **示例**

    >>> fig = compare_drift_methods(model, X_train, X_production)
    >>> fig.savefig('drift_methods_comparison.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("需要安装matplotlib才能绘图")

    # hscredit默认配色
    if colors is None:
        colors = ["#2639E9", "#F76E6C", "#FE7715"]

    if methods is None:
        methods = ['linear', 'quantile']

    # 辅助函数：设置坐标轴样式
    def _setup_axis_style(ax, color="#2639E9"):
        ax.spines['top'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 获取评分
    scores_ref = model.predict_proba(X_reference)[:, 1]
    scores_cur = model.predict_proba(X_current)[:, 1]

    # 创建图表
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=figsize)

    # 第一个子图: 原始分布
    ax = axes[0]
    ax.hist(scores_ref, bins=30, alpha=0.6, label='Reference', color=colors[0], density=True, edgecolor='white')
    ax.hist(scores_cur, bins=30, alpha=0.6, label='Current', color=colors[1], density=True, edgecolor='white')
    ax.set_xlabel('Score', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Original', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    _setup_axis_style(ax)

    # 其他子图: 各种校准方法
    for idx, method in enumerate(methods):
        ax = axes[idx + 1]

        try:
            calibrator = ScoreDriftCalibrator(method=method)

            if method == 'binning' and y_current is not None:
                calibrator.fit(model, X_current, y_current, X_reference=X_reference)
            else:
                calibrator.fit(model, X_current, X_reference=X_reference)

            scores_calib = calibrator.predict_score(X_current)

            ax.hist(scores_ref, bins=30, alpha=0.6, label='Reference', color=colors[0], density=True, edgecolor='white')
            ax.hist(scores_calib, bins=30, alpha=0.6, label='Calibrated', color=colors[2], density=True, edgecolor='white')
            ax.set_xlabel('Score', fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.set_title(f'Method: {method}', fontweight='bold')
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3, linestyle='--')
            _setup_axis_style(ax)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Method: {method}', fontweight='bold')
            _setup_axis_style(ax)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


# 将绘图方法添加到BaseDriftCalibrator类
BaseDriftCalibrator.plot_drift_comparison = plot_drift_comparison
