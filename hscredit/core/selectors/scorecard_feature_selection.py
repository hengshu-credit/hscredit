"""评分卡风格特征粗筛器.

参考 scorecardpipeline.FeatureSelection 的常用粗筛流程，
按缺失率、IV、相关性、单一值占比顺序筛选特征，
但完整复用 hscredit 现有筛选器体系与双风格 API。

**参考样例**

>>> from hscredit.core.selectors import ScorecardFeatureSelection
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'f{i}' for i in range(10)])
>>> y = pd.Series(np.random.randint(0, 2, 1000))
>>> selector = ScorecardFeatureSelection(
...     null_threshold=0.95,
...     iv_threshold=0.02,
...     corr_threshold=0.7,
...     mode_threshold=0.95,
... )
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector
from .null_selector import NullSelector
from .iv_selector import IVSelector
from .corr_selector import CorrSelector
from .mode_selector import ModeSelector


class ScorecardFeatureSelection(BaseFeatureSelector):
    """评分卡风格特征粗筛器.

    按如下顺序执行特征筛选：

    1. 缺失率筛选（NullSelector）
    2. IV值筛选（IVSelector）
    3. 相关性筛选（CorrSelector）
    4. 单一值筛选（ModeSelector）

    与 scorecardpipeline.FeatureSelection 的思路一致，
    但参数命名和基础行为遵循 hscredit 风格：

    - 缺失率阈值使用 ``null_threshold``
    - IV阈值使用 ``iv_threshold``
    - 相关性阈值使用 ``corr_threshold``
    - 单一值占比阈值使用 ``mode_threshold``
    - 强制保留使用 ``include``
    - 强制剔除使用 ``exclude`` / ``force_drop``
    - 支持 sklearn 风格 ``fit(X, y)``
    - 支持 scorecardpipeline 风格 ``fit(df)``

    **参数**

    :param null_threshold: 缺失率阈值，默认为0.95；设为None可关闭该阶段
    :param iv_threshold: IV阈值，默认为0.02；设为None可关闭该阶段
    :param corr_threshold: 相关性阈值，默认为0.7；设为None可关闭该阶段
    :param mode_threshold: 单一值占比阈值，默认为0.95；设为None可关闭该阶段
    :param corr_method: 相关系数计算方法，默认为'pearson'
        - 'pearson': 皮尔逊相关系数
        - 'spearman': 斯皮尔曼等级相关系数
        - 'kendall': 肯德尔相关系数
    :param corr_metric: 相关性筛选的保留指标，默认为'iv'
        - 'iv': 信息值（需要目标变量y）
        - 'ks': KS统计量
        - 'lift': LIFT值
        - 'bad_rate': 坏样本率
        指标通过分箱后的bin_tables_计算得到。
    :param corr_weights: 自定义相关性筛选权重，优先级高于corr_metric
    :param corr_binning_params: 透传给CorrSelector的分箱参数
    :param iv_regularization: IV计算正则项，默认为1.0
    :param mode_dropna: 计算单一值占比时是否将缺失值作为独立类别，默认为True
    :param target: 目标变量列名，默认为'target'
    :param include: 强制保留特征列表
    :param exclude: 强制剔除特征列表
    :param force_drop: 强制剔除特征列表，最终会合并到exclude
    :param target_rm: transform时是否移除目标列，默认为False
    :param n_jobs: 并行任务数

    **参考样例**

    >>> from hscredit.core.selectors import ScorecardFeatureSelection
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'f{i}' for i in range(10)])
    >>> y = pd.Series(np.random.randint(0, 2, 1000))
    >>> selector = ScorecardFeatureSelection(
    ...     null_threshold=0.95,
    ...     iv_threshold=0.02,
    ...     corr_threshold=0.7,
    ...     mode_threshold=0.95,
    ... )
    >>> selector.fit(X, y)
    >>> print(selector.selected_features_)
    >>> print(selector.stage_report_df_)
    """

    def __init__(
        self,
        null_threshold: Optional[float] = 0.95,
        iv_threshold: Optional[float] = 0.02,
        corr_threshold: Optional[float] = 0.7,
        mode_threshold: Optional[float] = 0.95,
        corr_method: str = 'pearson',
        corr_metric: str = 'iv',
        corr_weights: Optional[Union[pd.Series, Dict[str, float], List[float]]] = None,
        corr_binning_params: Optional[Dict[str, Any]] = None,
        iv_regularization: float = 1.0,
        mode_dropna: bool = True,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        target_rm: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__(
            target=target,
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            threshold='multi-stage',
            n_jobs=n_jobs,
        )
        self.null_threshold = null_threshold
        self.iv_threshold = iv_threshold
        self.corr_threshold = corr_threshold
        self.mode_threshold = mode_threshold
        self.corr_method = corr_method
        self.corr_metric = corr_metric
        self.corr_weights = corr_weights
        self.corr_binning_params = corr_binning_params
        self.iv_regularization = iv_regularization
        self.mode_dropna = mode_dropna
        self.target_rm = target_rm
        self.method_name = '评分卡特征粗筛'

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> 'ScorecardFeatureSelection':
        """拟合评分卡风格筛选器并同步兼容属性。"""
        super().fit(X, y)
        self._finalize_selection_result()
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray, List[str]],
    ) -> Union[pd.DataFrame, np.ndarray, List[str]]:
        """根据筛选结果转换数据。"""
        transformed = super().transform(X)

        if self.target_rm and isinstance(transformed, pd.DataFrame) and self.target in transformed.columns:
            return transformed.drop(columns=[self.target])

        return transformed

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """按评分卡粗筛顺序执行筛选。"""
        self._validate_configuration(y)
        self._get_feature_names(X)

        current_X = X.copy()
        self.stage_selectors_ = {}
        self.stage_reports_ = []
        self.stage_report_df_ = pd.DataFrame()
        self.corr_iv_scores_ = None
        all_dropped = []

        if self._is_stage_enabled(self.null_threshold) and len(current_X.columns) > 0:
            current_X = self._run_stage(
                stage_key='empty',
                stage_name='缺失率筛选',
                selector=NullSelector(
                    threshold=self.null_threshold,
                    target=self.target,
                    n_jobs=self.n_jobs,
                ),
                current_X=current_X,
                y=y,
                all_dropped=all_dropped,
            )

        iv_scores = None
        if self._is_stage_enabled(self.iv_threshold) and len(current_X.columns) > 0:
            iv_selector = IVSelector(
                threshold=self.iv_threshold,
                target=self.target,
                regularization=self.iv_regularization,
                n_jobs=self.n_jobs,
            )
            current_X = self._run_stage(
                stage_key='iv',
                stage_name='IV值筛选',
                selector=iv_selector,
                current_X=current_X,
                y=y,
                all_dropped=all_dropped,
            )
            iv_scores = getattr(iv_selector, 'scores_', None)

        if self._is_stage_enabled(self.corr_threshold) and len(current_X.columns) > 0:
            corr_weights = self._resolve_corr_weights(current_X, y, iv_scores)
            corr_selector = CorrSelector(
                threshold=self.corr_threshold,
                method=self.corr_method,
                metric=self.corr_metric,
                weights=corr_weights,
                binning_params=self.corr_binning_params,
                target=self.target,
                n_jobs=self.n_jobs,
            )
            current_X = self._run_stage(
                stage_key='corr',
                stage_name='相关性筛选',
                selector=corr_selector,
                current_X=current_X,
                y=y,
                all_dropped=all_dropped,
            )

        if self._is_stage_enabled(self.mode_threshold) and len(current_X.columns) > 0:
            current_X = self._run_stage(
                stage_key='identical',
                stage_name='单一值筛选',
                selector=ModeSelector(
                    threshold=self.mode_threshold,
                    dropna=self.mode_dropna,
                    target=self.target,
                    n_jobs=self.n_jobs,
                ),
                current_X=current_X,
                y=y,
                all_dropped=all_dropped,
            )

        self.selected_features_ = current_X.columns.tolist()
        self.scores_ = None

        if len(all_dropped) > 0:
            self.dropped_ = pd.concat(all_dropped, ignore_index=True)
        else:
            self.dropped_ = pd.DataFrame(
                columns=['特征', '剔除原因', '筛选阶段', '筛选阶段名称', '筛选器']
            )

        self.removed_features_ = self.dropped_['特征'].tolist() if len(self.dropped_) > 0 else []

        if self.stage_reports_:
            self.stage_report_df_ = pd.DataFrame(self.stage_reports_)

    def _run_stage(
        self,
        stage_key: str,
        stage_name: str,
        selector: BaseFeatureSelector,
        current_X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
        all_dropped: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """执行单个筛选阶段并记录结果。"""
        input_count = current_X.shape[1]
        selector.fit(current_X, y)

        self.stage_selectors_[stage_key] = selector
        self.stage_reports_.append({
            '阶段键': stage_key,
            '阶段名称': stage_name,
            '筛选器': selector.__class__.__name__,
            '输入特征数': input_count,
            '选中特征数': len(selector.selected_features_),
            '剔除特征数': len(getattr(selector, 'removed_features_', [])),
            '阈值': getattr(selector, 'threshold', None),
        })

        if hasattr(selector, 'dropped_') and selector.dropped_ is not None and len(selector.dropped_) > 0:
            dropped = selector.dropped_.copy()
            dropped['筛选阶段'] = stage_key
            dropped['筛选阶段名称'] = stage_name
            dropped['筛选器'] = selector.__class__.__name__
            all_dropped.append(dropped)

        selected = selector.selected_features_
        if len(selected) == 0:
            return current_X.iloc[:, 0:0]

        return current_X[selected]

    def _resolve_corr_weights(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
        iv_scores: Optional[pd.Series],
    ) -> Optional[Union[pd.Series, Dict[str, float], List[float]]]:
        """为相关性筛选解析保留权重。"""
        if self.corr_weights is not None:
            return self.corr_weights

        if iv_scores is not None:
            return iv_scores.reindex(X.columns).fillna(0.0)

        if str(self.corr_metric).lower() == 'iv':
            if y is None:
                raise ValueError('启用 corr_threshold 且使用 IV 作为保留指标时，需要传入 y 或 target 列')

            iv_selector = IVSelector(
                threshold=float('-inf'),
                target=self.target,
                regularization=self.iv_regularization,
                n_jobs=self.n_jobs,
            )
            iv_selector.fit(X, y)
            self.corr_iv_scores_ = iv_selector.scores_.reindex(X.columns).fillna(0.0)
            return self.corr_iv_scores_

        return None

    def _validate_configuration(
        self,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """校验配置合法性。"""
        self._validate_ratio_threshold(self.null_threshold, 'null_threshold')
        self._validate_ratio_threshold(self.corr_threshold, 'corr_threshold')
        self._validate_ratio_threshold(self.mode_threshold, 'mode_threshold')

        if self._is_stage_enabled(self.iv_threshold) and self.iv_threshold < 0:
            raise ValueError('iv_threshold 不能小于 0')

        requires_target = self._is_stage_enabled(self.iv_threshold)
        requires_target = requires_target or (
            self._is_stage_enabled(self.corr_threshold)
            and self.corr_weights is None
            and str(self.corr_metric).lower() == 'iv'
        )
        requires_target = requires_target or (
            self._is_stage_enabled(self.corr_threshold)
            and self.corr_weights is None
            and str(self.corr_metric).lower() != 'iv'
        )

        if requires_target and y is None:
            raise ValueError('当前筛选配置需要目标变量，请使用 fit(X, y) 或 fit(df) 且 df 包含 target 列')

    @staticmethod
    def _is_stage_enabled(threshold: Optional[float]) -> bool:
        """判断筛选阶段是否启用。"""
        return threshold is not None and threshold is not False

    @staticmethod
    def _validate_ratio_threshold(threshold: Optional[float], name: str) -> None:
        """校验 0-1 比例阈值。"""
        if threshold is None or threshold is False:
            return
        if not 0 <= threshold <= 1:
            raise ValueError(f'{name} 必须在 [0, 1] 范围内')

    def _finalize_selection_result(self) -> None:
        """整理最终选择结果并补齐兼容属性。"""
        selected_set = set(self.selected_features_)
        self.selected_features_ = [c for c in self._feature_names if c in selected_set]
        self.n_features_ = len(self.selected_features_)

        if hasattr(self, 'dropped_') and self.dropped_ is not None and len(self.dropped_) > 0:
            dropped_df = self.dropped_.copy()

            if '筛选阶段' not in dropped_df.columns:
                dropped_df['筛选阶段'] = None
            if '筛选阶段名称' not in dropped_df.columns:
                dropped_df['筛选阶段名称'] = None
            if '筛选器' not in dropped_df.columns:
                dropped_df['筛选器'] = None

            force_drop_mask = dropped_df['剔除原因'].astype(str).str.contains('强制剔除', na=False)
            dropped_df.loc[force_drop_mask, '筛选阶段'] = 'force_drop'
            dropped_df.loc[force_drop_mask, '筛选阶段名称'] = '强制剔除'

            dropped_df = dropped_df.loc[~dropped_df['特征'].isin(self.selected_features_)].copy()
            dropped_df = dropped_df.drop_duplicates(subset=['特征'], keep='first').reset_index(drop=True)

            self.dropped_ = dropped_df
            self.removed_features_ = dropped_df['特征'].tolist()
        else:
            self.dropped_ = pd.DataFrame(
                columns=['特征', '剔除原因', '筛选阶段', '筛选阶段名称', '筛选器']
            )
            self.removed_features_ = []

        self.select_columns = list(self.selected_features_)
        if not self.target_rm and self.target not in self.select_columns:
            self.select_columns.append(self.target)

        self.dropped = pd.DataFrame({
            'variable': self.dropped_['特征'] if len(self.dropped_) > 0 else pd.Series(dtype=object),
            'rm_reason': self.dropped_['筛选阶段'] if len(self.dropped_) > 0 else pd.Series(dtype=object),
        })

    def get_selection_report(self) -> Dict[str, Any]:
        """获取包含阶段明细的筛选报告。"""
        report = super().get_selection_report()
        if report.get('状态') == '未拟合':
            return report

        report['阈值'] = {
            'null_threshold': self.null_threshold,
            'iv_threshold': self.iv_threshold,
            'corr_threshold': self.corr_threshold,
            'mode_threshold': self.mode_threshold,
        }
        if hasattr(self, 'stage_reports_'):
            report['阶段明细'] = self.stage_reports_
        return report