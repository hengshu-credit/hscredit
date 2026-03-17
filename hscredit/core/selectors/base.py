"""特征筛选器基类.

定义特征筛选器的统一接口和通用方法。
所有筛选器都继承此类,确保API的一致性。

报告系统设计:
1. 单个筛选器报告: 每个筛选器实现 get_selection_report(),返回标准化报告
2. 全局报告收集器: SelectionReportCollector 自动收集 Pipeline 中所有筛选器的结果
3. 报告格式: 统一的中文格式,包含统计信息、选中/剔除特征、得分等
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SelectionReportCollector:
    """特征筛选报告收集器.

    自动收集 Pipeline 中所有筛选器的结果,生成汇总报告。
    支持在 sklearn Pipeline 中作为回调使用,或手动添加筛选器。

    **使用方式**

    ::

        >>> from hscredit.core.selection import (
        ...     SelectionReportCollector,
        ...     VarianceSelector,
        ...     CorrSelector
        ... )
        >>> collector = SelectionReportCollector()
        >>> 
        >>> # 方式1: 手动添加筛选器
        >>> selector1 = VarianceSelector(threshold=0.1)
        >>> selector1.fit(X)
        >>> collector.add_report(selector1)
        >>> 
        >>> selector2 = CorrSelector(threshold=0.8)
        >>> selector2.fit(X, y)
        >>> collector.add_report(selector2)
        >>> 
        >>> # 获取汇总报告
        >>> summary = collector.get_summary()
        >>> print(summary)
        >>> 
        >>> # 导出为DataFrame
        >>> df = collector.to_dataframe()
    """

    def __init__(self, name: str = "特征筛选流程"):
        """初始化报告收集器。

        :param name: 流程名称,用于报告中显示
        """
        self.name = name
        self.reports: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        self._feature_origin_count: Optional[int] = None

    def add_report(
        self,
        selector: 'BaseFeatureSelector',
        stage_name: Optional[str] = None
    ) -> 'SelectionReportCollector':
        """添加筛选器报告。

        :param selector: 已拟合的筛选器对象
        :param stage_name: 阶段名称,如'粗筛''精筛'等
        :return: self
        """
        if not hasattr(selector, 'get_selection_report'):
            raise ValueError("selector 必须实现 get_selection_report() 方法")

        report = selector.get_selection_report()
        
        # 添加阶段名称
        if stage_name:
            report['stage_name'] = stage_name
        else:
            report['stage_name'] = f"阶段{len(self.reports) + 1}"

        # 记录第一个筛选器的输入特征数作为原始特征数
        if self._feature_origin_count is None and len(self.reports) == 0:
            self._feature_origin_count = report.get('输入特征数')

        self.reports.append(report)
        return self

    def add_selector(
        self,
        selector: 'BaseFeatureSelector',
        stage_name: Optional[str] = None
    ) -> 'SelectionReportCollector':
        """添加筛选器（add_report的别名）。

        :param selector: 已拟合的筛选器对象
        :param stage_name: 阶段名称
        :return: self
        """
        return self.add_report(selector, stage_name)

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总报告。

        :return: 包含所有筛选器结果的字典
        """
        if len(self.reports) == 0:
            return {
                "状态": "无筛选记录",
                "message": "请先添加筛选器报告"
            }

        # 计算统计信息
        total_selected = self.reports[-1].get('选中特征数', 0) if self.reports else 0
        total_dropped = sum(r.get('输入特征数', 0) - r.get('选中特征数', 0) for r in self.reports)

        summary = {
            "流程名称": self.name,
            "创建时间": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "筛选轮次": len(self.reports),
            "原始特征数": self._feature_origin_count,
            "最终特征数": total_selected,
            "累计剔除特征数": total_dropped,
            "特征保留率": f"{total_selected / self._feature_origin_count * 100:.2f}%" if self._feature_origin_count else "N/A",
            "筛选器列表": [
                {
                    "阶段": r.get('stage_name', f'阶段{i+1}'),
                    "筛选器": r.get('筛选器', r.get('筛选方法', 'Unknown')),
                    "输入": r.get('输入特征数', 0),
                    "输出": r.get('选中特征数', 0),
                    "剔除": r.get('输入特征数', 0) - r.get('选中特征数', 0),
                    "阈值": r.get('阈值', 'N/A'),
                }
                for i, r in enumerate(self.reports)
            ]
        }

        return summary

    def get_feature_trace(self) -> pd.DataFrame:
        """获取特征追踪表。

        记录每个特征在每个筛选阶段的状态。

        :return: 特征追踪DataFrame
        """
        if len(self.reports) == 0:
            return pd.DataFrame()

        # 收集所有特征
        all_features = set()
        for r in self.reports:
            all_features.update(r.get('选中特征', []))
            if '剔除特征' in r:
                all_features.update(r.get('剔除特征', []))

        # 构建追踪表
        trace_data = []
        current_features = set(self.reports[0].get('选中特征', [])) if self.reports else set()

        for i, r in enumerate(self.reports):
            stage = r.get('stage_name', f'阶段{i+1}')
            selected = set(r.get('选中特征', []))
            dropped = set(r.get('剔除特征', []))
            scores = r.get('特征得分', {})
            dropped_reasons = r.get('剔除原因', [])

            for feat in all_features:
                status = '选中' if feat in selected else ('剔除' if feat in dropped else '未处理')
                
                # 获取得分或剔除原因
                if status == '选中':
                    score_value = scores.get(feat, 'N/A')
                elif status == '剔除':
                    # 找到该特征在剔除列表中的索引
                    try:
                        idx = list(dropped).index(feat)
                        score_value = dropped_reasons[idx] if idx < len(dropped_reasons) else 'N/A'
                    except (ValueError, IndexError):
                        score_value = 'N/A'
                else:
                    score_value = 'N/A'
                
                trace_data.append({
                    '特征': feat,
                    '阶段': stage,
                    '筛选器': r.get('筛选器', 'Unknown'),
                    '状态': status,
                    '得分/原因': score_value
                })

        return pd.DataFrame(trace_data)

    def get_dropped_summary(self) -> pd.DataFrame:
        """获取被剔除特征的汇总表。

        :return: 剔除特征汇总DataFrame
        """
        if len(self.reports) == 0:
            return pd.DataFrame()

        dropped_records = []
        for i, r in enumerate(self.reports):
            dropped_features = r.get('剔除特征', [])
            dropped_reasons = r.get('剔除原因', [])

            for j, feat in enumerate(dropped_features):
                reason = dropped_reasons[j] if j < len(dropped_reasons) else 'Unknown'
                dropped_records.append({
                    '特征': feat,
                    '阶段': r.get('stage_name', f'阶段{i+1}'),
                    '筛选器': r.get('筛选器', 'Unknown'),
                    '剔除原因': reason,
                    '得分': r.get('特征得分', {}).get(feat, 'N/A')
                })

        return pd.DataFrame(dropped_records)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame格式。

        :return: 筛选结果DataFrame
        """
        if len(self.reports) == 0:
            return pd.DataFrame()

        rows = []
        for i, r in enumerate(self.reports):
            row = {
                '阶段': r.get('stage_name', f'阶段{i+1}'),
                '筛选器': r.get('筛选器', r.get('筛选方法', 'Unknown')),
                '阈值': r.get('阈值', 'N/A'),
                '输入特征数': r.get('输入特征数', 0),
                '选中特征数': r.get('选中特征数', 0),
                '剔除特征数': r.get('输入特征数', 0) - r.get('选中特征数', 0),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def to_excel(self, filepath: str) -> None:
        """导出为Excel文件。

        :param filepath: 保存路径
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 写入汇总表
            summary_df = self.to_dataframe()
            summary_df.to_excel(writer, sheet_name='筛选汇总', index=False)

            # 写入特征追踪表
            trace_df = self.get_feature_trace()
            if len(trace_df) > 0:
                trace_df.to_excel(writer, sheet_name='特征追踪', index=False)

            # 写入剔除特征表
            dropped_df = self.get_dropped_summary()
            if len(dropped_df) > 0:
                dropped_df.to_excel(writer, sheet_name='剔除特征', index=False)

            # 写入详细报告（JSON格式）
            import json
            summary = self.get_summary()
            # 将numpy类型转换为Python原生类型
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            summary = convert(summary)
            summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
            with open(filepath.replace('.xlsx', '_report.json'), 'w', encoding='utf-8') as f:
                f.write(summary_json)

    def print_summary(self) -> None:
        """打印汇总报告到控制台。"""
        summary = self.get_summary()

        print("=" * 60)
        print(f"特征筛选报告 - {summary['流程名称']}")
        print("=" * 60)
        print(f"创建时间: {summary['创建时间']}")
        print(f"筛选轮次: {summary['筛选轮次']}")
        print(f"原始特征数: {summary['原始特征数']}")
        print(f"最终特征数: {summary['最终特征数']}")
        print(f"累计剔除: {summary['累计剔除特征数']}")
        print(f"特征保留率: {summary['特征保留率']}")
        print()
        print("筛选详情:")
        print("-" * 60)
        print(f"{'阶段':<10} {'筛选器':<20} {'输入':>6} {'输出':>6} {'剔除':>6}")
        print("-" * 60)
        for item in summary['筛选器列表']:
            print(f"{item['阶段']:<10} {item['筛选器']:<20} {item['输入']:>6} {item['输出']:>6} {item['剔除']:>6}")
        print("=" * 60)

    def __len__(self) -> int:
        """返回筛选器数量。"""
        return len(self.reports)

    def __repr__(self) -> str:
        return f"SelectionReportCollector(name='{self.name}', stages={len(self.reports)})"


class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC):
    """特征筛选器基类.

    所有特征筛选器都继承此类,实现统一的fit/transform接口。
    支持中文筛选报告生成。
    支持可选的分箱器,在筛选前对数据进行分箱处理。

    **参数**

    :param target: 目标变量列名,默认为'target'
    :param include: 强制保留的特征列表,这些特征无论如何都会被保留
    :param exclude: 强制剔除的特征列表,这些特征无论如何都会被剔除
    :param binner: 可选的分箱器,支持:
        - 训练好的分箱器（有fit方法）
        - 分箱器类（未训练的,需要传入类而非实例）
        - 分箱器实例（未训练的）
    :param threshold: 筛选阈值,不同筛选器含义不同
    :param n_jobs: 并行计算的任务数,默认为1

    **属性**

    - selected_features_: 选中的特征列表
    - removed_features_: 被剔除的特征列表
    - dropped_: 被剔除的特征及原因DataFrame
    - scores_: 各特征的筛选得分
    - n_features_in_: 输入特征数量
    - forced_dropped_: 被强制剔除的特征列表

    **中文筛选报告**

    报告包含以下内容:
    - 筛选方法名称
    - 阈值设置
    - 选中的特征数量和列表
    - 被剔除的特征及原因
    - 各特征的得分统计

    **使用示例**

    ::

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1,2,3], 'b': [1,1,1], 'c': [1,2,3]})
        >>> # 强制保留特征a,强制剔除特征c
        >>> selector = VarianceSelector(threshold=0.1, include=['a'], exclude=['c'])
        >>> selector.fit(X)
        >>> print(selector.selected_features_)
        ['a']
    """

    def __init__(
        self,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        binner: Optional[Any] = None,
        threshold: Union[float, int, str] = 0.0,
        n_jobs: int = 1,
        force_drop: Optional[List[str]] = None,
    ):
        self.target = target
        self.include = include
        self.exclude = exclude
        self.binner = binner
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.force_drop = force_drop

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> 'BaseFeatureSelector':
        """拟合筛选器,学习特征重要性。

        :param X: 输入特征,DataFrame或numpy数组
        :param y: 目标变量,仅部分筛选器需要
        :return: self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.n_features_in_ = X.shape[1]

        # 处理include参数（强制保留的特征）
        if self.include is None:
            self.include_ = []
        elif isinstance(self.include, str):
            self.include_ = [self.include]
        elif isinstance(self.include, (list, tuple, np.ndarray)):
            self.include_ = list(self.include)
        else:
            self.include_ = []

        # 处理exclude参数（强制剔除的特征）
        if self.exclude is None:
            self.exclude_ = []
        elif isinstance(self.exclude, str):
            self.exclude_ = [self.exclude]
        elif isinstance(self.exclude, (list, tuple, np.ndarray)):
            self.exclude_ = list(self.exclude)
        else:
            self.exclude_ = []

        # 如果有分箱器,先进行分箱
        if self.binner is not None:
            X = self._apply_binner(X, y)

        # 执行子类实现的具体fit逻辑
        self._fit_impl(X, y)

        # 创建初始dropped_（记录_fit_impl中剔除的特征）
        if hasattr(self, 'selected_features_') and self.selected_features_ is not None:
            dropped_cols = [c for c in X.columns if c not in self.selected_features_]
            if len(dropped_cols) > 0:
                reason = getattr(self, '_drop_reason', '不满足筛选条件')
                self.dropped_ = pd.DataFrame({
                    '特征': dropped_cols,
                    '剔除原因': [reason] * len(dropped_cols)
                })
                self.removed_features_ = dropped_cols
            else:
                self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因'])
                self.removed_features_ = []

        # 确保include的特征被保留
        self._apply_include(X)

        # 应用exclude（强制剔除）
        self._apply_exclude(X)

        self._is_fitted = True
        return self

    def _apply_binner(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """应用分箱器对数据进行分箱。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        :return: 分箱后的DataFrame
        """
        binner = self.binner

        # 检查是否是类（未训练的）还是实例（已训练的）
        is_class = isinstance(binner, type)

        if is_class:
            # 未训练的分箱器类,需要实例化并fit
            binner_instance = binner()
            if hasattr(binner_instance, 'fit'):
                if y is not None:
                    binner_instance.fit(X, y)
                else:
                    binner_instance.fit(X)
            self._binner_instance = binner_instance
        else:
            # 已经是实例
            self._binner_instance = binner
            if hasattr(binner, 'fit') and not hasattr(binner, 'fitted_') or not getattr(binner, 'is_fitted_', False):
                # 未训练的分箱器实例
                if y is not None:
                    binner.fit(X, y)
                else:
                    binner.fit(X)

        # 转换数据
        if hasattr(self._binner_instance, 'transform'):
            X_binned = self._binner_instance.transform(X)
        elif hasattr(self._binner_instance, 'apply'):
            # 某些分箱器使用apply方法
            X_binned = self._binner_instance.apply(X)
        else:
            # 如果没有transform方法,原样返回
            return X

        # 确保返回DataFrame
        if isinstance(X_binned, np.ndarray):
            X_binned = pd.DataFrame(X_binned, columns=X.columns, index=X.index)
        return X_binned

    def _apply_include(self, X: pd.DataFrame) -> None:
        """确保include的特征被保留。

        :param X: 输入特征DataFrame
        """
        if hasattr(self, 'selected_features_') and self.selected_features_ is not None:
            # 添加include的特征
            added = False
            for col in self.include_:
                if col in X.columns and col not in self.selected_features_:
                    self.selected_features_.append(col)
                    added = True

            # 如果有添加特征且dropped_不存在,则创建空的dropped_
            if added and not hasattr(self, 'dropped_'):
                dropped_cols = [c for c in X.columns if c not in self.selected_features_]
                if len(dropped_cols) > 0:
                    reason = getattr(self, '_drop_reason', '不满足筛选条件')
                    self.dropped_ = pd.DataFrame({
                        '特征': dropped_cols,
                        '剔除原因': [reason] * len(dropped_cols)
                    })
                    self.removed_features_ = dropped_cols
                else:
                    self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因'])
                    self.removed_features_ = []

    @abstractmethod
    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """子类实现的fit逻辑。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        pass

    def _apply_exclude(self, X: pd.DataFrame) -> None:
        """强制剔除指定的特征。

        :param X: 输入特征DataFrame
        """
        if not hasattr(self, 'selected_features_') or self.selected_features_ is None:
            return

        # 记录被强制剔除的特征
        self.forced_dropped_ = []

        # 移除exclude的特征（遍历用户传入的exclude_列表）
        for col in self.exclude_:
            if col in self.selected_features_:
                self.selected_features_.remove(col)
                self.forced_dropped_.append(col)
            elif col in X.columns:
                # 特征原本在X中但不在selected_features_中（已被筛选掉）
                # 仍然记录为强制剔除
                if col not in self.forced_dropped_:
                    self.forced_dropped_.append(col)

        # 更新dropped_报告,添加强制剔除的原因
        if hasattr(self, 'dropped_') and self.dropped_ is not None and len(self.dropped_) > 0:
            # 添加强制剔除的特征到dropped_
            for col in self.forced_dropped_:
                # 检查是否已经在dropped_中
                if col not in self.dropped_['特征'].values:
                    new_row = pd.DataFrame({
                        '特征': [col],
                        '剔除原因': ['强制剔除']
                    })
                    self.dropped_ = pd.concat([self.dropped_, new_row], ignore_index=True)
                else:
                    # 更新原因为强制剔除
                    self.dropped_.loc[self.dropped_['特征'] == col, '剔除原因'] = '强制剔除'
        elif len(self.forced_dropped_) > 0:
            # 创建新的dropped_记录
            self.dropped_ = pd.DataFrame({
                '特征': self.forced_dropped_,
                '剔除原因': ['强制剔除'] * len(self.forced_dropped_)
            })
        
        # 更新 removed_features_
        if hasattr(self, 'dropped_') and len(self.dropped_) > 0:
            self.removed_features_ = self.dropped_['特征'].tolist()
        else:
            self.removed_features_ = []

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """根据筛选结果转换数据。

        :param X: 输入特征
        :return: 筛选后的特征
        """
        if not hasattr(self, '_is_fitted'):
            raise ValueError("筛选器尚未拟合,请先调用fit方法")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 返回选中的特征
        selected = [c for c in self.selected_features_ if c in X.columns]
        return X[selected]

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并转换数据。

        :param X: 输入特征
        :param y: 目标变量
        :return: 筛选后的特征
        """
        return self.fit(X, y).transform(X)

    @property
    def select_columns_(self) -> List[str]:
        """获取选中的特征列表（向后兼容属性）。

        :return: 选中特征的列表
        """
        if hasattr(self, 'selected_features_'):
            return self.selected_features_
        return []

    def get_support_mask(self) -> np.ndarray:
        """获取特征选择掩码。

        :return: 布尔数组,True表示选中
        """
        if not hasattr(self, '_is_fitted'):
            raise ValueError("筛选器尚未拟合")

        mask = np.zeros(self.n_features_in_, dtype=bool)
        for col in self.selected_features_:
            if col in self._feature_names:
                idx = self._feature_names.index(col)
                mask[idx] = True
        return mask

    def get_selection_report(self) -> Dict[str, Any]:
        """获取中文筛选报告。

        报告格式说明:
        - 基础信息: 筛选器名称、方法、阈值、参数
        - 统计信息: 输入/输出特征数、剔除数、保留率
        - 选中特征: 选中的特征列表
        - 剔除特征: 被剔除的特征及原因（DataFrame格式）
        - 特征得分: 各特征的筛选得分

        :return: 包含筛选结果的字典
        """
        if not hasattr(self, '_is_fitted'):
            return {"状态": "未拟合", "message": "请先调用fit方法"}

        # 收集参数
        params = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key not in ['n_features_in_', 'selected_features_', 'removed_features_', 'scores_', 'dropped_']:
                if isinstance(value, (str, int, float, bool, type(None))):
                    params[key] = value
        
        # 处理threshold参数（可能名称不同）
        if hasattr(self, 'threshold') and self.threshold != 0.0:
            params['threshold'] = self.threshold

        # 添加强制保留/剔除的特征信息
        force_info = {}
        if hasattr(self, 'include_') and self.include_:
            force_info['强制保留'] = self.include_
        if hasattr(self, 'forced_dropped_') and self.forced_dropped_:
            force_info['强制剔除'] = self.forced_dropped_

        # 构建报告
        report = {
            # 基础信息
            "筛选器": self.__class__.__name__,
            "筛选方法": getattr(self, 'method_name', self.__class__.__name__),
            "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            # 参数信息
            "阈值": self.threshold,
            "参数": params,

            # 强制保留/剔除信息
            "强制操作": force_info if force_info else None,

            # 统计信息
            "输入特征数": self.n_features_in_,
            "选中特征数": len(self.selected_features_),
            "剔除特征数": self.n_features_in_ - len(self.selected_features_),
            "特征保留率": f"{len(self.selected_features_) / self.n_features_in_ * 100:.2f}%" if self.n_features_in_ > 0 else "0%",

            # 特征列表
            "选中特征": self.selected_features_,
        }

        # 添加dropped信息（DataFrame格式,便于后续分析）
        if hasattr(self, 'dropped_') and len(self.dropped_) > 0:
            report["剔除特征"] = self.dropped_['特征'].tolist()
            report["剔除原因"] = self.dropped_['剔除原因'].tolist()
            report["剔除详情"] = self.dropped_.to_dict('records')

        # 添加scores信息
        if hasattr(self, 'scores_') and self.scores_ is not None:
            # 转换numpy类型为Python原生类型
            scores_dict = {}
            for k, v in self.scores_.to_dict().items():
                if isinstance(v, (np.integer, np.floating)):
                    scores_dict[k] = float(v)
                else:
                    scores_dict[k] = v
            report["特征得分"] = scores_dict
            
            # 添加得分统计
            valid_scores = [v for v in scores_dict.values() if isinstance(v, (int, float))]
            if valid_scores:
                report["得分统计"] = {
                    "最大值": max(valid_scores),
                    "最小值": min(valid_scores),
                    "平均值": sum(valid_scores) / len(valid_scores),
                    "中位数": sorted(valid_scores)[len(valid_scores) // 2],
                }

        return report

    def get_selection_report_df(self) -> pd.DataFrame:
        """获取简化的DataFrame格式报告。

        适用于快速查看和导出。

        :return: 报告DataFrame
        """
        report = self.get_selection_report()
        
        # 提取关键信息
        row = {
            '筛选器': report.get('筛选器', ''),
            '筛选方法': report.get('筛选方法', ''),
            '阈值': report.get('阈值', ''),
            '输入特征数': report.get('输入特征数', 0),
            '选中特征数': report.get('选中特征数', 0),
            '剔除特征数': report.get('剔除特征数', 0),
            '保留率': report.get('特征保留率', ''),
        }
        
        return pd.DataFrame([row])

    def get_scores_df(self) -> pd.DataFrame:
        """获取特征得分的DataFrame。

        :return: 包含特征和得分的DataFrame
        """
        if not hasattr(self, 'scores_') or self.scores_ is None:
            return pd.DataFrame(columns=['特征', '得分', '状态'])
        
        scores = self.scores_.copy()
        selected = set(self.selected_features_)
        
        records = []
        for feat, score in scores.items():
            # 转换numpy类型
            if isinstance(score, (np.integer, np.floating)):
                score = float(score)
            
            status = '选中' if feat in selected else '剔除'
            records.append({
                '特征': feat,
                '得分': score,
                '状态': status
            })
        
        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values('得分', ascending=False)
        
        return df

    def get_dropped_df(self) -> pd.DataFrame:
        """获取被剔除特征的DataFrame。

        :return: 包含被剔除特征及原因的DataFrame
        """
        if hasattr(self, 'dropped_'):
            return self.dropped_
        return pd.DataFrame(columns=['特征', '剔除原因'])

    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """获取特征名称列表。

        :param X: 输入特征DataFrame
        :return: 特征名称列表
        """
        if hasattr(X, 'columns'):
            self._feature_names = X.columns.tolist()
        else:
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        return self._feature_names


class CompositeFeatureSelector(BaseFeatureSelector):
    """组合特征筛选器.

    将多个筛选器组合在一起,按顺序执行筛选。
    后续筛选器基于前面筛选器的结果进行筛选。
    支持通过include和exclude参数强制保留或剔除特定特征。

    **参数**

    :param selectors: 筛选器列表,按执行顺序排列
    :param strategy: 组合策略,'sequential'或'intersection'
        - 'sequential': 按顺序筛选,每轮剔除不满足条件的特征
        - 'intersection': 取所有筛选器选中特征的交集
    :param include: 强制保留的特征列表,这些特征无论如何都会被保留
    :param exclude: 强制剔除的特征列表,这些特征无论如何都会被剔除

    **示例**

    ::

        >>> from hscredit.core.selectors import (
        ...     VarianceSelector, CorrSelector, IVSelector
        ... )
        >>> # 强制保留特征id,强制剔除特征useless_col
        >>> composite = CompositeFeatureSelector([
        ...     VarianceSelector(threshold=0.01),
        ...     CorrSelector(threshold=0.8),
        ...     IVSelector(threshold=0.02),
        ... ], include=['id'], exclude=['useless_col'])
        >>> composite.fit(X, y)
    """

    def __init__(
        self,
        selectors: List[BaseFeatureSelector],
        strategy: str = 'sequential',
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        binner: Optional[Any] = None,
    ):
        super().__init__(target=target, include=include, exclude=exclude, binner=binner)
        self.selectors = selectors
        self.strategy = strategy

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """执行组合筛选。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if self.strategy == 'sequential':
            self._fit_sequential(X, y)
        else:
            self._fit_intersection(X, y)

    def _fit_sequential(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]) -> None:
        """顺序筛选策略。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        current_X = X.copy()
        all_dropped = []

        for i, selector in enumerate(self.selectors):
            # 使用当前特征进行筛选
            selector.fit(current_X, y)

            # 获取选中特征
            selected = selector.selected_features_

            # 记录被剔除的特征
            if hasattr(selector, 'dropped_') and len(selector.dropped_) > 0:
                dropped = selector.dropped_.copy()
                dropped['筛选轮次'] = i + 1
                dropped['筛选器'] = selector.__class__.__name__
                all_dropped.append(dropped)

            # 更新当前特征
            if len(selected) > 0:
                current_X = current_X[selected]
            else:
                break

        # 最终选中的特征
        self.selected_features_ = current_X.columns.tolist()
        self.scores_ = None
        self.forced_dropped_ = []  # 初始化forced_dropped_

        # 合并所有剔除记录
        if len(all_dropped) > 0:
            self.dropped_ = pd.concat(all_dropped, ignore_index=True)
            self.removed_features_ = self.dropped_['特征'].tolist()
        else:
            self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因', '筛选轮次', '筛选器'])
            self.removed_features_ = []

    def _fit_intersection(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]) -> None:
        """交集筛选策略。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        selected_sets = []

        for selector in self.selectors:
            selector.fit(X, y)
            selected_sets.append(set(selector.selected_features_))

        # 取交集
        self.selected_features_ = list(set.intersection(*selected_sets))
        self.scores_ = None
        self.forced_dropped_ = []  # 初始化forced_dropped_
        self.dropped_ = pd.DataFrame({
            '特征': [c for c in X.columns if c not in self.selected_features_],
            '剔除原因': '未被所有筛选器同时选中'
        })
        self.removed_features_ = self.dropped_['特征'].tolist()
