"""特征筛选器基类.

定义特征筛选器的统一接口和通用方法。
所有筛选器都继承此类,确保API的一致性。

报告系统设计:
1. 单个筛选器报告: 每个筛选器实现 get_selection_report(),返回标准化报告
2. 全局报告收集器: SelectionReportCollector 自动收集 Pipeline 中所有筛选器的结果
3. 报告格式: 统一的中文格式,包含统计信息、选中/剔除特征、得分等

**参考样例**

>>> from hscredit.core.selectors.base import BaseFeatureSelector, SelectionReportCollector
>>> from hscredit.core.selectors import VarianceSelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
>>> y = pd.Series(np.random.randint(0, 2, 100))
>>> selector = VarianceSelector(threshold=0.1)
>>> selector.fit(X, y)
>>> report = selector.get_selection_report()
>>> print(f"选中特征数: {report['选中特征数']}")
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ...exceptions import NotFittedError, ValidationError


def get_feature_importances(estimator) -> np.ndarray:
    """从任意模型中提取特征重要性.

    兼容以下模型类型:
    - 树模型 (feature_importances_): sklearn RandomForest/GradientBoosting,
      hscredit RiskModels, XGBClassifier, LGBMClassifier, CatBoostClassifier
    - 线性模型 (coef_): sklearn LogisticRegression, LinearSVC 等
    - 原生 xgboost Booster (get_score)
    - 原生 lightgbm Booster (feature_importance)
    - 原生 catboost CatBoost (get_feature_importance)

    **参数**

    :param estimator: 已训练的模型对象
    :return: 一维 numpy 数组，长度为特征数
    :raises ValidationError: 当无法从模型中提取重要性时
    """
    # 1. feature_importances_ — 最通用 (sklearn tree models, hscredit models, XGB/LGB/CB sklearn API)
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        if isinstance(importances, pd.Series):
            return importances.values.astype(float)
        return np.asarray(importances, dtype=float)

    # 2. coef_ — 线性模型 (sklearn LogisticRegression, LinearSVC, etc.)
    if hasattr(estimator, 'coef_'):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef[0]
        return np.abs(coef)

    # 3. 原生 xgboost Booster
    if hasattr(estimator, 'get_score'):
        try:
            scores = estimator.get_score(importance_type='gain')
            if hasattr(estimator, 'feature_names') and estimator.feature_names:
                return np.array([scores.get(f, 0.0) for f in estimator.feature_names], dtype=float)
            n = max(int(k.replace('f', '')) for k in scores) + 1 if scores else 0
            return np.array([scores.get(f'f{i}', 0.0) for i in range(n)], dtype=float)
        except Exception:
            pass

    # 4. 原生 lightgbm Booster
    if hasattr(estimator, 'feature_importance') and callable(estimator.feature_importance):
        try:
            return np.asarray(estimator.feature_importance(importance_type='gain'), dtype=float)
        except Exception:
            pass

    # 5. 原生 catboost (Pool-based API)
    if hasattr(estimator, 'get_feature_importance') and callable(estimator.get_feature_importance):
        try:
            return np.asarray(estimator.get_feature_importance(), dtype=float)
        except Exception:
            pass

    raise ValidationError(
        f"无法从 {type(estimator).__name__} 中提取特征重要性。"
        f"模型需要提供 feature_importances_、coef_ 属性，"
        f"或 get_score / feature_importance / get_feature_importance 方法。"
    )


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

        **参数**

        :param name: 流程名称，用于报告中显示，默认值为 "特征筛选流程"

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> collector = SelectionReportCollector(name="我的筛选流程")
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

        将已拟合的筛选器结果添加到收集器中，并生成阶段名称。阶段名称默认为"阶段{序号}"。

        **参数**

        :param selector: 已拟合的筛选器对象，必须实现 `get_selection_report()` 方法
        :param stage_name: 阶段名称，如 '粗筛'、'精筛' 等，默认根据已有报告数量自动生成

        :returns: self（支持链式调用）

        :raises ValidationError: 当 selector 未实现 `get_selection_report()` 方法时

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector, stage_name="粗筛")
        SelectionReportCollector(name='特征筛选流程', stages=1)
        """
        if not hasattr(selector, 'get_selection_report'):
            raise ValidationError("selector 必须实现 get_selection_report() 方法")

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

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总报告。

        汇总报告包含筛选流程的整体统计信息和各阶段的筛选详情。

        **返回字典键值说明**

        - **流程名称** (`str`): 筛选流程名称
        - **创建时间** (`str`): 报告创建时间，格式为 YYYY-MM-DD HH:MM:SS
        - **筛选轮次** (`int`): 执行的筛选阶段总数
        - **原始特征数** (`int`): 第一阶段输入的特征数量
        - **最终特征数** (`int`): 最后一个阶段输出的特征数量
        - **累计剔除特征数** (`int`): 所有阶段累计剔除的特征总数
        - **特征保留率** (`str`): 特征保留比例，格式为 "XX.XX%"，无记录时为 "N/A"
        - **筛选器列表** (`List[Dict]`): 各阶段的筛选器详情列表，每项包含阶段、筛选器、输入、输出、剔除、阈值

        :returns: 包含汇总统计和阶段详情的字典，无记录时返回包含 "状态" 和 "message" 的字典

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> summary = collector.get_summary()
        >>> print(f"原始特征数: {summary['原始特征数']}, 最终特征数: {summary['最终特征数']}")
        原始特征数: 5, 最终特征数: ...
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

        记录每个特征在每个筛选阶段的状态变化，包括选中、剔除或未处理。

        **返回 DataFrame 列说明**

        - **特征** (`str`): 特征名称
        - **阶段** (`str`): 筛选阶段的名称
        - **筛选器** (`str`): 该阶段使用的筛选器类名
        - **状态** (`str`): 特征在当前阶段的状态，取值为 '选中'、'剔除' 或 '未处理'
        - **得分/原因** (`Any`): 选中特征的得分，或剔除特征的原因

        :returns: 特征追踪 DataFrame，无记录时返回空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> trace = collector.get_feature_trace()
        >>> print(trace.head())
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

        汇总所有筛选阶段中被剔除的特征及其剔除原因。

        **返回 DataFrame 列说明**

        - **特征** (`str`): 被剔除的特征名称
        - **阶段** (`str`): 该特征被剔除的阶段名称
        - **筛选器** (`str`): 该阶段使用的筛选器类名
        - **剔除原因** (`str`): 特征被剔除的原因描述
        - **得分** (`Any`): 特征在筛选器中的得分，无得分时为 'N/A'

        :returns: 剔除特征汇总 DataFrame，无记录时返回空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> dropped = collector.get_dropped_summary()
        >>> print(f"共剔除 {len(dropped)} 个特征")
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
        """转换为 DataFrame 格式。

        将收集到的各筛选阶段报告汇总为一个 DataFrame，每行对应一个筛选阶段。

        **返回 DataFrame 列说明**

        - **阶段** (`str`): 筛选阶段的名称
        - **筛选器** (`str`): 该阶段使用的筛选器类名
        - **阈值** (`Any`): 该阶段使用的筛选阈值
        - **输入特征数** (`int`): 该阶段输入的特征数量
        - **选中特征数** (`int`): 该阶段输出的特征数量
        - **剔除特征数** (`int`): 该阶段剔除的特征数量

        :returns: 筛选结果汇总 DataFrame，无记录时返回空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> df = collector.to_dataframe()
        >>> print(df)
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
        """导出为 Excel 文件。

        将筛选报告导出为多 Sheet 的 Excel 文件，同时生成一份 JSON 格式的详细报告。

        **Excel 文件内容**

        - **筛选汇总** Sheet: 各筛选阶段的统计汇总
        - **特征追踪** Sheet: 每个特征在各阶段的状态变化（仅在有记录时生成）
        - **剔除特征** Sheet: 被剔除特征的详细信息（仅在有记录时生成）
        - **JSON 报告**: 与 Excel 同目录的 `{文件名}_report.json`，包含完整汇总信息

        :param filepath: 保存路径，必须以 `.xlsx` 结尾

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> collector.to_excel("筛选报告.xlsx")
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
        """打印汇总报告到控制台。

        以格式化的方式打印筛选流程的整体统计信息和各阶段详情。

        **打印内容**

        - 筛选流程名称和创建时间
        - 筛选轮次、原始/最终特征数、累计剔除数、特征保留率
        - 各阶段筛选器详情（阶段名、筛选器名称、输入/输出/剔除特征数）

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> collector.print_summary()  # doctest: +SKIP
        """
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
        """返回已添加的筛选器数量。

        :returns: 已收集的报告数量
        """
        return len(self.reports)

    def __repr__(self) -> str:
        """返回收集器的字符串表示。

        :returns: 形如 ``SelectionReportCollector(name='...', stages=N)`` 的字符串
        """
        return f"SelectionReportCollector(name='{self.name}', stages={len(self.reports)})"


class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC):
    """特征筛选器基类.

    所有特征筛选器都继承此类，实现统一的 fit/transform 接口。
    支持中文筛选报告生成。支持可选的分箱器，在筛选前对数据进行分箱处理。

    **参数**

    :param target: 目标变量列名，默认为 'target'。在 scorecardpipeline 风格中用于
        从 DataFrame 中提取目标列；在 sklearn 风格中若 fit 传入了 y 参数则优先使用 y
    :param include: 强制保留的特征列表，这些特征无论如何都会被保留
    :param exclude: 强制剔除的特征列表，这些特征无论如何都会被剔除
    :param binner: 可选的分箱器，支持已训练的分箱器实例、待训练的分箱器类（传入类而非实例）、
        或待训练的分箱器实例
    :param threshold: 筛选阈值，不同筛选器含义不同
    :param n_jobs: 并行计算的任务数，默认为 1
    :param force_drop: 强制剔除的特征列表，效果与 exclude 合并

    **属性**

    - selected_features_: 选中保留的特征列表
    - removed_features_: 被剔除的特征列表
    - dropped_: 被剔除的特征及原因 DataFrame，包含 '特征' 和 '剔除原因' 两列
    - scores_: 各特征的筛选得分的 Series
    - n_features_in_: fit 时输入的特征数量
    - forced_dropped_: 被强制剔除的特征列表
    - include_: 处理后的强制保留特征列表（字符串会被转为单元素列表）
    - exclude_: 处理后的强制剔除特征列表（包含 force_drop 的合并结果）

    **参考样例**

    **sklearn 风格（推荐用于纯特征矩阵）**

    ::

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1, include=['feat_0'], exclude=['feat_4'])
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> print(selector.selected_features_)
        ['feat_0', ...]

    **scorecardpipeline 风格（推荐用于完整数据框）**

    ::

        >>> from hscredit.core.selectors import IVSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'feat1': np.random.randn(1000),
        ...     'feat2': [1] * 1000,  # 低方差，会被剔除
        ...     'target': np.random.randint(0, 2, 1000)
        ... })
        >>> selector = IVSelector(threshold=0.02, target='target')
        >>> selector.fit(df)
        IVSelector(...)
        >>> report = selector.get_selection_report()
        >>> print(f"输入: {report['输入特征数']}, 输出: {report['选中特征数']}")
        输入: 2, 输出: 1
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
        """初始化特征筛选器。

        **参数**

        :param target: 目标变量列名，默认为 'target'。当 fit 传入 y 参数时优先使用 y
        :param include: 强制保留的特征列表，这些特征无论如何都会被保留
        :param exclude: 强制剔除的特征列表，这些特征无论如何都会被剔除
        :param binner: 可选的分箱器，支持已训练/未训练的分箱器实例或类
        :param threshold: 筛选阈值，不同筛选器含义不同
        :param n_jobs: 并行计算的任务数，默认为 1
        :param force_drop: 强制剔除的特征列表，效果与 exclude 合并
        """
        self.target = target
        self.include = include
        self.exclude = exclude
        self.binner = binner
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.force_drop = force_drop

    def _check_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Tuple[pd.DataFrame, Optional[Union[pd.Series, np.ndarray]]]:
        """检查并处理输入数据，支持两种 API 风格。

        **API 风格**

        1. **sklearn 风格**: ``fit(X, y)`` — X 是特征矩阵，y 是目标变量
        2. **scorecardpipeline 风格**: ``fit(df)`` — df 是包含特征和目标列的完整 DataFrame，
           目标列名通过初始化参数 target 指定

        **优先级规则**: 如果 y 不为 None，使用 sklearn 风格；否则检查 X 中是否包含 target 列。

        **参数**

        :param X: 输入特征矩阵（DataFrame 或 numpy 数组）或包含目标列的完整数据框
        :param y: 目标变量，可选。如果不为 None 则优先使用

        :returns: 二元组 ``(处理后的特征 DataFrame, 目标变量或 None)``

        **参考样例**

        >>> from hscredit.core.selectors.base import BaseFeatureSelector
        >>> import pandas as pd
        >>> class DummySelector(BaseFeatureSelector):
        ...     def _fit_impl(self, X, y): pass
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X_df = pd.DataFrame(np.random.randn(5, 3), columns=['a', 'b', 'c'])
        >>> y_s = pd.Series([0, 1, 0, 1, 0])
        >>> sel = DummySelector()
        >>> X_out, y_out = sel._check_input(X_df, y_s)
        >>> X_out.shape
        (5, 3)
        """
        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 如果y不为None,使用sklearn风格
        if y is not None:
            return X, y

        # y为None时,检查是否是scorecardpipeline风格
        if isinstance(X, pd.DataFrame) and self.target in X.columns:
            # 从X中分离目标列
            y = X[self.target]
            X = X.drop(columns=[self.target])
            return X, y

        # 没有目标变量(如无监督筛选器)
        return X, None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> 'BaseFeatureSelector':
        """拟合筛选器，学习特征重要性。

        支持两种 API 风格：sklearn 风格 ``fit(X, y)`` 和 scorecardpipeline 风格 ``fit(df)``。

        **参数**

        :param X: 输入特征矩阵（DataFrame 或 numpy 数组），或包含目标列的完整数据框
        :param y: 目标变量，可选。如果不为 None 则优先使用

        :returns: self

        :raises NotFittedError: 如果在子类实现中需要目标变量但 y 为 None

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> len(selector.selected_features_)
        ...
        """
        # 检查输入并分离特征和目标
        X_processed, y_processed = self._check_input(X, y)

        # 保存特征名
        self._get_feature_names(X_processed)
        self.n_features_in_ = X_processed.shape[1]

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

        # 处理force_drop参数（合并到exclude_中）
        if self.force_drop is not None:
            if isinstance(self.force_drop, str):
                force_drop_list = [self.force_drop]
            elif isinstance(self.force_drop, (list, tuple, np.ndarray)):
                force_drop_list = list(self.force_drop)
            else:
                force_drop_list = []
            # 合并到exclude_（去重）
            self.exclude_ = list(set(self.exclude_ + force_drop_list))

        # 如果有分箱器,先进行分箱
        if self.binner is not None:
            X_processed = self._apply_binner(X_processed, y_processed)

        # 执行子类实现的具体fit逻辑
        self._fit_impl(X_processed, y_processed)

        # 创建初始dropped_（只在子类没有设置dropped_时）
        if hasattr(self, 'selected_features_') and self.selected_features_ is not None:
            # 如果子类已经设置了详细的dropped_，则保留
            if not hasattr(self, 'dropped_') or self.dropped_ is None or len(self.dropped_) == 0:
                dropped_cols = [c for c in X_processed.columns if c not in self.selected_features_]
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
        self._apply_include(X_processed)

        # 应用exclude（强制剔除）
        self._apply_exclude(X_processed)

        self._is_fitted = True
        return self

    def _apply_binner(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """应用分箱器对数据进行分箱。

        支持传入已训练的分箱器实例或待训练的分箱器类/实例。如果传入未训练的分箱器，
        将自动在输入数据上进行拟合。

        **参数**

        :param X: 输入特征 DataFrame
        :param y: 目标变量，可选

        :returns: 分箱后的 DataFrame，列名与输入保持一致

        **参考样例**

        >>> from hscredit.core.selectors.base import BaseFeatureSelector
        >>> from hscredit.core.binning import DecisionTreeBinner
        >>> import pandas as pd
        >>> class DummySelector(BaseFeatureSelector):
        ...     def _fit_impl(self, X, y): pass
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        >>> sel = DummySelector(binner=DecisionTreeBinner(max_depth=2))
        >>> X_binned = sel._apply_binner(X)
        >>> X_binned.shape
        (100, 3)
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
        """确保 include 的特征被保留。

        将初始化时传入的 include 参数中的特征添加到 selected_features_ 列表中。

        **参数**

        :param X: 输入特征 DataFrame
        """
        if hasattr(self, 'selected_features_') and self.selected_features_ is not None:
            # 添加include的特征
            added = False
            for col in self.include_:
                if col in X.columns and col not in self.selected_features_:
                    self.selected_features_.append(col)
                    added = True

            # 如果有添加特征且dropped_不存在,则创建空的dropped_
            if added and (not hasattr(self, 'dropped_') or self.dropped_ is None):
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
        """子类实现的 fit 逻辑。

        子类必须重写此方法，在其中实现具体的特征筛选计算逻辑，
        并设置 self.selected_features_ 和 self.scores_ 属性。

        **参数**

        :param X: 输入特征 DataFrame（已通过 _check_input 处理）
        :param y: 目标变量，可选
        """
        pass

    def _apply_exclude(self, X: pd.DataFrame) -> None:
        """强制剔除指定的特征。

        将初始化时传入的 exclude 和 force_drop 参数中的特征从 selected_features_ 中移除，
        并记录到 forced_dropped_ 和 dropped_ 属性中。

        **参数**

        :param X: 输入特征 DataFrame
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
                    new_row = pd.DataFrame({'特征': [col], '剔除原因': ['强制剔除']})
                    self.dropped_ = pd.concat([self.dropped_, new_row], ignore_index=True)
                else:
                    # 在原原因后附加"[强制剔除]"标记
                    current_reason = self.dropped_.loc[self.dropped_['特征'] == col, '剔除原因'].iloc[0]
                    if '[强制剔除]' not in str(current_reason):
                        self.dropped_.loc[self.dropped_['特征'] == col, '剔除原因'] = f'{current_reason} [强制剔除]'
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
        X: Union[pd.DataFrame, np.ndarray, List[str]],
    ) -> Union[pd.DataFrame, np.ndarray, List[str]]:
        """根据筛选结果转换数据。

        根据 fit 阶段选中的特征，对输入数据进行筛选。

        **参数**

        :param X: 输入数据，支持 DataFrame、numpy 数组或特征名列表

        :returns: 筛选后的数据，类型与输入一致
            - DataFrame/ndarray 输入：返回仅包含选中特征的 DataFrame（若包含 target 列则透传）
            - 列表输入：返回筛选后的特征名列表

        :raises NotFittedError: 当筛选器尚未拟合时

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> X_selected = selector.transform(X)
        >>> X_selected.shape[1] < X.shape[1]
        True
        """
        if not hasattr(self, '_is_fitted'):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")

        # 如果传入的是列表，返回筛选后的特征列表
        if isinstance(X, list):
            selected = [c for c in X if c in self.selected_features_]
            return selected

        # 如果传入的是 DataFrame 或 ndarray
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 返回选中的特征
        selected = [c for c in self.selected_features_ if c in X.columns]

        # scorecardpipeline 风格: 如果 X 中包含 target 列，透传到输出
        target_col = getattr(self, 'target', None)
        if target_col and target_col in X.columns and target_col not in selected:
            return X[selected + [target_col]]

        return X[selected]

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并转换数据。

        等价于依次调用 fit(X, y) 和 transform(X)。

        **参数**

        :param X: 输入特征矩阵（DataFrame 或 numpy 数组），或包含目标列的完整数据框
        :param y: 目标变量，可选

        :returns: 筛选后的特征数据
        """
        return self.fit(X, y).transform(X)

    def get_support_mask(self) -> np.ndarray:
        """获取特征选择掩码。

        返回一个布尔数组，长度为输入特征数，True 表示该特征被选中。

        :returns: 布尔 numpy 数组，True 对应选中特征的下标

        :raises NotFittedError: 当筛选器尚未拟合时

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> mask = selector.get_support_mask()
        >>> mask.sum() == len(selector.selected_features_)
        True
        """
        if not hasattr(self, '_is_fitted'):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")

        mask = np.zeros(self.n_features_in_, dtype=bool)
        for col in self.selected_features_:
            if col in self._feature_names:
                idx = self._feature_names.index(col)
                mask[idx] = True
        return mask

    def get_selection_report(self) -> Dict[str, Any]:
        """获取中文筛选报告。

        报告包含筛选器的基础信息、统计信息、选中/剔除特征列表及特征得分。

        **返回字典键值说明**

        - **筛选器** (`str`): 筛选器类名
        - **筛选方法** (`str`): 筛选方法名称
        - **时间戳** (`str`): 报告生成时间，格式为 YYYY-MM-DD HH:MM:SS
        - **阈值** (`Any`): 筛选阈值
        - **参数** (`Dict`): 筛选器初始化参数（过滤后的有效参数）
        - **强制操作** (`Dict` | `None`): 强制保留/剔除的特征信息
        - **输入特征数** (`int`): fit 时的输入特征数量
        - **选中特征数** (`int`): 选中的特征数量
        - **剔除特征数** (`int`): 被剔除的特征数量
        - **特征保留率** (`str`): 保留比例，格式为 "XX.XX%"
        - **选中特征** (`List[str]`): 选中特征名称列表
        - **剔除特征** (`List[str]`，可选): 被剔除特征名称列表
        - **剔除原因** (`List[str]`，可选): 各剔除特征对应的原因
        - **剔除详情** (`List[Dict]`，可选): 剔除特征的详细信息
        - **特征得分** (`Dict`): 各特征得分的字典
        - **得分统计** (`Dict`，可选): 得分的统计摘要（最大值、最小值、平均值、中位数）

        :returns: 包含筛选结果的字典，未拟合时返回包含 "状态" 和 "message" 的字典

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> report = selector.get_selection_report()
        >>> print(f"输入: {report['输入特征数']}, 选中: {report['选中特征数']}")
        输入: 5, 选中: ...
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
        """获取简化的 DataFrame 格式报告。

        适用于快速查看和导出到 Excel 等场景。

        **返回 DataFrame 列说明**

        - **筛选器** (`str`): 筛选器类名
        - **筛选方法** (`str`): 筛选方法名称
        - **阈值** (`Any`): 筛选阈值
        - **输入特征数** (`int`): 输入特征数量
        - **选中特征数** (`int`): 选中特征数量
        - **剔除特征数** (`int`): 剔除特征数量
        - **保留率** (`str`): 特征保留比例

        :returns: 包含筛选报告关键指标的 DataFrame（单行）

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> report_df = selector.get_selection_report_df()
        >>> print(report_df)
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
        """获取特征得分的 DataFrame。

        **返回 DataFrame 列说明**

        - **特征** (`str`): 特征名称
        - **得分** (`float`): 特征在筛选器中的得分
        - **状态** (`str`): 特征状态，取值为 '选中' 或 '剔除'

        DataFrame 按得分降序排列。

        :returns: 包含特征得分的 DataFrame，无得分时返回仅含列名的空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> scores_df = selector.get_scores_df()
        >>> print(scores_df.head())
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
        """获取被剔除特征的 DataFrame。

        **返回 DataFrame 列说明**

        - **特征** (`str`): 被剔除的特征名称
        - **剔除原因** (`str`): 特征被剔除的原因描述

        :returns: 包含被剔除特征及原因的 DataFrame，无数据时返回仅含列名的空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> dropped_df = selector.get_dropped_df()
        >>> print(f"共剔除 {len(dropped_df)} 个特征")
        """
        if hasattr(self, 'dropped_'):
            return self.dropped_
        return pd.DataFrame(columns=['特征', '剔除原因'])

    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """获取特征名称列表。

        **参数**

        :param X: 输入特征 DataFrame

        :returns: 特征名称列表，同时设置 sklearn 兼容的 feature_names_in_ 属性
        """
        if hasattr(X, 'columns'):
            self._feature_names = X.columns.tolist()
        else:
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        # sklearn 兼容：设置 feature_names_in_ 属性
        self.feature_names_in_ = np.array(self._feature_names)
        return self._feature_names


class CompositeFeatureSelector(BaseFeatureSelector):
    """组合特征筛选器.

    将多个筛选器组合在一起，按顺序执行筛选。后续筛选器基于前面筛选器的结果进行筛选。
    支持通过 include 和 exclude 参数强制保留或剔除特定特征。

    **参数**

    :param selectors: 筛选器列表，按执行顺序排列
    :param strategy: 组合策略，可选 'sequential' 或 'intersection'
        - 'sequential': 按顺序筛选，每轮剔除不满足条件的特征
        - 'intersection': 取所有筛选器选中特征的交集
    :param include: 强制保留的特征列表，这些特征无论如何都会被保留
    :param exclude: 强制剔除的特征列表，这些特征无论如何都会被剔除
    :param target: 目标变量列名，默认为 'target'
    :param binner: 可选的分箱器

    **参考样例**

    ::

        >>> from hscredit.core.selectors import (
        ...     VarianceSelector, CorrSelector, IVSelector
        ... )
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        >>> y = pd.Series(np.random.randint(0, 2, 200))
        >>> composite = CompositeFeatureSelector([
        ...     VarianceSelector(threshold=0.01),
        ...     CorrSelector(threshold=0.8),
        ...     IVSelector(threshold=0.02),
        ... ])
        >>> composite.fit(X, y)
        CompositeFeatureSelector(...)
        >>> len(composite.selected_features_)
        ...
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
        """初始化组合特征筛选器。

        **参数**

        :param selectors: 筛选器列表，按执行顺序排列
        :param strategy: 组合策略，默认为 'sequential'
        :param target: 目标变量列名，默认为 'target'
        :param include: 强制保留的特征列表
        :param exclude: 强制剔除的特征列表
        :param binner: 可选的分箱器
        """
        super().__init__(target=target, include=include, exclude=exclude, binner=binner)
        self.selectors = selectors
        self.strategy = strategy

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """执行组合筛选。

        **参数**

        :param X: 输入特征 DataFrame
        :param y: 目标变量
        """
        if self.strategy == 'sequential':
            self._fit_sequential(X, y)
        else:
            self._fit_intersection(X, y)

    def _fit_sequential(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]) -> None:
        """顺序筛选策略。

        按列表顺序依次执行每个筛选器，后续筛选器仅在上一轮选中的特征上进行筛选。

        **参数**

        :param X: 输入特征 DataFrame
        :param y: 目标变量
        """
        current_X = X.copy()
        all_dropped = []

        for i, item in enumerate(self.selectors):
            # 处理 ('name', selector) 元组格式或直接的 selector 对象
            if isinstance(item, tuple) and len(item) == 2:
                selector_name = item[0]
                selector = item[1]
            else:
                selector_name = item.__class__.__name__
                selector = item

            # 使用当前特征进行筛选
            selector.fit(current_X, y)

            # 获取选中特征
            selected = selector.selected_features_

            # 记录被剔除的特征（穿透收集详细指标）
            if hasattr(selector, 'dropped_') and len(selector.dropped_) > 0:
                dropped = selector.dropped_.copy()
                dropped['筛选轮次'] = i + 1
                dropped['筛选器'] = selector_name
                dropped['筛选器类型'] = selector.__class__.__name__
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
            self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因', '筛选轮次', '筛选器', '筛选器类型'])
            self.removed_features_ = []

    def _fit_intersection(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]) -> None:
        """交集筛选策略。

        所有筛选器独立对全量特征进行筛选，最终取各筛选器选中特征的交集。

        **参数**

        :param X: 输入特征 DataFrame
        :param y: 目标变量
        """
        selected_sets = []
        all_dropped = []

        for item in self.selectors:
            # 处理 ('name', selector) 元组格式或直接的 selector 对象
            if isinstance(item, tuple) and len(item) == 2:
                selector_name = item[0]
                selector = item[1]
            else:
                selector_name = item.__class__.__name__
                selector = item

            selector.fit(X, y)
            selected_sets.append(set(selector.selected_features_))

            # 收集每个筛选器的剔除信息
            if hasattr(selector, 'dropped_') and len(selector.dropped_) > 0:
                dropped = selector.dropped_.copy()
                dropped['筛选器'] = selector_name
                dropped['筛选器类型'] = selector.__class__.__name__
                all_dropped.append(dropped)

        # 取交集
        self.selected_features_ = list(set.intersection(*selected_sets))
        self.scores_ = None
        self.forced_dropped_ = []  # 初始化forced_dropped_

        # 构建详细的剔除原因
        removed_features = [c for c in X.columns if c not in self.selected_features_]
        if len(removed_features) > 0:
            # 对于每个被剔除的特征，记录其被哪些筛选器剔除
            drop_reasons = []
            for feature in removed_features:
                rejected_by = []
                for item in self.selectors:
                    if isinstance(item, tuple) and len(item) == 2:
                        sel_name = item[0]
                        sel = item[1]
                    else:
                        sel_name = item.__class__.__name__
                        sel = item
                    if hasattr(sel, 'dropped_') and len(sel.dropped_) > 0:
                        if feature in sel.dropped_['特征'].values:
                            rejected_by.append(sel_name)
                if rejected_by:
                    drop_reasons.append(f"被以下筛选器剔除: {', '.join(rejected_by)}")
                else:
                    drop_reasons.append("未被所有筛选器同时选中")

            self.dropped_ = pd.DataFrame({
                '特征': removed_features,
                '剔除原因': drop_reasons
            })
        else:
            self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因'])

        self.removed_features_ = removed_features

        # 保存详细的筛选器结果（用于穿透查询）
        if len(all_dropped) > 0:
            self.detailed_dropped_ = pd.concat(all_dropped, ignore_index=True)
        else:
            self.detailed_dropped_ = pd.DataFrame()

    def get_selection_report_df(self) -> pd.DataFrame:
        """获取详细的 DataFrame 格式报告，穿透底层筛选器。

        记录每一步具体的筛选情况和筛选原因，包括每个特征在各阶段的状态变化、
        被剔除的原因和具体数值指标、以及每个子筛选器的详细信息。

        **返回 DataFrame 列说明**

        - **状态** (`str`): 报告生成状态，未拟合时为 '未拟合'
        - **message** (`str`): 状态消息，未拟合时为 '请先调用fit方法'

        未拟合时返回包含状态和消息的单行 DataFrame。

        :returns: 详细的筛选报告 DataFrame
        """
        if not hasattr(self, '_is_fitted'):
            return pd.DataFrame({'状态': ['未拟合'], 'message': ['请先调用fit方法']})
        
        records = []
        
        # 获取原始特征列表（从第一个筛选器的输入）
        all_features = set()
        for item in self.selectors:
            if isinstance(item, tuple) and len(item) == 2:
                selector = item[1]
            else:
                selector = item
            if hasattr(selector, '_feature_names'):
                all_features.update(selector._feature_names)
        
        all_features = sorted(list(all_features))
        
        if self.strategy == 'sequential':
            # 顺序策略：记录每个特征在每个阶段的状态变化
            current_features = set(all_features)
            
            for i, item in enumerate(self.selectors):
                if isinstance(item, tuple) and len(item) == 2:
                    selector_name = item[0]
                    selector = item[1]
                else:
                    selector_name = item.__class__.__name__
                    selector = item
                
                input_count = len(current_features)
                selected = set(selector.selected_features_) if hasattr(selector, 'selected_features_') else set()
                dropped_in_stage = current_features - selected
                
                # 获取该筛选器的详细报告
                selector_report = selector.get_selection_report() if hasattr(selector, 'get_selection_report') else {}
                scores = selector_report.get('特征得分', {}) if isinstance(selector_report, dict) else {}
                
                # 获取剔除详情
                dropped_details = {}
                if hasattr(selector, 'dropped_') and len(selector.dropped_) > 0:
                    for _, row in selector.dropped_.iterrows():
                        feat = row.get('特征', '')
                        reason = row.get('剔除原因', '不满足筛选条件')
                        dropped_details[feat] = reason
                
                # 为每个特征创建记录
                for feat in sorted(current_features):
                    score = scores.get(feat, 'N/A') if isinstance(scores, dict) else 'N/A'
                    
                    if feat in selected:
                        status = '选中'
                        reason = '满足筛选条件'
                    else:
                        status = '剔除'
                        reason = dropped_details.get(feat, '不满足筛选条件')
                    
                    records.append({
                        '特征': feat,
                        '筛选轮次': i + 1,
                        '筛选器': selector_name,
                        '筛选器类型': selector.__class__.__name__,
                        '策略': self.strategy,
                        '状态': status,
                        '得分': score if isinstance(score, (int, float)) else 'N/A',
                        '剔除原因': reason if status == '剔除' else '',
                        '该轮输入特征数': input_count,
                        '该轮输出特征数': len(selected),
                        '该轮剔除特征数': input_count - len(selected),
                    })
                
                # 更新当前特征集
                current_features = selected
        
        else:  # intersection 策略
            # 收集每个筛选器的结果
            selector_results = []
            
            for item in self.selectors:
                if isinstance(item, tuple) and len(item) == 2:
                    selector_name = item[0]
                    selector = item[1]
                else:
                    selector_name = item.__class__.__name__
                    selector = item
                
                selected = set(selector.selected_features_) if hasattr(selector, 'selected_features_') else set()
                
                # 获取剔除详情
                dropped_details = {}
                if hasattr(selector, 'dropped_') and len(selector.dropped_) > 0:
                    for _, row in selector.dropped_.iterrows():
                        feat = row.get('特征', '')
                        reason = row.get('剔除原因', '不满足筛选条件')
                        dropped_details[feat] = reason
                
                selector_results.append({
                    'name': selector_name,
                    'type': selector.__class__.__name__,
                    'selected': selected,
                    'dropped_details': dropped_details,
                    'report': selector.get_selection_report() if hasattr(selector, 'get_selection_report') else {}
                })
            
            final_selected = set(self.selected_features_) if hasattr(self, 'selected_features_') else set()
            
            # 为每个原始特征创建记录
            for feat in all_features:
                # 确定最终状态
                if feat in final_selected:
                    final_status = '选中'
                    final_reason = '被所有筛选器同时选中'
                else:
                    final_status = '剔除'
                    rejected_by = []
                    for result in selector_results:
                        if feat not in result['selected']:
                            rejected_by.append(result['name'])
                    final_reason = f"被以下筛选器剔除: {', '.join(rejected_by)}" if rejected_by else '未被所有筛选器同时选中'
                
                # 为每个筛选器创建一行记录
                for i, result in enumerate(selector_results):
                    scores = result['report'].get('特征得分', {}) if isinstance(result['report'], dict) else {}
                    score = scores.get(feat, 'N/A') if isinstance(scores, dict) else 'N/A'
                    
                    if feat in result['selected']:
                        stage_status = '选中'
                        stage_reason = '满足筛选条件'
                    else:
                        stage_status = '剔除'
                        stage_reason = result['dropped_details'].get(feat, '不满足筛选条件')
                    
                    records.append({
                        '特征': feat,
                        '筛选轮次': i + 1,
                        '筛选器': result['name'],
                        '筛选器类型': result['type'],
                        '策略': self.strategy,
                        '状态': stage_status,
                        '得分': score if isinstance(score, (int, float)) else 'N/A',
                        '剔除原因': stage_reason if stage_status == '剔除' else '',
                        '该轮输入特征数': len(all_features),
                        '该轮输出特征数': len(result['selected']),
                        '该轮剔除特征数': len(all_features) - len(result['selected']),
                        '最终状态': final_status,
                        '最终剔除原因': final_reason if final_status == '剔除' else '',
                    })
        
        df = pd.DataFrame(records)
        
        # 添加汇总信息行
        summary_records = []
        for i, item in enumerate(self.selectors):
            if isinstance(item, tuple) and len(item) == 2:
                selector_name = item[0]
                selector = item[1]
            else:
                selector_name = item.__class__.__name__
                selector = item
            
            report = selector.get_selection_report() if hasattr(selector, 'get_selection_report') else {}
            if isinstance(report, dict):
                summary_records.append({
                    '特征': '[SUMMARY]',
                    '筛选轮次': i + 1,
                    '筛选器': selector_name,
                    '筛选器类型': selector.__class__.__name__,
                    '策略': self.strategy,
                    '状态': '汇总',
                    '得分': '',
                    '剔除原因': '',
                    '该轮输入特征数': report.get('输入特征数', 'N/A'),
                    '该轮输出特征数': report.get('选中特征数', 'N/A'),
                    '该轮剔除特征数': report.get('剔除特征数', 'N/A'),
                    '阈值': report.get('阈值', 'N/A'),
                })
        
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            df = pd.concat([df, summary_df], ignore_index=True)
        
        return df
