"""WOE (Weight of Evidence) 编码器.

提供直接计算WOE的编码功能，不依赖分箱模块。
"""

from typing import Optional, List, Dict, Union, Any
import numpy as np
import pandas as pd

from .base import BaseEncoder
from ...exceptions import NotFittedError, FeatureNotFoundError


class WOEEncoder(BaseEncoder):
    """WOE (证据权重) 编码器.

    直接对类别特征计算WOE值，不依赖分箱功能。

    WOE计算公式：
    WOE = ln(P(坏样本|类别) / P(好样本|类别)) = ln(坏样本占比/好样本占比)

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列
    :param regularization: 正则化参数，防止除零，默认为1.0
    :param woe_clip: WOE值截断阈值，默认为5.0
        当某个分箱无坏样本或无好样本时，WOE可能变得极大（如±10以上），
        这会导致评分卡中对应分箱的分数异常。
        设置此参数可将WOE限制在[-woe_clip, woe_clip]范围内。
        设置为None则不进行截断。
    :param handle_unknown: transform 时遇到 fit 未见过的类别的处理方式，默认为 ``'value'``：

        - ``'value'``：编码为 0.0（中性 WOE）
        - ``'return_nan'``：编码为 NaN
        - ``'error'``：抛出异常

    :param handle_missing: 缺失值（NaN）的处理方式，默认为 ``'value'``：

        - ``'value'``：编码为 0.0（中性 WOE）
        - ``'return_nan'``：编码为 NaN
        - ``'error'``：fit/transform 遇缺失即抛出异常

    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True
    :param target: scorecardpipeline 风格的目标列名，提供后 fit 时从 X 中提取该列作为 y

    **属性**

    - mapping_: WOE编码映射字典，格式为 {col: {category: woe_value}}
    - iv_: 各特征的IV值，格式为 {col: iv_value}

    **参考样例**

    >>> from hscredit.core.encoders import WOEEncoder
    >>> encoder = WOEEncoder(cols=['category', 'score'])
    >>> X_encoded = encoder.fit_transform(X, y)
    >>> print(encoder.iv_)
    >>>
    >>> # 获取IV摘要
    >>> summary = encoder.summary()
    >>> print(summary)

    导出/加载（与 toad、scorecardpipeline 规则格式互通）::

        >>> rules = encoder.export(to_json='woe_rules.json')
        >>> WOEEncoder().load('woe_rules.json')

    **注意**

    与 :class:`~hscredit.core.binning.BaseBinning` 的 ``metric='woe'`` 不同，本编码器
    直接对原始类别取值计算 WOE、不做数值分箱，适合基数适中的类别特征。``regularization``
    采用加性平滑以避免某类别好/坏样本数为 0 时 WOE 取 ±∞。

    **引用**

    WOE（证据权重）与 IV（信息价值）出自信息论，系统应用于信用评分见
    Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley；
    公式与直观解释参考
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """

    # iv_ 为各特征信息价值，随映射一并序列化以便 import_mapping 后仍可查
    _EXTRA_STATE_ATTRS = ["iv_"]

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        regularization: float = 1.0,
        woe_clip: Optional[float] = 5.0,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
        target: Optional[str] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        """初始化WOE编码器。

        :param cols: 需要编码的列名列表
        :param regularization: 正则化参数，防止除零，默认为1.0
        :param woe_clip: WOE值截断阈值，默认为5.0
            当某个分箱的WOE绝对值超过此阈值时会被截断。
            这可以防止因分箱中无坏样本或无好样本导致的极端WOE值，
            避免评分卡中对应分箱的分数过大。
            设置为None则不进行截断。
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param target: scorecardpipeline风格的目标列名。如果提供，fit时从X中提取该列作为y
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self.regularization = regularization
        self.woe_clip = woe_clip

        self.iv_: Dict[str, float] = {}

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """自动识别需要编码的列。

        WOEEncoder支持数值型和类别型列，因此返回所有列（除了目标列）。

        :param X: 输入数据
        :return: 列名列表
        """
        return X.columns.tolist()

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合WOE编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量，二分类 (0/1)
        :raises ValueError: 当y为空或目标变量不是二元时抛出
        """
        if y is None:
            raise ValueError("WOEEncoder是有监督编码器，必须提供目标变量y")

        y = pd.Series(y).astype(int)

        unique = y.unique()
        if len(unique) != 2:
            raise ValueError(f"目标变量必须是二元的，当前有{len(unique)}个唯一值")
        if not set(unique).issubset({0, 1}):
            raise ValueError("目标变量必须是0和1")

        self._fit_columns(X, y, state_attrs=("mapping_", "iv_"))

    def _fit_column(self, column, values, y=None):
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()
        woe_map, iv = self._fit_categorical(values, y, total_good, total_bad)
        return {"mapping_": woe_map, "iv_": iv}

    def _fit_categorical(
        self, x: pd.Series, y: pd.Series, total_good: int, total_bad: int
    ) -> tuple:
        """拟合类别特征的WOE。

        :param x: 特征列
        :param y: 目标变量
        :param total_good: 好样本总数
        :param total_bad: 坏样本总数
        :return: WOE映射和IV值的元组 (woe_map, iv)
        """
        woe_map = {}

        for category in x.unique():
            if pd.isna(category):
                continue

            mask = x == category
            good_count = (y[mask] == 0).sum()
            bad_count = (y[mask] == 1).sum()

            woe = self._compute_woe(good_count, bad_count, total_good, total_bad)
            woe_map[category] = woe

        if self.handle_missing == 'value':
            woe_map[np.nan] = 0.0
        elif self.handle_missing == 'return_nan':
            woe_map[np.nan] = np.nan

        if self.handle_unknown == 'value':
            woe_map['__UNKNOWN__'] = 0.0
        elif self.handle_unknown == 'return_nan':
            woe_map['__UNKNOWN__'] = np.nan

        iv = self._compute_iv_categorical(x, y, total_good, total_bad)

        return woe_map, iv

    def _compute_woe(
        self, good_count: int, bad_count: int, total_good: int, total_bad: int
    ) -> float:
        """计算WOE值（带正则化和截断）。

        WOE = ln(坏样本占比 / 好样本占比)
        与 toad、scorecardpipeline 及 hscredit 分箱模块保持一致。
        坏样本集中的箱 WOE > 0，好样本集中的箱 WOE < 0，
        LR 系数为正，便于理解。

        当某个分箱无坏样本或无好样本时，WOE值可能变得极大，
        此时会根据 woe_clip 参数进行截断，防止评分卡分数异常。

        :param good_count: 好样本数量
        :param bad_count: 坏样本数量
        :param total_good: 好样本总数
        :param total_bad: 坏样本总数
        :return: WOE值
        """
        good_rate = (good_count + self.regularization) / (total_good + 2 * self.regularization)
        bad_rate = (bad_count + self.regularization) / (total_bad + 2 * self.regularization)

        woe = np.log(bad_rate / good_rate)

        # 截断极端WOE值，防止评分卡分数异常
        if self.woe_clip is not None:
            woe = np.clip(woe, -self.woe_clip, self.woe_clip)

        return woe

    def _compute_iv_categorical(
        self, x: pd.Series, y: pd.Series, total_good: int, total_bad: int
    ) -> float:
        """计算类别特征的IV。

        :param x: 特征列
        :param y: 目标变量
        :param total_good: 好样本总数
        :param total_bad: 坏样本总数
        :return: IV值
        """
        iv = 0.0
        for category in x.dropna().unique():
            mask = x == category
            good_count = (y[mask] == 0).sum()
            bad_count = (y[mask] == 1).sum()

            good_dist = (good_count + self.regularization) / (total_good + 2 * self.regularization)
            bad_dist = (bad_count + self.regularization) / (total_bad + 2 * self.regularization)

            iv += (bad_dist - good_dist) * np.log(bad_dist / good_dist)

        return iv

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据为WOE编码。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选）
        :return: 编码后的数据
        """
        output = self._transform_columns(X, y)
        output.attrs['hscredit_encoding'] = 'woe'
        output.attrs['hscredit_source'] = 'WOEEncoder'
        return output

    def _transform_column(self, column, values, y=None, context=None):
        woe_map = self.mapping_[column]
        mapped = values.map(woe_map)

        # 类型鲁棒回退：对“未命中且非缺失”的原始值，用字符串键再映射一次。
        # 覆盖两类键/输入类型不一致的场景，保证 export→load 往返一致：
        #   1) load() 后 woe_map 为字符串键，而 transform 输入为数值（如 int 100）；
        #   2) fit() 后 woe_map 为数值键，而 transform 输入为数字型字符串 '100'。
        unmapped = mapped.isna() & values.notna()
        if unmapped.any():
            str_map = {
                str(k): v
                for k, v in woe_map.items()
                if k != '__UNKNOWN__' and not (isinstance(k, float) and pd.isna(k))
            }
            if str_map:
                mapped.loc[unmapped] = values.loc[unmapped].astype(str).map(str_map)

        if self.handle_unknown == 'value':
            mapped = mapped.fillna(0.0)
        elif self.handle_unknown == 'error' and mapped.isna().any():
            raise ValueError(f"列'{column}'包含未知类别")
        return mapped

    def get_iv(self) -> Dict[str, float]:
        """获取各特征的IV值。

        :return: 特征名到IV值的映射字典
        """
        return self.iv_

    def get_mapping(self, col: Optional[str] = None) -> Union[Dict, Dict[str, Dict]]:
        """获取WOE编码映射。

        :param col: 列名。如果提供，返回该列的映射；
            如果为None，返回所有列的映射
        :return: WOE映射字典。当 col 指定时返回 {category: woe_value}，
            col 为 None 时返回 {col: {category: woe_value}}
        :raises NotFittedError: 当编码器尚未拟合时抛出
        :raises FeatureNotFoundError: 当指定的 col 不在编码器中时抛出
        """
        if not hasattr(self, 'mapping_') or not self.mapping_:
            raise NotFittedError("WOEEncoder 尚未拟合，请先调用 fit 方法")
        if col is None:
            return self.mapping_
        if col not in self.mapping_:
            raise FeatureNotFoundError(f"列 '{col}' 不在编码器中，请检查列名是否正确")
        return self.mapping_[col]

    def summary(self) -> pd.DataFrame:
        """获取 WOE 编码摘要表（按 IV 降序）。

        对每个已编码特征给出 IV 值及对应的预测能力评级，评级阈值为：

        - IV < 0.02：无预测力
        - 0.02 ≤ IV < 0.1：弱预测力
        - 0.1 ≤ IV < 0.3：中等预测力
        - 0.3 ≤ IV < 0.5：强预测力
        - IV ≥ 0.5：超强预测力（需检查是否标签泄漏）

        :return: 含 ``特征`` / ``IV值`` / ``预测能力`` 三列的 DataFrame，按 ``IV值`` 降序；
            未拟合或无特征时返回空 DataFrame

        **参考样例**

        >>> encoder.fit(X, y)
        >>> encoder.summary()
        """
        if not self.iv_:
            return pd.DataFrame()

        summary = []
        for col, iv in self.iv_.items():
            if iv < 0.02:
                power = '无预测力'
            elif iv < 0.1:
                power = '弱预测力'
            elif iv < 0.3:
                power = '中等预测力'
            elif iv < 0.5:
                power = '强预测力'
            else:
                power = '超强预测力(需检查)'

            summary.append({
                '特征': col,
                'IV值': round(iv, 4),
                '预测能力': power,
            })

        return pd.DataFrame(summary).sort_values('IV值', ascending=False)

    def export(self, to_json: Optional[str] = None) -> Dict[str, Dict]:
        """导出WOE编码规则，兼容 toad/scorecardpipeline 格式.

        导出格式与 toad.WOETransformer.export() 和 scorecardpipeline.WOETransformer.export() 保持一致。

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :return: WOE编码规则字典，格式为 {feature: {value: woe_value, ...}, ...}

        **参考样例**

        >>> from hscredit.core.encoders import WOEEncoder
        >>> encoder = WOEEncoder(cols=['category', 'city'])
        >>> encoder.fit(X, y)
        >>>
        >>> # 导出为字典
        >>> rules = encoder.export()
        >>>
        >>> # 导出并保存到 JSON 文件
        >>> rules = encoder.export(to_json='woe_rules.json')

        **与 toad/scorecardpipeline 的兼容性**

        导出的规则可以直接被 toad 和 scorecardpipeline 加载:

        >>> import toad
        >>> transformer = toad.transform.WOETransformer()
        >>> transformer.load(rules)
        >>>
        >>> from scorecardpipeline import WOETransformer
        >>> transformer = WOETransformer()
        >>> transformer.load(rules)
        """
        import json
        
        if not hasattr(self, 'mapping_') or not self.mapping_:
            raise ValueError("WOEEncoder 尚未拟合，请先调用 fit 方法")
        
        # 构建与 toad 兼容的格式: {feature: {value: woe_value}}
        rules = {}
        for col, woe_map in self.mapping_.items():
            # 将 WOE 映射转换为可 JSON 序列化的格式
            col_rules = {}
            for value, woe in woe_map.items():
                # 处理特殊值
                if value == '__UNKNOWN__':
                    continue  # toad 不保存 __UNKNOWN__
                if pd.isna(value):
                    col_rules['nan'] = woe  # toad 使用字符串 'nan'
                else:
                    col_rules[str(value)] = float(woe)
            rules[col] = col_rules
        
        if to_json is not None:
            # 确保目录存在
            import os
            dir_path = os.path.dirname(to_json)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            with open(to_json, 'w', encoding='utf-8') as f:
                json.dump(rules, f, ensure_ascii=False, indent=2)
        
        return rules

    def load(self, from_json: Union[str, Dict], update: bool = False) -> 'WOEEncoder':
        """加载WOE编码规则，兼容 toad/scorecardpipeline 格式.

        从字典或 JSON 文件加载WOE编码规则，支持 toad 和 scorecardpipeline 导出的格式。

        :param from_json: WOE规则字典或 JSON 文件路径
            - 字典: {'category': {'A': 0.5, 'B': -0.3}}
            - 文件路径: 'woe_rules.json'
        :param update: 是否更新现有规则（而非替换），默认为 False
        :return: self，支持链式调用

        **参考样例**

        >>> from hscredit.core.encoders import WOEEncoder
        >>> encoder = WOEEncoder()
        >>>
        >>> # 从字典加载
        >>> rules = {'category': {'A': 0.5, 'B': -0.3}}
        >>> encoder.load(rules)
        >>>
        >>> # 从 JSON 文件加载
        >>> encoder.load('woe_rules.json')
        >>>
        >>> # 更新现有规则
        >>> encoder.load({'new_feature': {'X': 0.2}}, update=True)

        **与 toad/scorecardpipeline 的兼容性**

        可以直接加载 toad 和 scorecardpipeline 导出的规则:

        >>> import toad
        >>> toad_transformer = toad.transform.WOETransformer()
        >>> toad_transformer.fit(df, y)
        >>> rules = toad_transformer.export()
        >>>
        >>> encoder = WOEEncoder()
        >>> encoder.load(rules)
        """
        import json
        
        if isinstance(from_json, str):
            # 从文件加载
            with open(from_json, 'r', encoding='utf-8') as f:
                rules = json.load(f)
        else:
            # 直接使用字典
            rules = from_json
        
        if not update:
            self.mapping_ = {}
            self.cols_ = []
        # 规范化容器：update=True 时若在未 fit 的新实例上调用，
        # cols_/mapping_ 仍为基类初始的 None，下方成员判断会崩溃，这里统一兜底为空容器
        if self.cols_ is None:
            self.cols_ = []
        if self.mapping_ is None:
            self.mapping_ = {}

        # 加载规则
        for col, col_rules in rules.items():
            if col not in self.mapping_:
                self.mapping_[col] = {}
            if col not in self.cols_:
                self.cols_.append(col)

            for value, woe in col_rules.items():
                # toad/scorecardpipeline 用字符串 'nan' 表示缺失键
                if value == 'nan':
                    self.mapping_[col][np.nan] = woe
                else:
                    # 保持原始（字符串）键，不做 int/float 强转：
                    # 否则数字型字符串类别（如城市码/商品码 '100'）会被转成 int，
                    # 与 transform 时的字符串输入不匹配，导致 WOE 全部落到未知值（静默错误）。
                    # 数值型输入的兼容由 _transform 的字符串回退映射保证。
                    self.mapping_[col][value] = woe

            # 添加未知值处理
            if self.handle_unknown == 'value':
                self.mapping_[col]['__UNKNOWN__'] = 0.0
            elif self.handle_unknown == 'return_nan':
                self.mapping_[col]['__UNKNOWN__'] = np.nan

        self._is_fitted = True
        return self
