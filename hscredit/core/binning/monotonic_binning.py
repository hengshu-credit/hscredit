"""单调性约束分箱算法.

基于单调性约束的分箱方法，支持多种单调性模式（参考optbinning实现）：
- 单调递增 (ascending)
- 单调递减 (descending)
- U型/谷值 (valley) - 先减后增
- 倒U型/峰值 (peak) - 先增后减
- 凸函数 (convex) - 二阶导数>=0
- 凹函数 (concave) - 二阶导数<=0
- 自动检测 (auto)

适用于金融风控评分卡场景，满足监管和业务对单调性的各种要求。
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from .base import BaseBinning


class MonotonicBinning(BaseBinning):
    """单调性约束分箱算法 - 支持U型和倒U型.

    通过初始分箱后合并相邻箱，确保坏样本率或WOE值满足指定的单调性约束。
    支持多种单调性模式，包括递增、递减、峰值、谷值、凸函数、凹函数等。
    参考optbinning的monotonic_trend参数实现。

    :param monotonic: 单调性约束类型，默认为'auto'
        - 'auto': 自动检测最佳趋势（允许单增、单减、正U、倒U）
        - 'auto_asc_desc': 自动检测，但只允许单增或单减
        - 'auto_heuristic': 使用启发式方法自动检测
        - 'ascending': 强制坏样本率递增
        - 'descending': 强制坏样本率递减
        - 'peak': 倒U型/峰值（先增后减）
        - 'valley': U型/谷值（先减后增）
        - 'convex': 凸函数（U型近似）
        - 'concave': 凹函数（倒U型近似）
        - 'peak_heuristic': 使用启发式方法检测峰值
        - 'valley_heuristic': 使用启发式方法检测谷值
        - False/None: 不强制单调性
    :param init_method: 初始分箱方法，默认为'quantile'
        - 'quantile': 等频分箱
        - 'uniform': 等距分箱
    :param init_n_bins: 初始分箱数，默认为20
    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False

    **属性**

    - splits_: 每个特征的分箱切分点
    - n_bins_: 每个特征的实际分箱数
    - bin_tables_: 每个特征的分箱统计表
    - monotonic_trend_: 每个特征检测到的单调趋势

    **示例**

    >>> from hscredit.core.binning import MonotonicBinning
    >>> # 峰值模式（倒U型）
    >>> binner = MonotonicBinning(monotonic='peak', max_n_bins=5)
    >>> binner.fit(X, y)
    >>>
    >>> # 谷值模式（U型）
    >>> binner = MonotonicBinning(monotonic='valley', max_n_bins=5)
    >>> binner.fit(X, y)
    >>>
    >>> # 自动检测
    >>> binner = MonotonicBinning(monotonic='auto', max_n_bins=5)
    >>> binner.fit(X, y)
    >>> print(f"检测到的模式: {binner.monotonic_trend_}")
    """

    # 支持的单调性模式（参考optbinning的monotonic_trend参数）
    VALID_MONOTONIC_MODES = [
        'auto', 'auto_asc_desc', 'auto_heuristic',
        'ascending', 'descending',
        'peak', 'valley', 'convex', 'concave',
        'peak_heuristic', 'valley_heuristic',
        None, False
    ]

    def __init__(
        self,
        target: str = 'target',
        monotonic: Union[bool, str, None] = 'auto',
        init_method: str = 'quantile',
        init_n_bins: int = 20,
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
        decimal: int = 4,
    ):
        super().__init__(
            target=target,
            min_n_bins=min_n_bins,
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_bad_rate=min_bad_rate,
            monotonic=monotonic,
            special_codes=special_codes,
            missing_separate=missing_separate,
            random_state=random_state,
            verbose=verbose,
            decimal=decimal,
        )
        self.init_method = init_method
        self.init_n_bins = max(init_n_bins, max_n_bins * 3)
        self.monotonic_trend_ = {}

        # 验证参数
        if init_method not in ['quantile', 'uniform']:
            raise ValueError("init_method必须是'quantile'或'uniform'")

        if monotonic not in self.VALID_MONOTONIC_MODES:
            raise ValueError(
                f"monotonic必须是以下之一: {self.VALID_MONOTONIC_MODES}"
            )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'MonotonicBinning':
        """拟合单调性约束分箱.

        :param X: 训练数据，shape (n_samples, n_features)
        :param y: 目标变量，二分类 (0/1)
        :param kwargs: 其他参数
        :return: 拟合后的分箱器
        """
        # 检查输入数据
        X, y = self._check_input(X, y)

        # 对每个特征进行分箱
        for feature in X.columns:
            if self.verbose:
                print(f"处理特征: {feature}")

            # 检测特征类型
            feature_type = self._detect_feature_type(X[feature])
            self.feature_types_[feature] = feature_type

            if feature_type == 'categorical':
                # 类别型特征：按坏样本率排序后处理
                splits = self._fit_categorical(X[feature], y)
                self.splits_[feature] = splits
            else:
                # 数值型特征：初始分箱 + 单调性调整
                splits = self._fit_numerical(X[feature], y)
                self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1 if isinstance(splits, np.ndarray) else len(splits)

            # 计算分箱统计信息
            bins = self._apply_bins(X[feature], splits)
            self.bin_tables_[feature] = self._compute_bin_stats(
                feature, X[feature], y, bins
            )

        self._is_fitted = True
        return self

    def _fit_numerical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> np.ndarray:
        """对数值型特征进行单调性约束分箱.

        :param x: 特征数据
        :param y: 目标变量
        :return: 切分点数组
        """
        # 处理缺失值和特殊值
        x_clean = x.copy()
        mask = x_clean.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x_clean != code)

        x_valid = x_clean[mask]
        y_valid = y[mask]

        if len(x_valid) == 0:
            return np.array([])

        # 步骤1: 初始分箱
        init_splits = self._get_initial_splits(x_valid, y_valid)

        if len(init_splits) == 0:
            return np.array([])

        # 步骤2: 确定单调性模式
        monotonic_mode = self._detect_monotonic_mode(x_valid, y_valid, init_splits)
        self.monotonic_trend_[x.name] = monotonic_mode

        if self.verbose:
            print(f"  检测到的单调性模式: {monotonic_mode}")

        # 步骤3: 根据单调性模式进行分箱优化
        if monotonic_mode in ['peak', 'valley']:
            final_splits = self._ensure_peak_valley(
                x_valid, y_valid, init_splits, monotonic_mode
            )
        elif monotonic_mode in ['convex', 'concave']:
            final_splits = self._ensure_convex_concave(
                x_valid, y_valid, init_splits, monotonic_mode
            )
        else:
            final_splits = self._ensure_monotonic(
                x_valid, y_valid, init_splits, monotonic_mode
            )

        return final_splits

    def _get_initial_splits(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> np.ndarray:
        """获取初始分箱切分点.

        :param x: 特征数据
        :param y: 目标变量
        :return: 初始切分点数组
        """
        # 使用等频或等距分箱作为初始分箱
        if self.init_method == 'quantile':
            # 等频分箱
            n_splits = min(self.init_n_bins - 1, len(x) // 10)
            n_splits = max(n_splits, self.min_n_bins - 1)

            if n_splits <= 0:
                return np.array([])

            quantiles = np.linspace(0, 1, n_splits + 2)
            splits = np.percentile(x, quantiles[1:-1] * 100)
        else:
            # 等距分箱
            x_min, x_max = x.min(), x.max()
            n_splits = min(self.init_n_bins - 1, len(x) // 10)
            n_splits = max(n_splits, self.min_n_bins - 1)

            if n_splits <= 0 or x_min == x_max:
                return np.array([])

            splits = np.linspace(x_min, x_max, n_splits + 2)[1:-1]

        # 去重并排序
        splits = np.unique(splits)
        return splits

    def _detect_monotonic_mode(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray
    ) -> str:
        """检测单调性模式（参考optbinning实现）.

        支持auto和auto_asc_desc两种自动检测模式：
        - auto: 允许单增、单减、peak（倒U）、valley（正U）
        - auto_asc_desc: 只允许单增或单减

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :return: 检测到的单调性模式
        """
        # 如果用户指定了具体模式，直接使用
        if self.monotonic in ['ascending', 'descending', 'peak', 'valley', 'convex', 'concave',
                             'peak_heuristic', 'valley_heuristic']:
            return self.monotonic
        elif self.monotonic is False or self.monotonic is None:
            return 'none'

        # 自动检测模式：auto, auto_asc_desc, auto_heuristic
        if len(splits) == 0:
            return 'ascending'

        # 计算初始分箱的坏样本率
        bins = pd.cut(x, bins=[-np.inf] + splits.tolist() + [np.inf], labels=False)
        temp_df = pd.DataFrame({'bin': bins, 'target': y})
        bin_stats = temp_df.groupby('bin')['target'].agg(['mean', 'count'])

        min_samples = self._get_min_samples(len(x))
        bin_stats = bin_stats[bin_stats['count'] >= min_samples]

        if len(bin_stats) < 3:
            # 分箱数太少，无法判断复杂模式，使用简单单调性
            asc_score = sum(1 for i in range(len(bin_stats)-1)
                          if bin_stats['mean'].iloc[i] <= bin_stats['mean'].iloc[i+1])
            desc_score = sum(1 for i in range(len(bin_stats)-1)
                           if bin_stats['mean'].iloc[i] >= bin_stats['mean'].iloc[i+1])
            return 'ascending' if asc_score >= desc_score else 'descending'

        bad_rates = bin_stats['mean'].values
        n_prebins = len(bad_rates)

        # 计算统计特征（参考optbinning的auto_monotonic_data）
        # 1. 趋势变化次数
        n_trend_changes = self._n_peaks_valleys(bad_rates)
        p_trend_changes = n_trend_changes / n_prebins

        # 2. 线性回归系数方向
        lr_coef = np.polyfit(np.arange(n_prebins), bad_rates, deg=1)[0]
        lr_sense = int(lr_coef > 0)  # 1 for ascending, 0 for descending

        # 3. 峰值/谷值位置
        pos_min = np.argmin(bad_rates)
        pos_max = np.argmax(bad_rates)

        # 4. 极值点统计
        n1 = n_prebins - 1
        p_bins_min_left = pos_min / n1 if n1 > 0 else 0
        p_bins_min_right = (n_prebins - pos_min - 1) / n1 if n1 > 0 else 0
        p_bins_max_left = pos_max / n1 if n1 > 0 else 0
        p_bins_max_right = (n_prebins - pos_max - 1) / n1 if n1 > 0 else 0

        total_records = bin_stats['count'].sum()
        p_records_min_left = bin_stats['count'].iloc[:pos_min].sum() / total_records if pos_min > 0 else 0
        p_records_min_right = bin_stats['count'].iloc[pos_min+1:].sum() / total_records if pos_min < n_prebins - 1 else 0
        p_records_max_left = bin_stats['count'].iloc[:pos_max].sum() / total_records if pos_max > 0 else 0
        p_records_max_right = bin_stats['count'].iloc[pos_max+1:].sum() / total_records if pos_max < n_prebins - 1 else 0

        # 5. 极值点三角形面积比例
        p_area = self._extreme_points_area(bad_rates)

        # 根据模式选择决策逻辑
        if self.monotonic == 'auto_asc_desc':
            # auto_asc_desc模式：只允许单增或单减，优先使用整体相关方向，避免误判塌缩成2箱
            corr = pd.Series(x).corr(pd.Series(y), method='spearman')
            if pd.notna(corr) and abs(corr) >= 0.02:
                return 'ascending' if corr > 0 else 'descending'
            return self._auto_asc_desc_decision(
                p_trend_changes, lr_sense,
                p_records_min_left, p_records_min_right,
                p_records_max_left, p_records_max_right,
                p_area
            )
        else:
            # auto/auto_heuristic模式：允许单增、单减、peak、valley
            return self._auto_decision(
                lr_sense,
                p_records_min_left, p_records_min_right,
                p_records_max_left, p_records_max_right,
                p_area
            )

    def _n_peaks_valleys(self, x: np.ndarray) -> int:
        """计算数组中峰值和谷值的数量（趋势变化次数）.

        :param x: 数据数组
        :return: 趋势变化次数
        """
        if len(x) < 3:
            return 0
        diff_sign = np.sign(x[1:] - x[:-1])
        return np.count_nonzero(diff_sign[1:] != diff_sign[:-1])

    def _extreme_points_area(self, x: np.ndarray) -> float:
        """计算极值点三角形面积与总矩形面积的比例.

        参考optbinning的extreme_points_area实现。

        :param x: 数据数组
        :return: 面积比例
        """
        n = len(x)
        if n < 3:
            return 0.0

        pos_min = np.argmin(x)
        pos_max = np.argmax(x)

        x_iter = x[1:-1]
        if len(x_iter) == 0:
            return 0.0

        xinit, xmin, xmax, xlast = 0, pos_min, pos_max, n
        yinit, ymin, ymax, ylast = x[0], x[pos_min], x[pos_max], x[-1]

        # 两个三角形面积
        triangle1 = np.array([[xinit, xmin, xmax], [yinit, ymin, ymax], [1, 1, 1]])
        triangle2 = np.array([[xmin, xmax, xlast], [ymin, ymax, ylast], [1, 1, 1]])

        area_1 = 0.5 * np.abs(np.linalg.det(triangle1))
        area_2 = 0.5 * np.abs(np.linalg.det(triangle2))
        sum_area = area_1 + area_2

        p_area = sum_area / ((ymax - ymin) * n) if (ymax - ymin) > 0 else 0
        return p_area

    def _auto_decision(self, lr_sense: int,
                       p_records_min_left: float, p_records_min_right: float,
                       p_records_max_left: float, p_records_max_right: float,
                       p_area: float) -> str:
        """auto模式决策逻辑（允许单增、单减、peak、valley）.

        简化版决策树，参考optbinning的auto_monotonic_decision。

        :param lr_sense: 线性回归方向 (0=descending, 1=ascending)
        :param p_records_min_left: 谷值左侧样本比例
        :param p_records_min_right: 谷值右侧样本比例
        :param p_records_max_left: 峰值左侧样本比例
        :param p_records_max_right: 峰值右侧样本比例
        :param p_area: 极值点面积比例
        :return: 检测到的单调性模式
        """
        # 基于面积和极值点位置的决策
        if p_area <= 0.25:
            # 低面积比例，趋势较平缓
            if lr_sense == 0:
                # 整体下降趋势
                if p_records_min_right <= 0.05:
                    return 'descending'
                else:
                    return 'valley'
            else:
                return 'ascending'
        else:
            # 高面积比例，有明显极值点
            if p_records_min_right <= 0.1:
                if lr_sense == 0:
                    return 'descending'
                else:
                    return 'ascending'
            else:
                # 检查peak或valley
                if p_records_max_left > 0.1 and p_records_max_right > 0.1:
                    # 峰值在中间
                    return 'peak'
                elif p_records_min_left > 0.1 and p_records_min_right > 0.1:
                    # 谷值在中间
                    return 'valley'
                else:
                    # 默认根据线性趋势
                    return 'ascending' if lr_sense == 1 else 'descending'

    def _auto_asc_desc_decision(self, p_trend_changes: float, lr_sense: int,
                                p_records_min_left: float, p_records_min_right: float,
                                p_records_max_left: float, p_records_max_right: float,
                                p_area: float) -> str:
        """auto_asc_desc模式决策逻辑（只允许单增或单减）.

        参考optbinning的auto_monotonic_asc_desc_decision。

        :param p_trend_changes: 趋势变化比例
        :param lr_sense: 线性回归方向
        :param p_records_min_left: 谷值左侧样本比例
        :param p_records_min_right: 谷值右侧样本比例
        :param p_records_max_left: 峰值左侧样本比例
        :param p_records_max_right: 峰值右侧样本比例
        :param p_area: 极值点面积比例
        :return: 'ascending' 或 'descending'
        """
        if lr_sense == 0:
            # 整体下降趋势
            if p_area <= 0.5:
                if p_records_max_right <= 0.05:
                    return 'descending'
                else:
                    return 'ascending'
            else:
                return 'descending'
        else:
            # 整体上升趋势
            if p_records_max_left <= 0.05:
                return 'ascending'
            else:
                if p_records_min_left <= 0.8:
                    if p_area <= 0.5:
                        return 'descending'
                    else:
                        return 'ascending'
                else:
                    if p_trend_changes <= 0.5:
                        return 'descending'
                    else:
                        return 'ascending'

    def _is_peak_pattern(self, x: np.ndarray) -> bool:
        """检查是否为峰值模式（倒U型）.

        峰值模式：先递增后递减

        :param x: 坏样本率数组
        :return: 是否为峰值模式
        """
        if len(x) < 3:
            return False

        t = np.argmax(x)
        if t == 0 or t == len(x) - 1:
            return False

        # 检查前半部分递增，后半部分递减
        left_asc = np.all(x[1:t+1] - x[:t] >= -1e-10)
        right_desc = np.all(x[t+1:] - x[t:-1] <= 1e-10)

        return left_asc and right_desc

    def _is_valley_pattern(self, x: np.ndarray) -> bool:
        """检查是否为谷值模式（U型）.

        谷值模式：先递减后递增

        :param x: 坏样本率数组
        :return: 是否为谷值模式
        """
        if len(x) < 3:
            return False

        t = np.argmin(x)
        if t == 0 or t == len(x) - 1:
            return False

        # 检查前半部分递减，后半部分递增
        left_desc = np.all(x[1:t+1] - x[:t] <= 1e-10)
        right_asc = np.all(x[t+1:] - x[t:-1] >= -1e-10)

        return left_desc and right_asc

    def _is_convex(self, x: np.ndarray) -> bool:
        """检查是否为凸函数（二阶导数>=0）.

        :param x: 坏样本率数组
        :return: 是否为凸函数
        """
        n = len(x)
        if n < 3:
            return False

        for i in range(1, n - 1):
            if x[i+1] - 2 * x[i] + x[i-1] < -1e-10:
                return False
        return True

    def _is_concave(self, x: np.ndarray) -> bool:
        """检查是否为凹函数（二阶导数<=0）.

        :param x: 坏样本率数组
        :return: 是否为凹函数
        """
        n = len(x)
        if n < 3:
            return False

        for i in range(1, n - 1):
            if -x[i+1] + 2 * x[i] - x[i-1] < -1e-10:
                return False
        return True

    def _ensure_peak_valley(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray,
        mode: str
    ) -> np.ndarray:
        """确保分箱满足峰值或谷值模式.

        策略：
        1. 找到峰值/谷值位置
        2. 分别对左右两部分应用单调性约束
        3. 合并结果

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :param mode: 'peak' 或 'valley'
        :return: 调整后的切分点
        """
        if len(splits) == 0:
            return splits

        splits = list(splits)
        min_samples = self._get_min_samples(len(x))

        max_iter = 100
        for iteration in range(max_iter):
            if len(splits) < 2:
                break

            # 计算当前分箱的坏样本率
            bins = pd.cut(x, bins=[-np.inf] + splits + [np.inf], labels=False)
            temp_df = pd.DataFrame({'bin': bins, 'target': y})
            bin_stats = temp_df.groupby('bin').agg({
                'target': ['mean', 'count', 'sum']
            }).reset_index()
            bin_stats.columns = ['bin', 'bad_rate', 'count', 'bad']
            
            # 确保所有分箱都在bin_stats中
            bin_stats = self._ensure_all_bins_in_stats(bin_stats, len(splits) + 1)

            bad_rates = bin_stats['bad_rate'].values

            # 找到峰值/谷值位置
            if mode == 'peak':
                peak_idx = np.argmax(bad_rates)
            else:  # valley
                peak_idx = np.argmin(bad_rates)

            # 检查是否满足模式
            if self._check_peak_valley(bad_rates, mode, peak_idx):
                break

            # 找到违反约束的位置
            violation_idx = self._find_peak_valley_violation(bad_rates, mode, peak_idx)

            if violation_idx is None or violation_idx >= len(splits):
                break

            # 合并切分点
            merged_count = bin_stats.iloc[violation_idx]['count'] + bin_stats.iloc[violation_idx + 1]['count']

            if merged_count >= min_samples:
                splits.pop(violation_idx)
            else:
                # 尝试其他位置
                merged = False
                for i in range(len(splits)):
                    test_count = bin_stats.iloc[i]['count'] + bin_stats.iloc[i + 1]['count']
                    if test_count >= min_samples:
                        splits.pop(i)
                        merged = True
                        break

                if not merged:
                    break

            if len(splits) + 1 <= self.min_n_bins:
                break

        # 最终检查分箱数
        if len(splits) + 1 > self.max_n_bins:
            splits = self._reduce_bins_peak_valley(x, y, splits, mode)

        return np.array(splits)

    def _check_peak_valley(
        self,
        bad_rates: np.ndarray,
        mode: str,
        peak_idx: int
    ) -> bool:
        """检查是否满足峰值/谷值模式.

        :param bad_rates: 坏样本率数组
        :param mode: 'peak' 或 'valley'
        :param peak_idx: 峰值/谷值索引
        :return: 是否满足模式
        """
        if mode == 'peak':
            # 峰值前递增，峰值后递减
            left_ok = all(bad_rates[i] <= bad_rates[i+1] + 1e-10
                         for i in range(peak_idx))
            right_ok = all(bad_rates[i] >= bad_rates[i+1] - 1e-10
                          for i in range(peak_idx, len(bad_rates)-1))
            return left_ok and right_ok
        else:  # valley
            # 谷值前递减，谷值后递增
            left_ok = all(bad_rates[i] >= bad_rates[i+1] - 1e-10
                         for i in range(peak_idx))
            right_ok = all(bad_rates[i] <= bad_rates[i+1] + 1e-10
                          for i in range(peak_idx, len(bad_rates)-1))
            return left_ok and right_ok

    def _find_peak_valley_violation(
        self,
        bad_rates: np.ndarray,
        mode: str,
        peak_idx: int
    ) -> Optional[int]:
        """找到违反峰值/谷值模式的位置.

        :param bad_rates: 坏样本率数组
        :param mode: 'peak' 或 'valley'
        :param peak_idx: 峰值/谷值索引
        :return: 违反位置的索引，如果没有违反返回None
        """
        if mode == 'peak':
            # 检查峰值前是否递增
            for i in range(peak_idx):
                if bad_rates[i] > bad_rates[i+1] + 1e-10:
                    return i
            # 检查峰值后是否递减
            for i in range(peak_idx, len(bad_rates) - 1):
                if bad_rates[i] < bad_rates[i+1] - 1e-10:
                    return i
        else:  # valley
            # 检查谷值前是否递减
            for i in range(peak_idx):
                if bad_rates[i] < bad_rates[i+1] - 1e-10:
                    return i
            # 检查谷值后是否递增
            for i in range(peak_idx, len(bad_rates) - 1):
                if bad_rates[i] > bad_rates[i+1] + 1e-10:
                    return i
        return None

    def _reduce_bins_peak_valley(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: List,
        mode: str
    ) -> List:
        """减少分箱数同时尽量保持峰值/谷值模式.

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 当前切分点
        :param mode: 'peak' 或 'valley'
        :return: 调整后的切分点
        """
        min_samples = self._get_min_samples(len(x))

        while len(splits) + 1 > self.max_n_bins and len(splits) > 0:
            bins = pd.cut(x, bins=[-np.inf] + splits + [np.inf], labels=False)
            temp_df = pd.DataFrame({'bin': bins, 'target': y})
            bin_stats = temp_df.groupby('bin').agg({
                'target': ['mean', 'count', 'sum']
            }).reset_index()
            bin_stats.columns = ['bin', 'bad_rate', 'count', 'bad']
            
            # 确保所有分箱都在bin_stats中
            bin_stats = self._ensure_all_bins_in_stats(bin_stats, len(splits) + 1)

            # 选择对模式影响最小的合并
            iv_losses = []
            for i in range(len(splits)):
                merged_count = bin_stats.iloc[i]['count'] + bin_stats.iloc[i+1]['count']
                if merged_count >= min_samples:
                    iv_loss = abs(bin_stats.iloc[i]['bad_rate'] - bin_stats.iloc[i+1]['bad_rate'])
                    iv_losses.append((i, iv_loss))

            if not iv_losses:
                break

            iv_losses.sort(key=lambda x: x[1])
            merge_idx = iv_losses[0][0]
            splits.pop(merge_idx)

        return splits

    def _ensure_convex_concave(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray,
        mode: str
    ) -> np.ndarray:
        """确保分箱满足凸函数或凹函数模式.

        简化处理：对于凸/凹函数，使用峰值/谷值近似

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :param mode: 'convex' 或 'concave'
        :return: 调整后的切分点
        """
        if mode == 'convex':
            # 凸函数近似为U型（valley）
            return self._ensure_peak_valley(x, y, splits, 'valley')
        else:  # concave
            # 凹函数近似为倒U型（peak）
            return self._ensure_peak_valley(x, y, splits, 'peak')

    def _ensure_monotonic(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray,
        direction: str
    ) -> np.ndarray:
        """确保分箱满足单调性约束.

        核心算法：合并相邻箱直到满足单调性约束。

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 初始切分点
        :param direction: 单调性方向 'ascending', 'descending' 或 'none'
        :return: 调整后的切分点
        """
        if direction == 'none' or len(splits) == 0:
            return splits

        splits = list(splits)
        min_samples = self._get_min_samples(len(x))

        max_iter = 100
        for iteration in range(max_iter):
            if len(splits) == 0:
                break

            bins = pd.cut(x, bins=[-np.inf] + splits + [np.inf], labels=False)
            temp_df = pd.DataFrame({'bin': bins, 'target': y})
            bin_stats = temp_df.groupby('bin').agg({
                'target': ['mean', 'count', 'sum']
            }).reset_index()
            bin_stats.columns = ['bin', 'bad_rate', 'count', 'bad']
            
            # 确保所有分箱都在bin_stats中
            bin_stats = self._ensure_all_bins_in_stats(bin_stats, len(splits) + 1)

            bad_rates = bin_stats['bad_rate'].values

            if self._check_monotonic_simple(bad_rates, direction):
                break

            violation_idx = self._find_violation_simple(bad_rates, direction)

            if violation_idx is None or violation_idx >= len(splits):
                break

            merged_count = bin_stats.iloc[violation_idx]['count'] + bin_stats.iloc[violation_idx + 1]['count']

            if merged_count >= min_samples:
                splits.pop(violation_idx)
            else:
                merged = False
                for i in range(len(splits)):
                    test_count = bin_stats.iloc[i]['count'] + bin_stats.iloc[i + 1]['count']
                    if test_count >= min_samples:
                        splits.pop(i)
                        merged = True
                        break

                if not merged:
                    break

            if len(splits) + 1 <= self.min_n_bins:
                break

        if len(splits) + 1 > self.max_n_bins:
            splits = self._reduce_bins_simple(x, y, splits, direction)

        return np.array(splits)

    def _check_monotonic_simple(
        self,
        bad_rates: np.ndarray,
        direction: str
    ) -> bool:
        """简单单调性检查.

        :param bad_rates: 坏样本率数组
        :param direction: 单调性方向
        :return: 是否满足单调性
        """
        if direction == 'ascending':
            return all(bad_rates[i] <= bad_rates[i+1] + 1e-10
                      for i in range(len(bad_rates)-1))
        elif direction == 'descending':
            return all(bad_rates[i] >= bad_rates[i+1] - 1e-10
                      for i in range(len(bad_rates)-1))
        return True

    def _find_violation_simple(
        self,
        bad_rates: np.ndarray,
        direction: str
    ) -> Optional[int]:
        """找到违反单调性的位置.

        :param bad_rates: 坏样本率数组
        :param direction: 单调性方向
        :return: 违反位置的索引，如果没有违反返回None
        """
        if direction == 'ascending':
            for i in range(len(bad_rates) - 1):
                if bad_rates[i] > bad_rates[i+1] + 1e-10:
                    return i
        elif direction == 'descending':
            for i in range(len(bad_rates) - 1):
                if bad_rates[i] < bad_rates[i+1] - 1e-10:
                    return i
        return None

    def _reduce_bins_simple(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: List,
        direction: str
    ) -> List:
        """简单减少分箱数.

        :param x: 特征数据
        :param y: 目标变量
        :param splits: 当前切分点
        :param direction: 单调性方向
        :return: 调整后的切分点
        """
        min_samples = self._get_min_samples(len(x))

        while len(splits) + 1 > self.max_n_bins and len(splits) > 0:
            bins = pd.cut(x, bins=[-np.inf] + splits + [np.inf], labels=False)
            temp_df = pd.DataFrame({'bin': bins, 'target': y})
            bin_stats = temp_df.groupby('bin').agg({
                'target': ['mean', 'count', 'sum']
            }).reset_index()
            bin_stats.columns = ['bin', 'bad_rate', 'count', 'bad']
            
            # 确保所有分箱都在bin_stats中
            bin_stats = self._ensure_all_bins_in_stats(bin_stats, len(splits) + 1)

            iv_losses = []
            for i in range(len(splits)):
                merged_count = bin_stats.iloc[i]['count'] + bin_stats.iloc[i+1]['count']
                if merged_count >= min_samples:
                    iv_loss = abs(bin_stats.iloc[i]['bad_rate'] - bin_stats.iloc[i+1]['bad_rate'])
                    iv_losses.append((i, iv_loss))

            if not iv_losses:
                break

            iv_losses.sort(key=lambda x: x[1])
            merge_idx = iv_losses[0][0]
            splits.pop(merge_idx)

        return splits

    def _fit_categorical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> List:
        """对类别型特征进行单调性约束分箱.

        :param x: 特征数据
        :param y: 目标变量
        :return: 切分点（类别列表）
        """
        x_clean = x.copy()
        mask = x_clean.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x_clean != code)

        x_valid = x_clean[mask]
        y_valid = y[mask]

        cat_stats = pd.DataFrame({
            'category': x_valid,
            'target': y_valid
        }).groupby('category')['target'].agg(['mean', 'count'])

        min_samples = self._get_min_samples(len(x_valid))
        cat_stats = cat_stats[cat_stats['count'] >= min_samples]

        if len(cat_stats) <= self.min_n_bins:
            return cat_stats.sort_values('mean').index.tolist()

        cat_stats = cat_stats.sort_values('mean')
        categories = cat_stats.index.tolist()

        # 类别型特征主要支持简单单调性
        direction = self._detect_monotonic_direction_cat(cat_stats['mean'].values)

        if self.monotonic in ['ascending', 'descending']:
            direction = self.monotonic
        elif self.monotonic in ['peak', 'valley', 'convex', 'concave']:
            # 复杂模式退化为简单单调性
            direction = 'ascending'
        elif self.monotonic is False or self.monotonic is None:
            return categories

        if direction == 'descending':
            categories = categories[::-1]

        if len(categories) > self.max_n_bins:
            categories = self._merge_categories(cat_stats, direction)

        return categories

    def _detect_monotonic_direction_cat(
        self,
        bad_rates: np.ndarray
    ) -> str:
        """检测类别型特征的单调性方向.

        :param bad_rates: 坏样本率数组
        :return: 单调性方向
        """
        if self.monotonic in ['ascending', 'descending']:
            return self.monotonic
        elif self.monotonic is False or self.monotonic is None:
            return 'none'

        # 类别型特征默认使用排序后的顺序（递增）
        return 'ascending'

    def _merge_categories(
        self,
        cat_stats: pd.DataFrame,
        direction: str
    ) -> List:
        """合并类别以满足分箱数限制.

        :param cat_stats: 类别统计
        :param direction: 单调性方向
        :return: 合并后的类别列表
        """
        cat_stats = cat_stats.sort_values('mean')
        categories = cat_stats.index.tolist()

        while len(categories) > self.max_n_bins:
            min_diff = float('inf')
            merge_idx = 0

            bad_rates = cat_stats['mean'].values
            for i in range(len(bad_rates) - 1):
                diff = abs(bad_rates[i] - bad_rates[i+1])
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i

            cat1 = categories[merge_idx]
            cat2 = categories[merge_idx + 1]
            merged_cat = f"{cat1},{cat2}"

            merged_count = cat_stats.iloc[merge_idx]['count'] + cat_stats.iloc[merge_idx + 1]['count']
            merged_bad = cat_stats.iloc[merge_idx]['mean'] * cat_stats.iloc[merge_idx]['count'] + \
                        cat_stats.iloc[merge_idx + 1]['mean'] * cat_stats.iloc[merge_idx + 1]['count']
            merged_rate = merged_bad / merged_count

            categories.pop(merge_idx + 1)
            categories[merge_idx] = merged_cat

            cat_stats = cat_stats.drop([cat1, cat2])
            cat_stats.loc[merged_cat] = {'mean': merged_rate, 'count': merged_count}
            cat_stats = cat_stats.sort_values('mean')

        if direction == 'descending':
            categories = categories[::-1]

        return categories

    def _ensure_all_bins_in_stats(
        self,
        bin_stats: pd.DataFrame,
        n_bins: int
    ) -> pd.DataFrame:
        """确保bin_stats包含所有分箱（即使某些分箱为空）.
        
        :param bin_stats: 分箱统计表
        :param n_bins: 分箱数量
        :return: 补全后的分箱统计表
        """
        expected_bins = list(range(n_bins))
        for bin_idx in expected_bins:
            if bin_idx not in bin_stats['bin'].values:
                bin_stats = pd.concat([bin_stats, pd.DataFrame({
                    'bin': [bin_idx],
                    'bad_rate': [0.0],
                    'count': [0],
                    'bad': [0]
                })], ignore_index=True)
        
        return bin_stats.sort_values('bin').reset_index(drop=True)

    def _get_min_samples(self, n_total: int) -> int:
        """获取最小样本数.

        :param n_total: 总样本数
        :return: 最小样本数
        """
        if self.min_bin_size < 1:
            return int(n_total * self.min_bin_size)
        return int(self.min_bin_size)

    def _apply_bins(
        self,
        x: pd.Series,
        splits: Union[np.ndarray, List]
    ) -> np.ndarray:
        """应用分箱.

        :param x: 特征数据
        :param splits: 切分点
        :return: 分箱索引
        """
        if isinstance(splits, list):
            bins = np.zeros(len(x), dtype=int)
            for i, cat in enumerate(splits):
                if ',' in str(cat):
                    cats = str(cat).split(',')
                    for c in cats:
                        bins[x == c] = i
                else:
                    bins[x == cat] = i
            bins[x.isna()] = -1
            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2
            return bins
        else:
            bins = np.zeros(len(x), dtype=int)

            if self.missing_separate:
                bins[x.isna()] = -1

            if self.special_codes:
                for code in self.special_codes:
                    bins[x == code] = -2

            mask = x.notna()
            if self.special_codes:
                for code in self.special_codes:
                    mask = mask & (x != code)

            if len(splits) > 0:
                bins[mask] = np.digitize(x[mask], splits)
            else:
                bins[mask] = 0

            return bins

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        metric: str = 'indices',
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """应用分箱转换.
        
        将原始特征值转换为分箱索引、分箱标签或WOE值。
        
        :param X: 待转换数据, DataFrame或数组格式
        :param metric: 转换类型, 可选值:
            - 'indices': 返回分箱索引 (0, 1, 2, ...), 用于后续处理
            - 'bins': 返回分箱标签字符串, 用于可视化或报告
            - 'woe': 返回WOE值, 用于逻辑回归建模
        :param kwargs: 其他参数
        :return: 转换后的数据, 格式与输入X相同
        
        :example:
        >>> binner = MonotonicBinning()
        >>> binner.fit(X_train, y_train)
        >>> 
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>> 
        >>> # 获取WOE编码 (用于建模)
        >>> X_woe = binner.transform(X_test, metric='woe')
        """
        if not self._is_fitted:
            raise ValueError("分箱器尚未拟合，请先调用fit方法")

        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                if X.ndim == 1:
                    X = pd.DataFrame(X, columns=['feature'])
                else:
                    X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature not in self.splits_:
                result[feature] = X[feature]
                continue

            splits = self.splits_[feature]
            bins = self._apply_bins(X[feature], splits)

            if metric == 'indices':
                result[feature] = bins
            elif metric == 'bins':
                labels = self._get_bin_labels(splits, bins)
                result[feature] = [labels[b] if b >= 0 else ('missing' if b == -1 else 'special') for b in bins]
            elif metric == 'woe':
                # 优先使用_woe_maps_（从export/load导入）
                if hasattr(self, '_woe_maps_') and feature in self._woe_maps_:
                    woe_map = self._woe_maps_[feature]
                elif feature in self.bin_tables_:
                    bin_table = self.bin_tables_[feature]
                    woe_map = dict(zip(range(len(bin_table)), bin_table['分档WOE值'].values))
                    self._enrich_woe_map(woe_map, bin_table)
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                result[feature] = pd.Series(bins).map(woe_map).values
            else:
                raise ValueError(f"未知的metric: {metric}")

        return result

    def plot_binning(
        self,
        feature: str,
        metric: str = 'bad_rate',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """绘制分箱结果可视化.

        :param feature: 特征名
        :param metric: 绘制的指标，'bad_rate' 或 'woe'
        :param figsize: 图形大小
        :param save_path: 保存路径
        :return: matplotlib图形对象
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("需要安装matplotlib: pip install matplotlib")

        if feature not in self.bin_tables_:
            raise KeyError(f"特征 '{feature}' 未找到")

        table = self.bin_tables_[feature]
        table = table[table['分箱标签'].isin(['缺失', 'special']) == False]

        fig, ax = plt.subplots(figsize=figsize)

        x_pos = range(len(table))
        values = table[metric].values

        # 根据单调性模式设置颜色
        trend = self.monotonic_trend_.get(feature, 'unknown')
        if trend in ['peak', 'concave']:
            colors = ['lightcoral' if i == np.argmax(values) else 'steelblue' for i in range(len(table))]
        elif trend in ['valley', 'convex']:
            colors = ['lightgreen' if i == np.argmin(values) else 'steelblue' for i in range(len(table))]
        else:
            colors = ['steelblue'] * len(table)

        ax.bar(x_pos, values, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Bin', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'Feature: {feature} | Trend: {trend}', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Bin {i}' for i in range(len(table))], rotation=45)

        # 添加数值标签
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax


if __name__ == '__main__':
    # 测试代码
    print("=" * 70)
    print("MonotonicBinning - 支持U型和倒U型的单调性分箱测试")
    print("=" * 70)

    # 生成测试数据
    np.random.seed(42)
    n = 1000

    # 峰值模式数据（倒U型）
    x_peak = np.random.uniform(0, 100, n)
    y_peak_prob = 0.1 + 0.3 * np.exp(-((x_peak - 50) ** 2) / 800)
    y_peak = (np.random.random(n) < y_peak_prob).astype(int)

    # 谷值模式数据（U型）
    x_valley = np.random.uniform(0, 100, n)
    y_valley_prob = 0.4 - 0.3 * np.exp(-((x_valley - 50) ** 2) / 800)
    y_valley_prob = np.clip(y_valley_prob, 0.05, 0.95)
    y_valley = (np.random.random(n) < y_valley_prob).astype(int)

    # 测试1: 峰值模式
    print("\n1. 峰值模式测试 (monotonic='peak'):")
    print("-" * 50)

    X_peak = pd.DataFrame({'peak_feature': x_peak})
    binner_peak = MonotonicBinning(monotonic='peak', max_n_bins=5, verbose=True)
    binner_peak.fit(X_peak, y_peak)

    table = binner_peak.get_bin_table('peak_feature')
    print("\n分箱统计表:")
    print(table[['bin', 'count', 'bad', 'bad_rate', 'woe']])

    bad_rates = table[table['bin'] != 'missing']['bad_rate'].values
    print(f"\n坏样本率分布: {bad_rates}")
    print(f"是否为峰值模式: {binner_peak._is_peak_pattern(bad_rates)}")

    # 测试2: 谷值模式
    print("\n2. 谷值模式测试 (monotonic='valley'):")
    print("-" * 50)

    X_valley = pd.DataFrame({'valley_feature': x_valley})
    binner_valley = MonotonicBinning(monotonic='valley', max_n_bins=5, verbose=True)
    binner_valley.fit(X_valley, y_valley)

    table2 = binner_valley.get_bin_table('valley_feature')
    print("\n分箱统计表:")
    print(table2[['bin', 'count', 'bad', 'bad_rate', 'woe']])

    bad_rates2 = table2[table2['bin'] != 'missing']['bad_rate'].values
    print(f"\n坏样本率分布: {bad_rates2}")
    print(f"是否为谷值模式: {binner_valley._is_valley_pattern(bad_rates2)}")

    # 测试3: 递增模式
    print("\n3. 递增模式测试 (monotonic='ascending'):")
    print("-" * 50)

    x_asc = np.random.uniform(0, 100, n)
    y_asc_prob = 0.05 + 0.4 * (x_asc / 100)
    y_asc = (np.random.random(n) < y_asc_prob).astype(int)

    X_asc = pd.DataFrame({'asc_feature': x_asc})
    binner_asc = MonotonicBinning(monotonic='ascending', max_n_bins=5)
    binner_asc.fit(X_asc, y_asc)

    table3 = binner_asc.get_bin_table('asc_feature')
    print(table3[['bin', 'count', 'bad', 'bad_rate']])

    bad_rates3 = table3[table3['bin'] != 'missing']['bad_rate'].values
    is_asc = all(bad_rates3[i] <= bad_rates3[i+1] for i in range(len(bad_rates3)-1))
    print(f"\n是否递增: {is_asc}")

    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)
