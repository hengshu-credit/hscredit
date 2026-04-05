"""OR-Tools 运筹规划分箱算法.

基于 Google OR-Tools CP-SAT 求解器的最优化分箱方法。
将分箱问题建模为整数规划问题，求解全局最优解。

算法流程：
1. 问题建模：将分箱定义为整数规划问题
2. 决策变量：每个候选分割点是否被选中
3. 约束条件：分箱数限制、单调性、样本数约束
4. 目标函数：最大化 IV、KS 或自定义指标
5. 求解：使用 CP-SAT 求解器找到全局最优解

依赖：
    pip install ortools
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import warnings

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    warnings.warn(
        "OR-Tools 未安装，ORBinning 将不可用。"
        "请使用 pip install ortools 安装。",
        ImportWarning
    )

from .base import BaseBinning
from hscredit.core.metrics import composite_binning_quality
from hscredit.core.metrics._binning import _composite_binning_quality_components


class ORBinning(BaseBinning):
    """OR-Tools 运筹规划分箱.

    基于 Google OR-Tools CP-SAT 求解器的最优化分箱方法。
    支持多种优化目标和约束条件，能够找到全局最优分箱方案。
    支持自定义目标函数，可实现复合指标优化。

    **参数**

    :param target: 目标变量列名，默认为'target'。在scorecardpipeline风格中使用，
        当fit时只传入df且y为None时，从df中提取该列作为目标变量。
    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
        - 如果 < 1, 表示占比 (如 0.01 表示 1%)
        - 如果 >= 1, 表示绝对数量 (如 100 表示最少100个样本)
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param monotonic: 坏样本率单调性约束，默认为auto
        - False: 不要求单调性
        - True 或 'auto': 自动检测并应用最佳单调方向
        - 'ascending': 强制坏样本率递增
        - 'descending': 强制坏样本率递减
    :param objective: 优化目标，默认为'iv'
        - 'iv': 最大化 IV 值
        - 'ks': 最大化 KS 统计量
        - 'gini': 最大化 Gini 系数
        - 'entropy': 最小化熵（信息增益最大）
        - 'chi2': 最大化卡方统计量
        - 'custom': 使用自定义目标函数（通过 custom_objective 参数传入）
    :param custom_objective: 自定义目标函数，当 objective='custom' 时使用
        - 类型: Callable[[List[Dict], int, int], float]
        - 参数: bin_stats(List[Dict]) 每个箱的统计信息, total_good(int) 总好样本数, total_bad(int) 总坏样本数
        - 返回: float 目标函数值，越大越好（OR-Tools 求解器始终最大化）
        - 箱统计字典包含: 'count', 'good', 'bad', 'bad_rate', 'good_rate', 'lift', 'woe'
    :param n_prebins: 预分箱数量（候选分割点数），默认为20
        - 候选点越多，求解越精确，但计算时间越长
    :param max_candidates: 最大候选分割点数，默认为50
        - 如果唯一值超过此数，将使用分位数采样
    :param time_limit: 求解时间限制（秒），默认为60
        - 超过此时间将返回当前找到的最优解
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :param special_codes: 特殊值列表，默认为None
    :param random_state: 随机种子，默认为None

    **示例**

    sklearn风格 (推荐)::

        >>> from hscredit.core.binning import ORBinning
        >>> # 最大化 IV
        >>> binner = ORBinning(max_n_bins=5, objective='iv', monotonic=True)
        >>> binner.fit(X_train, y_train)
        >>> X_binned = binner.transform(X_test)
        >>> bin_table = binner.get_bin_table('feature_name')

    scorecardpipeline风格 (目标列在DataFrame中)::

        >>> from hscredit.core.binning import ORBinning
        >>> # 初始化时指定目标列名，fit时传入完整DataFrame
        >>> binner = ORBinning(target='target', max_n_bins=5, objective='ks', time_limit=60)
        >>> binner.fit(df)  # df包含特征列和目标列'target'
        >>> X_binned = binner.transform(df.drop(columns=['target']))

    混合风格 (y参数优先)::

        >>> # 即使初始化时指定了target，fit时传入y会优先使用y
        >>> binner = ORBinning(target='target', objective='iv')
        >>> binner.fit(df, y=external_y)  # 使用external_y，忽略df中的'target'列
    >>>
    >>> # 自定义目标：最大化 LIFT + IV
    >>> def custom_obj(bin_stats, total_good, total_bad):
    ...     total_iv = sum(stat.get('woe', 0) * (stat.get('bad_rate', 0) - stat.get('good_rate', 0)) 
    ...                    for stat in bin_stats if 'woe' in stat)
    ...     total_lift = sum(abs(stat.get('lift', 1) - 1) for stat in bin_stats)
    ...     return total_iv + total_lift * 0.1  # IV + 0.1 * LIFT偏差
    >>>
    >>> binner = ORBinning(objective='custom', custom_objective=custom_obj)
    >>> binner.fit(X_train, y_train)

    **注意**

    OR-Tools 分箱的特点:
    1. 能够找到全局最优解（而非贪心算法的局部最优）
    2. 支持复杂的约束条件组合
    3. 支持自定义目标函数，实现复合指标优化
    4. 计算时间较长，适合对分箱质量要求高的场景
    5. 可以设置时间限制，在时间和精度之间权衡

    **自定义目标函数示例**

    1. 最大 LIFT + IV::

        def max_lift_iv(bin_stats, total_good, total_bad):
            total_iv = sum(stat['woe'] * (stat['bad_rate'] - stat['good_rate']) 
                          for stat in bin_stats)
            total_lift = sum(stat['lift'] for stat in bin_stats)
            return total_iv + total_lift * 0.01

    2. 最小 LIFT + IV（LIFT 越接近1越好）::

        def min_lift_iv(bin_stats, total_good, total_bad):
            total_iv = sum(stat['woe'] * (stat['bad_rate'] - stat['good_rate']) 
                          for stat in bin_stats)
            lift_penalty = sum(abs(stat['lift'] - 1) for stat in bin_stats)
            return total_iv - lift_penalty * 0.1  # 减去惩罚项

    3. 最大/最小 LIFT 离1的距离求和 + IV::

        def lift_distance_iv(bin_stats, total_good, total_bad):
            total_iv = sum(stat['woe'] * (stat['bad_rate'] - stat['good_rate']) 
                          for stat in bin_stats)
            max_lift_dist = max(abs(stat['lift'] - 1) for stat in bin_stats)
            min_lift_dist = min(abs(stat['lift'] - 1) for stat in bin_stats)
            return total_iv + (max_lift_dist + min_lift_dist) * 0.1
    """

    def __init__(
        self,
        target: str = 'target',
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = 'auto',
        objective: str = 'iv',
        custom_objective: Optional[callable] = None,
        n_prebins: int = 20,
        max_candidates: int = 50,
        time_limit: int = 60,
        missing_separate: bool = True,
        special_codes: Optional[List] = None,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not ORTOOLS_AVAILABLE:
            raise ImportError(
                "OR-Tools 未安装，无法使用 ORBinning。"
                "请使用 pip install ortools 安装。"
            )
        
        super().__init__(
            target=target,
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_bad_rate=min_bad_rate,
            monotonic=monotonic,
            missing_separate=missing_separate,
            special_codes=special_codes,
            random_state=random_state,
            **kwargs
        )
        
        # 验证优化目标
        valid_objectives = ['iv', 'ks', 'gini', 'entropy', 'chi2', 'custom']
        if objective not in valid_objectives:
            raise ValueError(f"不支持的优化目标: {objective}，可选: {valid_objectives}")
        
        if objective == 'custom' and custom_objective is None:
            raise ValueError("当 objective='custom' 时，必须提供 custom_objective 参数")
        
        self.objective = objective
        self.custom_objective = custom_objective
        self.n_prebins = n_prebins
        self.max_candidates = max_candidates
        self.time_limit = time_limit

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'ORBinning':
        """拟合 OR-Tools 运筹规划分箱.

        支持两种API风格：
        1. sklearn风格: fit(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit(df) - df是完整数据框，目标列名在初始化时通过target参数传入

        优先级规则：如果y不是None，直接使用y（优先）；否则从X中提取target列。

        :param X: 训练数据
            - sklearn风格: 特征矩阵，shape (n_samples, n_features)
            - scorecardpipeline风格: 完整数据框，包含特征列和目标列
        :param y: 目标变量（可选）
            - sklearn风格: 传入目标变量
            - scorecardpipeline风格: 不传，从X中提取
            - 如果传入y，优先使用y而忽略X中的target列
        :return: 拟合后的分箱器
        """
        # 检查输入数据
        X, y = self._check_input(X, y)

        # 对每个特征进行分箱
        for feature in X.columns:
            self._fit_feature(feature, X[feature], y)

        self._apply_post_fit_constraints(X, y, enforce_monotonic=True)
        self._is_fitted = True
        return self

    def _fit_feature(
        self,
        feature: str,
        X: pd.Series,
        y: pd.Series
    ) -> None:
        """对单个特征进行分箱.

        :param feature: 特征名
        :param X: 特征数据
        :param y: 目标变量
        """
        # 检测特征类型
        feature_type = self._detect_feature_type(X)
        self.feature_types_[feature] = feature_type

        # 处理缺失值和特殊值
        missing_mask = X.isna()
        special_mask = pd.Series(False, index=X.index)
        if self.special_codes:
            special_mask = X.isin(self.special_codes)

        # 获取有效数据
        valid_mask = ~(missing_mask | special_mask)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if feature_type == 'categorical':
            # 类别型变量：按目标变量排序后分箱
            splits = self._or_categorical(X_valid, y_valid)
            self.splits_[feature] = np.array(splits) if splits else np.array([])
            self.n_bins_[feature] = len(splits) + 1 if splits else len(X_valid.unique())
        else:
            # 数值型变量：使用 OR-Tools 优化
            splits = self._or_numerical(X_valid, y_valid)
            self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

        # 生成分箱索引
        bins = self._assign_bins(X, feature)

        # 计算分箱统计
        bin_table = self._compute_bin_stats(feature, X, y, bins)
        self.bin_tables_[feature] = bin_table

    def _get_candidate_splits(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> List[float]:
        """获取候选分割点.

        :param X: 特征数据
        :param y: 目标变量
        :return: 候选分割点列表
        """
        x_vals = X.values
        
        # 获取唯一值
        unique_values = np.unique(x_vals)
        
        if len(unique_values) <= self.max_n_bins:
            return []
        
        # 限制候选点数量
        if len(unique_values) > self.max_candidates + 1:
            # 使用分位数选择候选点
            quantiles = np.linspace(0, 1, self.max_candidates + 1)
            candidates = np.percentile(unique_values, quantiles[1:-1] * 100)
        else:
            # 使用相邻唯一值的中点
            candidates = (unique_values[:-1] + unique_values[1:]) / 2
        
        return sorted(candidates.tolist())

    def _or_numerical(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> List[float]:
        """对数值型变量使用 OR-Tools 优化分箱.

        :param X: 特征数据
        :param y: 目标变量
        :return: 最优分割点列表
        """
        x_vals = X.values
        y_vals = y.values
        n_samples = len(x_vals)
        
        if n_samples == 0:
            return []
        
        # 获取候选分割点
        candidates = self._get_candidate_splits(X, y)
        n_candidates = len(candidates)
        
        if n_candidates == 0:
            return []
        
        if n_candidates < self.min_n_bins - 1:
            # 候选点太少，直接返回
            return candidates[:max(0, self.max_n_bins - 1)]
        
        # 排序数据
        sorted_indices = np.argsort(x_vals)
        x_sorted = x_vals[sorted_indices]
        y_sorted = y_vals[sorted_indices]
        
        # 计算总体统计
        total_good = np.sum(y_vals == 0)
        total_bad = np.sum(y_vals == 1)
        
        if total_good == 0 or total_bad == 0:
            return []
        
        # 计算样本数约束
        min_samples = self._get_min_samples(n_samples)
        max_samples = self._get_max_samples(n_samples) if self.max_bin_size else n_samples
        
        # 创建 CP-SAT 模型
        model = cp_model.CpModel()
        
        # 决策变量：每个候选分割点是否被选中
        select = [model.NewBoolVar(f'select_{i}') for i in range(n_candidates)]
        
        # 约束1：选中的分割点数量在 [min_n_bins-1, max_n_bins-1] 范围内
        model.Add(sum(select) >= self.min_n_bins - 1)
        model.Add(sum(select) <= self.max_n_bins - 1)
        
        # 预计算每个候选分割点分割后的箱统计
        bin_stats = self._compute_bin_stats_for_candidates(
            x_sorted, y_sorted, candidates
        )
        
        # 约束2：每箱样本数约束
        for i in range(len(candidates) + 1):
            # 计算第 i 个箱的样本数
            bin_count = self._get_bin_count_var(model, select, bin_stats, i)
            model.Add(bin_count >= min_samples)
            if self.max_bin_size:
                model.Add(bin_count <= max_samples)
        
        # 约束3：单调性约束
        if self.monotonic:
            self._add_monotonic_constraints(model, select, bin_stats)
        
        # 目标函数
        objective_var = self._create_objective(model, select, bin_stats, total_good, total_bad)
        
        # 设置优化方向
        # 注意：OR-Tools 求解器始终最大化目标函数
        # 对于需要最小化的目标（如 entropy），返回负值
        # 对于自定义目标，假设用户已考虑优化方向
        if self.objective == 'entropy':
            model.Minimize(objective_var)
        else:
            model.Maximize(objective_var)
        
        # 先准备启发式候选，避免求解器给出可行但质量较差的结果
        greedy_splits = self._greedy_fallback(x_sorted, y_sorted, candidates)
        greedy_score = self._score_splits(
            x_sorted, y_sorted, total_good, total_bad, greedy_splits
        )
        composite_search_splits = self._search_composite_optimal_splits(
            x_sorted, y_sorted, candidates
        )
        composite_search_score = self._score_splits(
            x_sorted, y_sorted, total_good, total_bad, composite_search_splits
        )
        graph_search_splits = self._search_segment_graph_splits(
            x_sorted, y_sorted, candidates
        )
        graph_search_score = self._score_splits(
            x_sorted, y_sorted, total_good, total_bad, graph_search_splits
        )
        segment_dp_splits = self._search_segment_dp_splits(
            x_sorted, y_sorted, candidates
        )
        segment_dp_score = self._score_splits(
            x_sorted, y_sorted, total_good, total_bad, segment_dp_splits
        )

        # 求解
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = 8
        
        status = solver.Solve(model)
        
        # 提取结果
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            selected_splits = [
                candidates[i] for i in range(n_candidates)
                if solver.Value(select[i]) == 1
            ]
            selected_splits = sorted(selected_splits)
            solver_score = self._score_splits(
                x_sorted, y_sorted, total_good, total_bad, selected_splits
            )
            scored_candidates = [
                (solver_score, selected_splits),
                (greedy_score, greedy_splits),
                (composite_search_score, composite_search_splits),
                (graph_search_score, graph_search_splits),
                (segment_dp_score, segment_dp_splits),
            ]
            return max(scored_candidates, key=lambda item: item[0])[1]
        else:
            warnings.warn(
                f"OR-Tools 求解失败（状态: {status}），使用启发式搜索作为备选",
                UserWarning
            )
            fallback_candidates = [
                (greedy_score, greedy_splits),
                (composite_search_score, composite_search_splits),
                (graph_search_score, graph_search_splits),
                (segment_dp_score, segment_dp_splits),
            ]
            return max(fallback_candidates, key=lambda item: item[0])[1]

    def _compute_bin_stats_for_candidates(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        candidates: List[float]
    ) -> List[Dict]:
        """预计算每个候选分割点相关的箱统计.

        :param x_sorted: 排序后的特征值
        :param y_sorted: 排序后的目标变量
        :param candidates: 候选分割点列表
        :return: 每个候选点的箱统计信息
        """
        n_candidates = len(candidates)
        bin_stats = []
        
        for i in range(n_candidates + 1):
            # 确定第 i 个箱的范围
            if i == 0:
                start_idx = 0
                end_idx = np.searchsorted(x_sorted, candidates[0], side='right')
            elif i == n_candidates:
                start_idx = np.searchsorted(x_sorted, candidates[-1], side='right')
                end_idx = len(x_sorted)
            else:
                start_idx = np.searchsorted(x_sorted, candidates[i-1], side='right')
                end_idx = np.searchsorted(x_sorted, candidates[i], side='right')
            
            # 计算统计
            y_bin = y_sorted[start_idx:end_idx]
            count = len(y_bin)
            bad_count = np.sum(y_bin == 1)
            good_count = count - bad_count
            
            bin_stats.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'count': count,
                'good': good_count,
                'bad': bad_count,
            })
        
        return bin_stats

    def _get_bin_count_var(
        self,
        model: cp_model.CpModel,
        select: List[Any],
        bin_stats: List[Dict],
        bin_idx: int
    ) -> Any:
        """获取指定箱的样本数变量（用于约束）.

        简化处理：使用线性约束近似
        """
        # 返回该箱的固定样本数
        return bin_stats[bin_idx]['count']

    def _add_monotonic_constraints(
        self,
        model: cp_model.CpModel,
        select: List[Any],
        bin_stats: List[Dict]
    ):
        """添加单调性约束.

        简化实现：假设选择连续的分割点
        """
        # 实际实现中需要根据选中的分割点动态计算
        # 这里简化处理，依赖目标函数自然趋向单调
        pass

    def _create_objective(
        self,
        model: cp_model.CpModel,
        select: List[Any],
        bin_stats: List[Dict],
        total_good: int,
        total_bad: int
    ) -> Any:
        """创建目标函数变量.

        :param model: CP-SAT 模型
        :param select: 选择变量
        :param bin_stats: 箱统计
        :param total_good: 总好样本数
        :param total_bad: 总坏样本数
        :return: 目标函数变量
        """
        n_bins = len(bin_stats)
        eps = 1e-10
        
        # 预处理箱统计，计算衍生指标
        processed_stats = self._compute_derived_stats(bin_stats, total_good, total_bad)
        
        if self.objective == 'iv':
            # IV = sum((bad_rate - good_rate) * log(bad_rate / good_rate))
            # 简化为线性目标：最大化 bad_count * good_dist - good_count * bad_dist
            iv_terms = []
            for stat in processed_stats:
                if stat['count'] > 0:
                    # 使用整数运算避免浮点数问题
                    # IV 贡献近似为 (bad/total_bad - good/total_good)^2
                    bad_rate_scaled = stat['bad'] * total_good
                    good_rate_scaled = stat['good'] * total_bad
                    diff = bad_rate_scaled - good_rate_scaled
                    iv_terms.append(diff * diff)
            
            # 使用加权平均
            max_iv = sum(iv_terms) if iv_terms else 1
            scaled_iv = min(max_iv, 1e9)  # 防止溢出
            return model.NewIntVar(0, int(scaled_iv), 'objective')
        
        elif self.objective == 'ks':
            # KS = max|cumulative_bad_rate - cumulative_good_rate|
            # 简化为最大化箱间的差异
            ks_value = model.NewIntVar(0, 10000, 'ks')
            return ks_value
        
        elif self.objective == 'gini':
            # Gini 系数
            gini_value = model.NewIntVar(0, 10000, 'gini')
            return gini_value
        
        elif self.objective == 'entropy':
            # 熵（需要最小化）
            entropy_value = model.NewIntVar(0, 10000, 'entropy')
            return entropy_value
        
        elif self.objective == 'chi2':
            # 卡方统计量
            chi2_value = model.NewIntVar(0, 10000, 'chi2')
            return chi2_value
        
        else:  # custom
            # 自定义目标函数 - 使用启发式近似
            # CP-SAT 求解器需要线性约束，这里使用启发式评分
            custom_value = model.NewIntVar(-1000000, 1000000, 'custom')
            return custom_value
    
    def _compute_derived_stats(
        self,
        bin_stats: List[Dict],
        total_good: int,
        total_bad: int
    ) -> List[Dict]:
        """计算衍生统计指标.
        
        为每个箱计算 LIFT、WOE、bad_rate、good_rate 等指标.
        
        :param bin_stats: 原始箱统计
        :param total_good: 总好样本数
        :param total_bad: 总坏样本数
        :return: 包含衍生指标的箱统计列表
        """
        eps = 1e-10
        total_samples = total_good + total_bad
        overall_bad_rate = total_bad / total_samples if total_samples > 0 else 0
        
        processed_stats = []
        for stat in bin_stats:
            count = stat['count']
            good = stat['good']
            bad = stat['bad']
            
            if count == 0:
                processed_stats.append({
                    **stat,
                    'bad_rate': 0,
                    'good_rate': 0,
                    'lift': 1.0,
                    'woe': 0,
                })
                continue
            
            bad_rate = bad / count
            good_rate = good / count
            
            # LIFT = 箱内坏样本率 / 总体坏样本率
            lift = bad_rate / overall_bad_rate if overall_bad_rate > eps else 1.0
            
            # WOE = ln(好样本分布 / 坏样本分布)
            good_dist = good / total_good if total_good > 0 else eps
            bad_dist = bad / total_bad if total_bad > 0 else eps
            woe = np.log(good_dist / bad_dist) if good_dist > eps and bad_dist > eps else 0
            
            processed_stats.append({
                **stat,
                'bad_rate': bad_rate,
                'good_rate': good_rate,
                'lift': lift,
                'woe': woe,
            })
        
        return processed_stats

    def _lift_quality_bonus(
        self,
        bin_stats: List[Dict],
        total_good: int,
        total_bad: int
    ) -> float:
        """通过统一 metrics 复合指标评估分箱质量。"""
        bins = []
        for idx, stat in enumerate(bin_stats):
            bins.extend([idx] * int(stat.get('count', 0)))

        if len(bins) == 0:
            return 0.0

        y_parts = []
        for stat in bin_stats:
            good = int(stat.get('good', 0))
            bad = int(stat.get('bad', 0))
            if good > 0:
                y_parts.append(np.zeros(good, dtype=int))
            if bad > 0:
                y_parts.append(np.ones(bad, dtype=int))
        y = np.concatenate(y_parts) if y_parts else np.array([], dtype=int)
        if len(y) == 0:
            return 0.0

        monotonic = self.monotonic if self.monotonic in ['ascending', 'descending', 'peak', 'valley'] else 'descending'
        return composite_binning_quality(
            bins=np.asarray(bins, dtype=int),
            y=y,
            metric='lift',
            monotonic=monotonic,
        )

    def _build_bins_from_splits(
        self,
        x_sorted: np.ndarray,
        splits: List[float]
    ) -> np.ndarray:
        if len(x_sorted) == 0:
            return np.array([], dtype=int)
        return np.searchsorted(np.asarray(sorted(splits), dtype=float), x_sorted, side='right').astype(int)

    def _lift_profile_focus_score(
        self,
        bins: np.ndarray,
        y_sorted: np.ndarray
    ) -> float:
        monotonic = self.monotonic if self.monotonic in ['ascending', 'descending', 'peak', 'valley'] else 'descending'
        comp = _composite_binning_quality_components(
            bins=bins,
            y=y_sorted,
            metric='lift',
            monotonic=monotonic,
        )
        return float(
            comp.get('head_peak_bonus', 0.0) * 2.20
            + comp.get('tail_zero_bonus', 0.0) * 1.80
            - comp.get('tail_collapse_penalty', 0.0) * 1.50
            + comp.get('head_cumulative_gain', 0.0) * 1.10
            + comp.get('spread', 0.0) * 0.18
            + comp.get('marginal_return', 0.0) * 0.55
            - comp.get('marginal_decay_penalty', 0.0) * 0.35
            - comp.get('zero_pairs_penalty', 0.0) * 0.50
        )

    def _score_splits(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        total_good: int,
        total_bad: int,
        splits: List[float]
    ) -> float:
        base_score = self._calculate_objective_score(
            x_sorted, y_sorted, total_good, total_bad, splits
        )
        bins = self._build_bins_from_splits(x_sorted, splits)
        monotonic = self.monotonic if self.monotonic in ['ascending', 'descending', 'peak', 'valley'] else 'descending'
        composite_score = composite_binning_quality(
            bins=bins,
            y=y_sorted,
            metric='lift',
            monotonic=monotonic,
        )
        focus_score = self._lift_profile_focus_score(bins, y_sorted)
        return float(base_score + composite_score * 1.35 + focus_score * 1.15)

    def _segment_composite_score(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        splits: List[float]
    ) -> float:
        bins = self._build_bins_from_splits(x_sorted, splits)
        monotonic = self.monotonic if self.monotonic in ['ascending', 'descending', 'peak', 'valley'] else 'descending'
        comp = _composite_binning_quality_components(
            bins=bins,
            y=y_sorted,
            metric='lift',
            monotonic=monotonic,
        )
        return float(
            comp['quadratic_score']
            + comp.get('head_peak_bonus', 0.0) * 1.50
            + comp['head_cumulative_gain'] * 1.35
            + comp.get('tail_zero_bonus', 0.0) * 1.20
            - comp.get('tail_collapse_penalty', 0.0) * 1.10
            + comp['tail_compression_gain'] * 0.85
            + comp['marginal_return'] * 0.90
            - comp['marginal_decay_penalty'] * 0.35
            + comp['share_floor_bonus'] * 0.20
            - comp['zero_pairs_penalty']
            + comp['monotonic_bonus']
        )

    def _search_segment_dp_splits(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        candidates: List[float]
    ) -> List[float]:
        """方案B第三阶段：段级路径搜索，直接偏向高头部 lift 与低尾部 lift。"""
        total_good = int(np.sum(y_sorted == 0))
        total_bad = int(np.sum(y_sorted == 1))
        n_samples = len(x_sorted)
        if total_good == 0 or total_bad == 0 or n_samples == 0:
            return []

        candidate_pool = sorted(set(candidates))
        if len(candidate_pool) == 0:
            return []

        positions = [int(np.searchsorted(x_sorted, value, side='right')) for value in candidate_pool]
        prefix_bad = np.concatenate([[0], np.cumsum(y_sorted == 1)])
        prefix_good = np.concatenate([[0], np.cumsum(y_sorted == 0)])
        overall_bad_rate = total_bad / max(n_samples, 1)
        eps = 1e-12

        def segment_stats(start: int, end: int) -> Tuple[float, float, float]:
            count = end - start
            if count <= 0:
                return 0.0, 0.0, 0.0
            bad = float(prefix_bad[end] - prefix_bad[start])
            good = float(prefix_good[end] - prefix_good[start])
            bad_rate = bad / count if count > 0 else 0.0
            lift = bad_rate / max(overall_bad_rate, eps) if bad_rate > 0 else 0.0
            share = count / n_samples
            return bad_rate, lift, share

        beam_width = min(12, max(5, len(candidate_pool) // 3))
        states: List[Tuple[float, List[int], int, float]] = [(0.0, [], 0, np.inf)]
        best_score = -np.inf
        best_splits: List[float] = []

        for depth in range(self.max_n_bins - 1):
            next_states: List[Tuple[float, List[int], int, float]] = []
            for partial_score, path, start_idx, last_bad_rate in states:
                for pos_idx in range(start_idx, len(positions)):
                    end = positions[pos_idx]
                    bad_rate, lift, share = segment_stats(0 if len(path) == 0 else positions[path[-1]], end)
                    if end <= (0 if len(path) == 0 else positions[path[-1]]):
                        continue
                    monotonic_penalty = 0.0
                    if self.monotonic in [True, 'auto', 'descending'] and last_bad_rate < np.inf and bad_rate > last_bad_rate + 1e-10:
                        monotonic_penalty = (bad_rate - last_bad_rate) * 8.0
                    segment_gain = (
                        max(lift - 1.0, 0.0) * (2.4 if len(path) == 0 else 0.8)
                        + max(1.0 - lift, 0.0) * (0.15 if depth < self.max_n_bins - 2 else 1.6)
                        + share * (0.45 if len(path) == 0 else 0.1)
                        - monotonic_penalty
                    )
                    next_states.append((partial_score + segment_gain, path + [pos_idx], pos_idx + 1, bad_rate))

            if not next_states:
                break

            next_states = sorted(next_states, key=lambda item: item[0], reverse=True)
            dedup = {}
            for score, path, next_idx, last_bad_rate in next_states:
                dedup.setdefault(tuple(path), (score, next_idx, last_bad_rate))
            states = [
                (score, list(path), next_idx, last_bad_rate)
                for path, (score, next_idx, last_bad_rate) in list(dedup.items())[:beam_width]
            ]

            for _, path, _, _ in states:
                splits = [candidate_pool[idx] for idx in path]
                refined = self._local_refine_splits(
                    x_sorted, y_sorted, total_good, total_bad, splits, candidate_pool
                )
                final_score = self._score_splits(
                    x_sorted, y_sorted, total_good, total_bad, refined
                )
                if final_score > best_score:
                    best_score = final_score
                    best_splits = refined

        return sorted(set(best_splits))

    def _search_segment_graph_splits(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        candidates: List[float]
    ) -> List[float]:
        """方案B第二阶段：基于候选段图的动态组合搜索。"""
        total_good = int(np.sum(y_sorted == 0))
        total_bad = int(np.sum(y_sorted == 1))
        if total_good == 0 or total_bad == 0:
            return []

        candidate_pool = sorted(set(candidates))
        if len(candidate_pool) == 0:
            return []

        beam_width = min(10, max(4, len(candidate_pool) // 3))
        states: List[Tuple[float, List[float], int]] = [(0.0, [], -1)]
        best_score = -np.inf
        best_splits: List[float] = []

        for _depth in range(self.max_n_bins - 1):
            next_states: List[Tuple[float, List[float], int]] = []
            for _, current, last_idx in states:
                for idx in range(last_idx + 1, len(candidate_pool)):
                    candidate = candidate_pool[idx]
                    trial = current + [candidate]
                    full_score = self._score_splits(
                        x_sorted, y_sorted, total_good, total_bad, trial
                    )
                    seg_score = self._segment_composite_score(x_sorted, y_sorted, trial)
                    next_states.append((full_score + seg_score * 0.55, trial, idx))

            if not next_states:
                break

            next_states = sorted(next_states, key=lambda item: item[0], reverse=True)
            dedup = {}
            for score, trial, idx in next_states:
                dedup.setdefault(tuple(trial), (score, idx))
            states = [(score, list(trial), idx) for trial, (score, idx) in list(dedup.items())[:beam_width]]

            for _, trial, _ in states:
                refined = self._local_refine_splits(
                    x_sorted, y_sorted, total_good, total_bad, trial, candidate_pool
                )
                refined_score = self._score_splits(
                    x_sorted, y_sorted, total_good, total_bad, refined
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_splits = refined

        return sorted(set(best_splits))

    def _search_composite_optimal_splits(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        candidates: List[float]
    ) -> List[float]:
        """方案B当前实现：候选段组合 + beam search + 动态复合评分。"""
        total_good = int(np.sum(y_sorted == 0))
        total_bad = int(np.sum(y_sorted == 1))
        if total_good == 0 or total_bad == 0:
            return []

        candidate_pool = sorted(set(candidates))
        if len(candidate_pool) == 0:
            return []

        beam_width = min(8, max(3, len(candidate_pool) // 4))
        states: List[Tuple[float, List[float]]] = [(0.0, [])]
        best_splits: List[float] = []
        best_score = -np.inf

        for _depth in range(self.max_n_bins - 1):
            next_states: List[Tuple[float, List[float]]] = []
            for _, current in states:
                used = set(current)
                last_value = current[-1] if current else None
                for candidate in candidate_pool:
                    if candidate in used:
                        continue
                    if last_value is not None and candidate <= last_value:
                        continue
                    trial = sorted(current + [candidate])
                    score = self._score_splits(x_sorted, y_sorted, total_good, total_bad, trial)
                    segment_score = self._segment_composite_score(x_sorted, y_sorted, trial)
                    next_states.append((score + segment_score * 0.35, trial))

            if not next_states:
                break

            dedup = {}
            for score, trial in sorted(next_states, key=lambda item: item[0], reverse=True):
                dedup[tuple(trial)] = score
            states = [(score, list(trial)) for trial, score in list(dedup.items())[:beam_width]]

            for score, trial in states:
                refined = self._local_refine_splits(
                    x_sorted, y_sorted, total_good, total_bad, trial, candidate_pool
                )
                refined_score = self._score_splits(x_sorted, y_sorted, total_good, total_bad, refined)
                if refined_score > best_score:
                    best_score = refined_score
                    best_splits = refined

        if len(best_splits) == 0:
            return []
        return sorted(set(best_splits))

    def _greedy_fallback(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        candidates: List[float]
    ) -> List[float]:
        """贪心算法备选方案.

        当 OR-Tools 求解失败时使用。
        """
        n_samples = len(x_sorted)
        total_good = np.sum(y_sorted == 0)
        total_bad = np.sum(y_sorted == 1)
        
        if total_good == 0 or total_bad == 0:
            return []
        
        selected_splits = []
        remaining_candidates = candidates.copy()
        
        while len(selected_splits) < self.max_n_bins - 1 and remaining_candidates:
            best_score = -np.inf
            best_candidate = None
            best_idx = -1
            
            for i, candidate in enumerate(remaining_candidates):
                test_splits = sorted(selected_splits + [candidate])
                score = self._calculate_objective_score(
                    x_sorted, y_sorted, total_good, total_bad, test_splits
                )
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate is not None:
                selected_splits.append(best_candidate)
                remaining_candidates.pop(best_idx)
            else:
                break

        selected_splits = self._local_refine_splits(
            x_sorted, y_sorted, total_good, total_bad, selected_splits, candidates
        )
        return sorted(selected_splits)

    def _local_refine_splits(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        total_good: int,
        total_bad: int,
        splits: List[float],
        candidates: List[float]
    ) -> List[float]:
        """在贪心结果基础上做局部替换搜索，强化复合 lift 目标。"""
        current = sorted(set(splits))
        if len(current) == 0:
            return current

        candidate_pool = sorted(set(candidates))
        best_score = self._score_splits(
            x_sorted, y_sorted, total_good, total_bad, current
        )

        improved = True
        for _ in range(6):
            if not improved:
                break
            improved = False
            best_candidate = current

            # 1) 替换一个切分点
            for i in range(len(current)):
                for candidate in candidate_pool:
                    if candidate in current and not np.isclose(candidate, current[i], atol=1e-10, rtol=0):
                        continue
                    trial = current.copy()
                    trial[i] = candidate
                    trial = sorted(set(trial))
                    if len(trial) != len(current):
                        continue
                    score = self._score_splits(
                        x_sorted, y_sorted, total_good, total_bad, trial
                    )
                    if score > best_score + 1e-12:
                        best_score = score
                        best_candidate = trial
                        improved = True

            # 2) 在允许范围内尝试加一个切分点
            if len(current) < self.max_n_bins - 1:
                for candidate in candidate_pool:
                    if candidate in current:
                        continue
                    trial = sorted(set(current + [candidate]))
                    score = self._score_splits(
                        x_sorted, y_sorted, total_good, total_bad, trial
                    )
                    if score > best_score + 1e-12:
                        best_score = score
                        best_candidate = trial
                        improved = True

            # 3) 尝试删一个切分点，避免局部坏点拖累整体
            if len(current) > max(1, self.min_n_bins - 1):
                for i in range(len(current)):
                    trial = current[:i] + current[i + 1:]
                    score = self._score_splits(
                        x_sorted, y_sorted, total_good, total_bad, trial
                    )
                    if score > best_score + 1e-12:
                        best_score = score
                        best_candidate = trial
                        improved = True

            current = sorted(set(best_candidate))

        return current

    def _calculate_objective_score(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        total_good: int,
        total_bad: int,
        splits: List[float]
    ) -> float:
        """计算目标函数分数.

        :param x_sorted: 排序后的特征值
        :param y_sorted: 排序后的目标变量
        :param total_good: 总好样本数
        :param total_bad: 总坏样本数
        :param splits: 分割点列表
        :return: 目标函数值
        """
        if not splits:
            return 0.0
        
        eps = 1e-10
        
        # 找到分割点位置
        split_positions = [np.searchsorted(x_sorted, s, side='right') for s in sorted(splits)]
        split_positions = [0] + split_positions + [len(x_sorted)]
        
        # 构建箱统计
        bin_stats = []
        for i in range(len(split_positions) - 1):
            start = split_positions[i]
            end = split_positions[i + 1]
            
            if start >= end:
                bin_stats.append({'count': 0, 'good': 0, 'bad': 0})
                continue
            
            good_in_bin = np.sum(y_sorted[start:end] == 0)
            bad_in_bin = np.sum(y_sorted[start:end] == 1)
            
            bin_stats.append({
                'count': end - start,
                'good': good_in_bin,
                'bad': bad_in_bin,
            })
        
        # 处理自定义目标函数
        if self.objective == 'custom':
            processed_stats = self._compute_derived_stats(bin_stats, total_good, total_bad)
            return self.custom_objective(processed_stats, total_good, total_bad)
        
        if self.objective == 'iv':
            iv = 0.0
            for i in range(len(split_positions) - 1):
                start = split_positions[i]
                end = split_positions[i + 1]
                
                if start >= end:
                    continue
                
                good_in_bin = np.sum(y_sorted[start:end] == 0)
                bad_in_bin = np.sum(y_sorted[start:end] == 1)
                
                good_dist = good_in_bin / total_good
                bad_dist = bad_in_bin / total_bad
                
                if good_dist > eps and bad_dist > eps:
                    iv += (bad_dist - good_dist) * np.log(bad_dist / good_dist)
            return iv + self._lift_quality_bonus(bin_stats, total_good, total_bad)
        
        elif self.objective == 'ks':
            max_ks = 0
            cum_good = 0
            cum_bad = 0
            
            for i in range(len(split_positions) - 1):
                start = split_positions[i]
                end = split_positions[i + 1]
                
                if start >= end:
                    continue
                
                good_in_bin = np.sum(y_sorted[start:end] == 0)
                bad_in_bin = np.sum(y_sorted[start:end] == 1)
                
                cum_good += good_in_bin
                cum_bad += bad_in_bin
                
                ks = abs(cum_good / total_good - cum_bad / total_bad)
                max_ks = max(max_ks, ks)
            
            return max_ks
        
        elif self.objective == 'gini':
            # Gini 系数近似
            gini = 0.0
            for i in range(len(split_positions) - 1):
                start = split_positions[i]
                end = split_positions[i + 1]
                
                if start >= end:
                    continue
                
                n_bin = end - start
                p = np.sum(y_sorted[start:end] == 1) / n_bin if n_bin > 0 else 0
                gini += (n_bin / len(x_sorted)) * p * (1 - p)
            
            return 1 - gini  # 返回 1-Gini 以最大化
        
        elif self.objective == 'entropy':
            # 信息熵（需要最小化，返回负值）
            entropy = 0.0
            for i in range(len(split_positions) - 1):
                start = split_positions[i]
                end = split_positions[i + 1]
                
                if start >= end:
                    continue
                
                n_bin = end - start
                p = np.sum(y_sorted[start:end] == 1) / n_bin if n_bin > 0 else 0
                
                if 0 < p < 1:
                    entropy -= (n_bin / len(x_sorted)) * (p * np.log2(p) + (1-p) * np.log2(1-p))
            
            return -entropy  # 返回负值以最大化
        
        else:  # chi2
            # 卡方统计量
            expected_bad = total_bad / len(splits)
            chi2 = 0.0
            
            for i in range(len(split_positions) - 1):
                start = split_positions[i]
                end = split_positions[i + 1]
                
                if start >= end:
                    continue
                
                observed_bad = np.sum(y_sorted[start:end] == 1)
                
                if expected_bad > 0:
                    chi2 += (observed_bad - expected_bad) ** 2 / expected_bad
            
            return chi2

    def _or_categorical(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> List[float]:
        """对类别型变量使用 OR-Tools 优化分箱.

        :param X: 特征数据
        :param y: 目标变量
        :return: 分割点列表（编码边界）
        """
        # 计算总体统计
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()
        
        if total_good == 0 or total_bad == 0:
            return []
        
        # 使用向量化操作计算类别统计
        df = pd.DataFrame({'X': X, 'y': y})
        category_stats = df.groupby('X')['y'].agg(['sum', 'count']).reset_index()
        category_stats.columns = ['category', 'bad_count', 'count']
        category_stats['good_count'] = category_stats['count'] - category_stats['bad_count']
        
        # 计算坏样本率用于排序
        category_stats['bad_rate'] = category_stats['bad_count'] / category_stats['count']
        category_stats = category_stats.sort_values('bad_rate')
        
        # 过滤掉样本数过少的类别
        min_samples = self._get_min_samples(len(X))
        category_stats = category_stats[category_stats['count'] >= min_samples]
        
        if len(category_stats) <= self.max_n_bins:
            return []
        
        # 返回编码边界
        n_categories = len(category_stats)
        return [i - 0.5 for i in range(1, min(n_categories, self.max_n_bins))]

    def _assign_bins(
        self,
        X: pd.Series,
        feature: str
    ) -> np.ndarray:
        """为数据分配分箱索引.

        :param X: 特征数据
        :param feature: 特征名
        :return: 分箱索引数组
        """
        x_vals = X.values
        
        if self.feature_types_[feature] == 'categorical':
            codes = pd.Categorical(X).codes
            return np.where(X.isna(), -1, codes)
        else:
            splits = self.splits_[feature]
            n = len(x_vals)
            bins = np.zeros(n, dtype=int)
            
            # 处理缺失值
            missing_mask = X.isna()
            bins[missing_mask] = -1
            
            # 处理特殊值
            if self.special_codes:
                for code in self.special_codes:
                    bins[x_vals == code] = -2
            
            # 正常值
            valid_mask = ~missing_mask
            if self.special_codes:
                for code in self.special_codes:
                    valid_mask = valid_mask & (x_vals != code)
            
            if valid_mask.any() and len(splits) > 0:
                bins[valid_mask] = np.searchsorted(
                    splits, x_vals[valid_mask], side='right'
                )
            
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
        :param kwargs: 其他参数(保留兼容性)
        :return: 转换后的数据, 格式与输入X相同

        :example:
        >>> binner = ORBinning(objective='iv')
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
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature in self.splits_:
                bins = self._assign_bins(X[feature], feature)

                if metric == 'indices':
                    result[feature] = bins
                elif metric == 'bins':
                    labels = self._get_bin_labels(self.splits_[feature], bins)
                    result[feature] = [labels[b] if b >= 0 else ('missing' if b == -1 else 'special')
                                      for b in bins]
                elif metric == 'woe':
                    # 优先使用_woe_maps_（从export/load导入）
                    if hasattr(self, '_woe_maps_') and feature in self._woe_maps_:
                        woe_map = self._woe_maps_[feature]
                    elif feature in self.bin_tables_:
                        woe_map = dict(zip(
                            range(len(self.bin_tables_[feature])),
                            self.bin_tables_[feature]['分档WOE值']
                        ))
                    else:
                        raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                    result[feature] = [woe_map.get(b, 0) for b in bins]
                else:
                    raise ValueError(f"不支持的metric: {metric}")
            else:
                result[feature] = X[feature]

        return result


# =============================================================================
# 常用自定义目标函数
# =============================================================================

class CustomObjectives:
    """常用自定义目标函数集合.
    
    提供预定义的复合指标目标函数，可与 ORBinning 的 custom_objective 参数配合使用.
    
    **使用示例**
    
    >>> from hscredit.core.binning import ORBinning, CustomObjectives
    >>> 
    >>> # 使用预定义的最大 LIFT + IV 目标
    >>> binner = ORBinning(
    ...     objective='custom',
    ...     custom_objective=CustomObjectives.max_lift_iv(lift_weight=0.1)
    ... )
    >>> binner.fit(X_train, y_train)
    >>> 
    >>> # 使用自定义权重
    >>> binner = ORBinning(
    ...     objective='custom',
    ...     custom_objective=CustomObjectives.min_lift_distance_iv(
    ...         iv_weight=1.0, 
    ...         lift_weight=0.5
    ...     )
    ... )
    """
    
    @staticmethod
    def max_lift_iv(lift_weight: float = 0.1, iv_weight: float = 1.0):
        """最大 LIFT + IV 目标函数.
        
        目标：最大化（所有分箱中LIFT最大的那一箱的LIFT值）+ IV 值
        
        :param lift_weight: 最大LIFT项权重
        :param iv_weight: IV 项权重
        :return: 目标函数
        
        **原理**
        
        LIFT = 箱内坏样本率 / 总体坏样本率
        - LIFT > 1: 该箱坏样本率高于平均水平
        - LIFT < 1: 该箱坏样本率低于平均水平
        
        通过最大化最大LIFT，鼓励产生至少一个强区分能力的分箱.
        """
        def objective(bin_stats: List[Dict], total_good: int, total_bad: int) -> float:
            eps = 1e-10
            total_iv = 0.0
            lift_values = []
            
            for stat in bin_stats:
                if stat['count'] == 0:
                    continue
                
                # IV 贡献
                if 'woe' in stat:
                    woe = stat['woe']
                    bad_rate = stat.get('bad_rate', 0)
                    good_rate = stat.get('good_rate', 0)
                    total_iv += woe * (bad_rate - good_rate)
                
                # 收集LIFT值
                if 'lift' in stat:
                    lift_values.append(stat['lift'])
            
            # 最大LIFT（所有分箱中最大的那个）
            max_lift = max(lift_values) if lift_values else 1.0
            
            return iv_weight * total_iv + lift_weight * max_lift
        
        return objective
    
    @staticmethod
    def min_lift_iv(lift_weight: float = 0.1, iv_weight: float = 1.0):
        """最小 LIFT + IV 目标函数.
        
        目标：最大化（所有分箱中LIFT最小的那一箱的LIFT值）+ IV 值
        
        :param lift_weight: 最小LIFT项权重
        :param iv_weight: IV 项权重
        :return: 目标函数
        
        **原理**
        
        通过最大化最小LIFT，鼓励产生均匀分布的区分能力，避免某些分箱过于弱势.
        """
        def objective(bin_stats: List[Dict], total_good: int, total_bad: int) -> float:
            eps = 1e-10
            total_iv = 0.0
            lift_values = []
            
            for stat in bin_stats:
                if stat['count'] == 0:
                    continue
                
                # IV 贡献
                if 'woe' in stat:
                    woe = stat['woe']
                    bad_rate = stat.get('bad_rate', 0)
                    good_rate = stat.get('good_rate', 0)
                    total_iv += woe * (bad_rate - good_rate)
                
                # 收集LIFT值
                if 'lift' in stat:
                    lift_values.append(stat['lift'])
            
            # 最小LIFT（所有分箱中最小的那个）
            min_lift = min(lift_values) if lift_values else 1.0
            
            return iv_weight * total_iv + lift_weight * min_lift
        
        return objective
    
    @staticmethod
    def max_min_lift_sum_iv(lift_weight: float = 0.1, iv_weight: float = 1.0):
        """最大LIFT + 最小LIFT + IV 目标函数.
        
        目标：最大化（最大LIFT值 + 最小LIFT值）+ IV 值
        
        :param lift_weight: LIFT和项权重
        :param iv_weight: IV 项权重
        :return: 目标函数
        
        **原理**
        
        同时考虑最强和最弱分箱的LIFT值，确保分箱既有强区分箱，整体区分度也较好.
        """
        def objective(bin_stats: List[Dict], total_good: int, total_bad: int) -> float:
            eps = 1e-10
            total_iv = 0.0
            lift_values = []
            
            for stat in bin_stats:
                if stat['count'] == 0:
                    continue
                
                # IV 贡献
                if 'woe' in stat:
                    woe = stat['woe']
                    bad_rate = stat.get('bad_rate', 0)
                    good_rate = stat.get('good_rate', 0)
                    total_iv += woe * (bad_rate - good_rate)
                
                # 收集LIFT值
                if 'lift' in stat:
                    lift_values.append(stat['lift'])
            
            if not lift_values:
                return iv_weight * total_iv
            
            # 最大LIFT + 最小LIFT
            max_lift = max(lift_values)
            min_lift = min(lift_values)
            lift_sum = max_lift + min_lift
            
            return iv_weight * total_iv + lift_weight * lift_sum
        
        return objective
    
    @staticmethod
    def max_lift_distance_iv(lift_weight: float = 0.1, iv_weight: float = 1.0):
        """最大LIFT离1的距离 + IV 目标函数.
        
        目标：最大化（最大LIFT离1的距离）+ IV 值
        
        :param lift_weight: 最大LIFT距离项权重
        :param iv_weight: IV 项权重
        :return: 目标函数
        
        **原理**
        
        鼓励产生至少一个与基准（LIFT=1）有明显偏离的强区分分箱.
        """
        def objective(bin_stats: List[Dict], total_good: int, total_bad: int) -> float:
            eps = 1e-10
            total_iv = 0.0
            lift_values = []
            
            for stat in bin_stats:
                if stat['count'] == 0:
                    continue
                
                # IV 贡献
                if 'woe' in stat:
                    woe = stat['woe']
                    bad_rate = stat.get('bad_rate', 0)
                    good_rate = stat.get('good_rate', 0)
                    total_iv += woe * (bad_rate - good_rate)
                
                # 收集LIFT值
                if 'lift' in stat:
                    lift_values.append(stat['lift'])
            
            if not lift_values:
                return iv_weight * total_iv
            
            # 最大LIFT离1的距离
            max_lift_dist = max(abs(l - 1.0) for l in lift_values)
            
            return iv_weight * total_iv + lift_weight * max_lift_dist
        
        return objective
    
    @staticmethod
    def min_lift_distance_iv(lift_weight: float = 0.1, iv_weight: float = 1.0):
        """最小LIFT离1的距离 + IV 目标函数.
        
        目标：最大化（最小LIFT离1的距离）+ IV 值
        
        :param lift_weight: 最小LIFT距离项权重
        :param iv_weight: IV 项权重
        :return: 目标函数
        
        **原理**
        
        鼓励最弱分箱也有较好的区分能力，避免某些分箱过于接近基准线.
        """
        def objective(bin_stats: List[Dict], total_good: int, total_bad: int) -> float:
            eps = 1e-10
            total_iv = 0.0
            lift_values = []
            
            for stat in bin_stats:
                if stat['count'] == 0:
                    continue
                
                # IV 贡献
                if 'woe' in stat:
                    woe = stat['woe']
                    bad_rate = stat.get('bad_rate', 0)
                    good_rate = stat.get('good_rate', 0)
                    total_iv += woe * (bad_rate - good_rate)
                
                # 收集LIFT值
                if 'lift' in stat:
                    lift_values.append(stat['lift'])
            
            if not lift_values:
                return iv_weight * total_iv
            
            # 最小LIFT离1的距离
            min_lift_dist = min(abs(l - 1.0) for l in lift_values)
            
            return iv_weight * total_iv + lift_weight * min_lift_dist
        
        return objective
    
    @staticmethod
    def lift_distance_sum_iv(lift_weight: float = 0.1, iv_weight: float = 1.0):
        """最大/最小LIFT离1的距离求和 + IV 目标函数.
        
        目标：最大化（最大LIFT离1的距离 + 最小LIFT离1的距离）+ IV 值
        
        :param lift_weight: LIFT距离和项权重
        :param iv_weight: IV 项权重
        :return: 目标函数
        
        **原理**
        
        同时考虑最强和最弱分箱与基准线的偏离程度，平衡极端区分能力.
        """
        def objective(bin_stats: List[Dict], total_good: int, total_bad: int) -> float:
            eps = 1e-10
            total_iv = 0.0
            lift_values = []
            
            for stat in bin_stats:
                if stat['count'] == 0:
                    continue
                
                # IV 贡献
                if 'woe' in stat:
                    woe = stat['woe']
                    bad_rate = stat.get('bad_rate', 0)
                    good_rate = stat.get('good_rate', 0)
                    total_iv += woe * (bad_rate - good_rate)
                
                # 收集LIFT值
                if 'lift' in stat:
                    lift_values.append(stat['lift'])
            
            if not lift_values:
                return iv_weight * total_iv
            
            # 最大LIFT距离 + 最小LIFT距离
            lift_distances = [abs(l - 1.0) for l in lift_values]
            dist_sum = max(lift_distances) + min(lift_distances)
            
            return iv_weight * total_iv + lift_weight * dist_sum
        
        return objective
    
    @staticmethod
    def max_ks_iv(ks_weight: float = 0.5, iv_weight: float = 1.0):
        """最大 KS + IV 复合目标函数.
        
        同时考虑 KS 统计量和 IV 值。
        
        :param ks_weight: KS 项权重
        :param iv_weight: IV 项权重
        :return: 目标函数
        """
        def objective(bin_stats: List[Dict], total_good: int, total_bad: int) -> float:
            eps = 1e-10
            total_samples = total_good + total_bad
            
            if total_good == 0 or total_bad == 0:
                return 0.0
            
            total_iv = 0.0
            cum_good = 0
            cum_bad = 0
            max_ks = 0.0
            
            for stat in bin_stats:
                if stat['count'] == 0:
                    continue
                
                good = stat['good']
                bad = stat['bad']
                
                # IV 贡献
                if 'woe' in stat:
                    woe = stat['woe']
                    bad_rate = stat.get('bad_rate', 0)
                    good_rate = stat.get('good_rate', 0)
                    total_iv += woe * (bad_rate - good_rate)
                
                # KS 计算
                cum_good += good
                cum_bad += bad
                ks = abs(cum_good / total_good - cum_bad / total_bad)
                max_ks = max(max_ks, ks)
            
            return iv_weight * total_iv + ks_weight * max_ks
        
        return objective
    
    @staticmethod
    def woe_variance_iv(variance_weight: float = 0.1, iv_weight: float = 1.0):
        """WOE 方差 + IV 目标函数.
        
        鼓励 WOE 在各箱之间有较大差异，提高区分度。
        
        :param variance_weight: WOE 方差项权重
        :param iv_weight: IV 项权重
        :return: 目标函数
        """
        def objective(bin_stats: List[Dict], total_good: int, total_bad: int) -> float:
            woe_values = []
            total_iv = 0.0
            
            for stat in bin_stats:
                if stat['count'] == 0:
                    continue
                
                # IV 贡献
                if 'woe' in stat:
                    woe = stat['woe']
                    woe_values.append(woe)
                    bad_rate = stat.get('bad_rate', 0)
                    good_rate = stat.get('good_rate', 0)
                    total_iv += woe * (bad_rate - good_rate)
            
            # WOE 方差
            if len(woe_values) > 1:
                woe_variance = np.var(woe_values)
            else:
                woe_variance = 0.0
            
            return iv_weight * total_iv + variance_weight * woe_variance
        
        return objective
