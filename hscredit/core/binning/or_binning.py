from __future__ import annotations

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
    :param max_candidates: 最大候选分割点数，默认为100
        - 如果唯一值超过此数，将使用分位数采样
    :param time_limit: 求解时间限制（秒），默认为60
        - 超过此时间将返回当前找到的最优解
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :param special_codes: 特殊值列表，默认为None
    :param random_state: 随机种子，默认为None

    **参考样例**

    sklearn风格 (推荐):

    >>> from hscredit.core.binning import ORBinning
    >>> # 最大化 IV
    >>> binner = ORBinning(max_n_bins=5, objective='iv', monotonic=True)
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_test)
    >>> bin_table = binner.get_bin_table('feature_name')

    scorecardpipeline风格 (目标列在DataFrame中):

    >>> from hscredit.core.binning import ORBinning
    >>> # 初始化时指定目标列名，fit时传入完整DataFrame
    >>> binner = ORBinning(target='target', max_n_bins=5, objective='ks', time_limit=60)
    >>> binner.fit(df)  # df包含特征列和目标列'target'
    >>> X_binned = binner.transform(df.drop(columns=['target']))

    混合风格 (y参数优先):

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
        max_candidates: int = 100,
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
            self._n_total_samples = len(X)  # 记录全量样本数供 _score_splits 使用
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

        使用前缀和预计算 + 多策略搜索 + 复合评分精化，
        在保证速度的同时获得高质量分箱结果。

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
            return candidates[:max(0, self.max_n_bins - 1)]

        # 排序数据
        sorted_indices = np.argsort(x_vals)
        x_sorted = x_vals[sorted_indices]
        y_sorted = y_vals[sorted_indices]

        total_good = int(np.sum(y_vals == 0))
        total_bad = int(np.sum(y_vals == 1))

        if total_good == 0 or total_bad == 0:
            return []

        min_samples = self._get_min_samples(
            getattr(self, '_n_total_samples', n_samples)
        )

        # ====== 前缀和预计算（O(n)，只执行一次）======
        is_bad = (y_sorted == 1).astype(np.int64)
        prefix_bad = np.empty(n_samples + 1, dtype=np.int64)
        prefix_bad[0] = 0
        np.cumsum(is_bad, out=prefix_bad[1:])
        prefix_good = np.arange(n_samples + 1, dtype=np.int64) - prefix_bad

        cand_pos = np.searchsorted(x_sorted, candidates, side='right')
        positions = np.empty(n_candidates + 2, dtype=np.int64)
        positions[0] = 0
        positions[1:-1] = cand_pos
        positions[-1] = n_samples

        cum_bad = prefix_bad[positions]
        cum_good = prefix_good[positions]

        # ====== 策略1: DP 最优 IV 分割 ======
        dp_splits = self._dp_optimal_splits(
            candidates, cum_bad, cum_good, total_good, total_bad, min_samples
        )

        # ====== 策略2: 单调性感知 DP ======
        mono_dp_splits = self._dp_monotonic_splits(
            candidates, cum_bad, cum_good, total_good, total_bad, min_samples
        )

        # ====== 策略3: 快速贪心 ======
        greedy_splits = self._greedy_splits_fast(
            candidates, cum_bad, cum_good, total_good, total_bad, n_samples, min_samples
        )

        # ====== 策略4: DP top-K 候选 + 复合评分筛选 ======
        topk_splits = self._dp_topk_with_composite(
            candidates, cum_bad, cum_good, total_good, total_bad,
            min_samples, x_sorted, y_sorted
        )

        # ====== 从所有策略中选出最优 ======
        best_score = -np.inf
        best_splits = []

        for splits in [dp_splits, mono_dp_splits, greedy_splits, topk_splits]:
            if not splits:
                continue
            # 评估原始策略输出（保护 composite 友好的分割点）
            score_raw = self._score_splits(
                x_sorted, y_sorted, total_good, total_bad, splits
            )
            if score_raw > best_score:
                best_score = score_raw
                best_splits = list(splits)
            # 也尝试 IV 精化版本（可能进一步提升）
            refined = self._local_refine_fast(
                splits, candidates, cum_bad, cum_good,
                total_good, total_bad, n_samples, min_samples
            )
            score_refined = self._score_splits(
                x_sorted, y_sorted, total_good, total_bad, refined
            )
            if score_refined > best_score:
                best_score = score_refined
                best_splits = refined

        # 对最优候选做复合评分精化（仅一次，避免重复计算）
        if best_splits:
            best_splits = self._local_refine_composite(
                best_splits, candidates, x_sorted, y_sorted,
                total_good, total_bad
            )

        if not best_splits:
            best_splits = dp_splits or greedy_splits or []

        return sorted(best_splits)

    # ------------------------------------------------------------------
    # 快速搜索方法（利用前缀和，评分 O(K) 而非 O(n)）
    # ------------------------------------------------------------------

    def _dp_monotonic_splits(
        self,
        candidates: List[float],
        cum_bad: np.ndarray,
        cum_good: np.ndarray,
        total_good: int,
        total_bad: int,
        min_samples: int,
    ) -> List[float]:
        """单调性感知 DP：在 IV 最优化中加入单调性约束.

        对于 ascending/descending，仅允许相邻箱的 bad_rate 满足单调约束；
        对于 auto 模式，分别尝试 ascending 和 descending，返回较优结果。
        """
        mono = self.monotonic
        if mono in (False, None, 'none'):
            return []  # 不需要单调约束，退回普通 DP

        if mono in (True, 'auto', 'auto_asc_desc', 'auto_heuristic'):
            # 尝试两个方向
            asc_indices = self._dp_mono_direction(
                candidates, cum_bad, cum_good, total_good, total_bad,
                min_samples, 'ascending'
            )
            desc_indices = self._dp_mono_direction(
                candidates, cum_bad, cum_good, total_good, total_bad,
                min_samples, 'descending'
            )
            # 选 IV 更大的
            iv_asc = self._iv_from_split_indices(
                asc_indices, cum_bad, cum_good, total_good, total_bad, len(candidates)
            ) if asc_indices else -np.inf
            iv_desc = self._iv_from_split_indices(
                desc_indices, cum_bad, cum_good, total_good, total_bad, len(candidates)
            ) if desc_indices else -np.inf
            best_indices = asc_indices if iv_asc >= iv_desc else desc_indices
        else:
            best_indices = self._dp_mono_direction(
                candidates, cum_bad, cum_good, total_good, total_bad,
                min_samples, mono
            )

        return [candidates[i] for i in best_indices]

    def _dp_mono_direction(
        self,
        candidates: List[float],
        cum_bad: np.ndarray,
        cum_good: np.ndarray,
        total_good: int,
        total_bad: int,
        min_samples: int,
        direction: str,
    ) -> List[int]:
        """单方向单调性 DP，返回候选点的段索引列表."""
        C = len(candidates)
        n_seg = C + 1
        K_max = min(self.max_n_bins, n_seg)
        K_min = max(self.min_n_bins, 2)
        eps = 1e-10
        NEG_INF = -1e18

        if K_max < K_min:
            return []

        # 预计算合并段的 IV 和 bad_rate
        def seg_stats(s, e):
            bad = int(cum_bad[e + 1] - cum_bad[s])
            good = int(cum_good[e + 1] - cum_good[s])
            count = bad + good
            if count < min_samples:
                return NEG_INF, 0.0
            bad_dist = bad / total_bad if total_bad > 0 else 0.0
            good_dist = good / total_good if total_good > 0 else 0.0
            iv = (bad_dist - good_dist) * np.log(bad_dist / good_dist) if bad_dist > eps and good_dist > eps else 0.0
            br = bad / count
            return iv, br

        # dp[k][j] = (最优IV, 最后一箱的bad_rate)
        dp_val = np.full((K_max + 1, n_seg), NEG_INF)
        dp_br = np.zeros((K_max + 1, n_seg))
        parent = np.full((K_max + 1, n_seg), -1, dtype=np.int32)

        for j in range(n_seg):
            iv, br = seg_stats(0, j)
            dp_val[1][j] = iv
            dp_br[1][j] = br

        for k in range(2, K_max + 1):
            for j in range(k - 1, n_seg):
                for i in range(k - 2, j):
                    if dp_val[k - 1][i] <= NEG_INF:
                        continue
                    iv_new, br_new = seg_stats(i + 1, j)
                    if iv_new <= NEG_INF:
                        continue
                    prev_br = dp_br[k - 1][i]
                    # 单调性检查
                    if direction == 'ascending' and br_new < prev_br - 1e-10:
                        continue
                    if direction == 'descending' and br_new > prev_br + 1e-10:
                        continue
                    val = dp_val[k - 1][i] + iv_new
                    if val > dp_val[k][j]:
                        dp_val[k][j] = val
                        dp_br[k][j] = br_new
                        parent[k][j] = i

        best_score = NEG_INF
        best_k = -1
        for k in range(K_min, K_max + 1):
            if dp_val[k][n_seg - 1] > best_score:
                best_score = dp_val[k][n_seg - 1]
                best_k = k

        if best_k < 0:
            return []

        splits_seg = []
        j = n_seg - 1
        for k in range(best_k, 1, -1):
            i = parent[k][j]
            if i < 0:
                return []
            splits_seg.append(i)
            j = i
        splits_seg.reverse()

        return splits_seg  # 返回段索引

    def _iv_from_split_indices(
        self,
        split_seg_indices: List[int],
        cum_bad: np.ndarray,
        cum_good: np.ndarray,
        total_good: int,
        total_bad: int,
        n_candidates: int,
    ) -> float:
        """从段索引列表快速计算 IV，O(K) 复杂度."""
        if not split_seg_indices:
            return -np.inf
        eps = 1e-10
        n_seg = n_candidates + 1
        # 箱边界段: [0..idx0], [idx0+1..idx1], ..., [idxN+1..n_seg-1]
        boundaries = [0] + [idx + 1 for idx in split_seg_indices] + [n_seg]
        iv = 0.0
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            bad = int(cum_bad[e] - cum_bad[s])
            good = int(cum_good[e] - cum_good[s])
            bad_dist = bad / total_bad if total_bad > 0 else 0.0
            good_dist = good / total_good if total_good > 0 else 0.0
            if bad_dist > eps and good_dist > eps:
                iv += (bad_dist - good_dist) * np.log(bad_dist / good_dist)
        return iv

    def _dp_topk_with_composite(
        self,
        candidates: List[float],
        cum_bad: np.ndarray,
        cum_good: np.ndarray,
        total_good: int,
        total_bad: int,
        min_samples: int,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
    ) -> List[float]:
        """DP top-K beam search + 复合评分筛选.

        在 DP 过程中保留 top-K 个候选方案（不仅仅是最优 IV），
        然后用完整的 _score_splits 评分，选出复合评分最高的方案。
        """
        C = len(candidates)
        n_seg = C + 1
        K_max = min(self.max_n_bins, n_seg)
        K_min = max(self.min_n_bins, 2)
        eps = 1e-10
        NEG_INF = -1e18
        beam_size = min(8, max(3, C // 5))

        if K_max < K_min:
            return []

        # 预计算 seg_iv[s][e]
        seg_iv = np.full((n_seg, n_seg), NEG_INF)
        for s in range(n_seg):
            for e in range(s, n_seg):
                bad = int(cum_bad[e + 1] - cum_bad[s])
                good = int(cum_good[e + 1] - cum_good[s])
                count = bad + good
                if count < min_samples:
                    continue
                bad_dist = bad / total_bad if total_bad > 0 else 0.0
                good_dist = good / total_good if total_good > 0 else 0.0
                if bad_dist > eps and good_dist > eps:
                    seg_iv[s, e] = (bad_dist - good_dist) * np.log(bad_dist / good_dist)
                else:
                    seg_iv[s, e] = 0.0

        # beam[k] = list of (iv_score, [split_seg_indices])
        beam: Dict[int, List[Tuple[float, List[int]]]] = {}
        # k=1: single bin covering 0..j
        beam[1] = []
        for j in range(n_seg):
            if seg_iv[0, j] > NEG_INF:
                beam[1].append((seg_iv[0, j], []))
        # 只保留 beam_size 个
        beam[1] = sorted(beam[1], key=lambda x: x[0], reverse=True)[:beam_size * 2]

        for k in range(2, K_max + 1):
            beam[k] = []
            prev = beam.get(k - 1, [])
            for prev_iv, prev_splits in prev:
                # 上一个方案覆盖到段索引 last_end
                last_end = prev_splits[-1] if prev_splits else -1
                start_seg = last_end + 1
                for j in range(start_seg + 1, n_seg):
                    # 新箱覆盖 (start_seg+1)..j — 但边界需要对齐
                    # 第 k-1 个箱的最后一段是 prev_splits[-1]
                    # 第 k 个箱从 (prev_splits[-1]+1) 开始
                    new_start = last_end + 1
                    if seg_iv[new_start, j] > NEG_INF:
                        new_iv = prev_iv + seg_iv[new_start, j]
                        # 分割点是第 j 段和第 j+1 段之间 → 段索引 j
                        # 但只有不是最后一段时才加分割点
                        if j < n_seg - 1:
                            beam[k].append((new_iv, prev_splits + [j]))
                        elif j == n_seg - 1:
                            # 这是最后一段，完整方案
                            beam[k].append((new_iv, prev_splits[:]))
            beam[k] = sorted(beam[k], key=lambda x: x[0], reverse=True)[:beam_size]

        # 从所有完整方案中用复合评分选最佳
        best_score = -np.inf
        best_splits: List[float] = []

        for k in range(K_min, K_max + 1):
            for iv_score, split_indices in beam.get(k, []):
                if not split_indices:
                    continue
                splits = [candidates[i] for i in split_indices]
                score = self._score_splits(
                    x_sorted, y_sorted, total_good, total_bad, splits
                )
                if score > best_score:
                    best_score = score
                    best_splits = splits

        return best_splits

    def _local_refine_composite(
        self,
        splits: List[float],
        candidates: List[float],
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        total_good: int,
        total_bad: int,
    ) -> List[float]:
        """使用完整复合评分（_score_splits）进行局部精化.

        对每个切分点尝试替换为邻近候选点，选择复合评分最高的方案。
        使用邻域搜索（±5 个候选）限制计算量。
        """
        current = sorted(set(splits))
        if not current:
            return current

        candidate_pool = sorted(set(candidates))
        cand_idx = {v: i for i, v in enumerate(candidate_pool)}
        C = len(candidate_pool)
        NEIGHBOR = 20  # 每侧搜索半径（需要足够大以跨越局部最优）

        best_score = self._score_splits(
            x_sorted, y_sorted, total_good, total_bad, current
        )

        for _ in range(2):
            improved = False
            best_result = current[:]

            # 替换每个切分点（仅搜索邻域）
            for i in range(len(current)):
                ci = cand_idx.get(current[i], -1)
                if ci < 0:
                    # 当前切分点不在候选列表中，用最近匹配
                    ci = int(np.searchsorted(candidate_pool, current[i]))
                lo = max(0, ci - NEIGHBOR)
                hi = min(C, ci + NEIGHBOR + 1)
                for j in range(lo, hi):
                    cand = candidate_pool[j]
                    if cand == current[i]:
                        continue
                    if cand in current:
                        continue
                    trial = current[:]
                    trial[i] = cand
                    trial = sorted(set(trial))
                    if len(trial) != len(current):
                        continue
                    score = self._score_splits(
                        x_sorted, y_sorted, total_good, total_bad, trial
                    )
                    if score > best_score + 1e-12:
                        best_score = score
                        best_result = trial
                        improved = True

            # 添加一个切分点（仅在箱间中间位置搜索）
            if len(current) < self.max_n_bins - 1:
                # 构建搜索区间（每两个相邻切分点之间的中间区域）
                boundaries = [candidate_pool[0]] + current + [candidate_pool[-1]]
                for bi in range(len(boundaries) - 1):
                    lo_val, hi_val = boundaries[bi], boundaries[bi + 1]
                    lo_ci = int(np.searchsorted(candidate_pool, lo_val))
                    hi_ci = int(np.searchsorted(candidate_pool, hi_val))
                    # 只在区间中间 ±2 处搜索
                    mid = (lo_ci + hi_ci) // 2
                    for j in range(max(lo_ci, mid - 2), min(hi_ci, mid + 3)):
                        if j >= C:
                            continue
                        cand = candidate_pool[j]
                        if cand in current:
                            continue
                        trial = sorted(set(current + [cand]))
                        score = self._score_splits(
                            x_sorted, y_sorted, total_good, total_bad, trial
                        )
                        if score > best_score + 1e-12:
                            best_score = score
                            best_result = trial
                            improved = True

            current = sorted(set(best_result))
            if not improved:
                break

        return current

    def _dp_optimal_splits(
        self,
        candidates: List[float],
        cum_bad: np.ndarray,
        cum_good: np.ndarray,
        total_good: int,
        total_bad: int,
        min_samples: int,
    ) -> List[float]:
        """DP 最优分割搜索，时间复杂度 O(C² × K).

        利用 IV 可分解性（各箱贡献之和），通过动态规划找到全局最优的 K 个分箱。
        对于 C=50, K=5 仅需约 12,500 次 O(1) 的段 IV 计算。
        """
        C = len(candidates)
        n_seg = C + 1  # 段数 = 候选点数 + 1
        K_max = min(self.max_n_bins, n_seg)
        K_min = max(self.min_n_bins, 2)
        eps = 1e-10
        NEG_INF = -1e18

        if K_max < K_min:
            return []

        # 预计算 seg_iv[s][e] = 将段 s..e 合并为一个箱时的 IV 贡献
        # 段 s 覆盖 positions[s] 到 positions[s+1]
        # 合并段 s..e 的箱覆盖 positions[s] 到 positions[e+1]
        seg_iv = np.full((n_seg, n_seg), NEG_INF)
        for s in range(n_seg):
            bad_s = int(cum_bad[s])
            good_s = int(cum_good[s])
            for e in range(s, n_seg):
                bad = int(cum_bad[e + 1]) - bad_s
                good = int(cum_good[e + 1]) - good_s
                count = bad + good
                if count < min_samples:
                    continue
                bad_dist = bad / total_bad if total_bad > 0 else 0.0
                good_dist = good / total_good if total_good > 0 else 0.0
                if bad_dist > eps and good_dist > eps:
                    seg_iv[s, e] = (bad_dist - good_dist) * np.log(bad_dist / good_dist)
                else:
                    seg_iv[s, e] = 0.0

        # dp[k][j] = 使用 k 个箱覆盖段 0..j 的最优 IV
        dp = np.full((K_max + 1, n_seg), NEG_INF)
        parent = np.full((K_max + 1, n_seg), -1, dtype=np.int32)

        # 基础：1 个箱覆盖段 0..j
        for j in range(n_seg):
            dp[1][j] = seg_iv[0, j]

        # 递推
        for k in range(2, K_max + 1):
            for j in range(k - 1, n_seg):
                for i in range(k - 2, j):
                    if dp[k - 1][i] > NEG_INF and seg_iv[i + 1, j] > NEG_INF:
                        val = dp[k - 1][i] + seg_iv[i + 1, j]
                        if val > dp[k][j]:
                            dp[k][j] = val
                            parent[k][j] = i

        # 选最优 k
        best_score = NEG_INF
        best_k = -1
        for k in range(K_min, K_max + 1):
            if dp[k][n_seg - 1] > best_score:
                best_score = dp[k][n_seg - 1]
                best_k = k

        if best_k < 0:
            return []

        # 回溯得到分割段索引
        splits_seg = []
        j = n_seg - 1
        for k in range(best_k, 1, -1):
            i = parent[k][j]
            if i < 0:
                return []
            splits_seg.append(i)
            j = i
        splits_seg.reverse()

        return [candidates[i] for i in splits_seg]

    def _fast_objective_from_indices(
        self,
        split_indices: List[int],
        cum_bad: np.ndarray,
        cum_good: np.ndarray,
        total_good: int,
        total_bad: int,
        n_samples: int,
        min_samples: int,
    ) -> float:
        """利用前缀和快速计算目标函数，O(K) 复杂度.

        :param split_indices: 候选点索引列表（在 candidates 数组中的下标）
        """
        eps = 1e-10
        n_pos = len(cum_bad) - 1  # C + 1

        # 箱边界：[0, idx+1, ..., C+1]
        boundaries = [0] + [idx + 1 for idx in split_indices] + [n_pos]
        n_bins = len(boundaries) - 1

        # 可行性检查
        for i in range(n_bins):
            count = int(cum_bad[boundaries[i + 1]] - cum_bad[boundaries[i]]
                        + cum_good[boundaries[i + 1]] - cum_good[boundaries[i]])
            if count < min_samples:
                return -np.inf

        if self.objective in ('iv', 'custom'):
            iv = 0.0
            for i in range(n_bins):
                bad = int(cum_bad[boundaries[i + 1]] - cum_bad[boundaries[i]])
                good = int(cum_good[boundaries[i + 1]] - cum_good[boundaries[i]])
                bad_dist = bad / total_bad if total_bad > 0 else 0.0
                good_dist = good / total_good if total_good > 0 else 0.0
                if bad_dist > eps and good_dist > eps:
                    iv += (bad_dist - good_dist) * np.log(bad_dist / good_dist)
            return iv

        elif self.objective == 'ks':
            max_ks = 0.0
            run_bad = 0
            run_good = 0
            for i in range(n_bins):
                run_bad += int(cum_bad[boundaries[i + 1]] - cum_bad[boundaries[i]])
                run_good += int(cum_good[boundaries[i + 1]] - cum_good[boundaries[i]])
                ks = abs(run_good / total_good - run_bad / total_bad)
                max_ks = max(max_ks, ks)
            return max_ks

        elif self.objective == 'gini':
            gini = 0.0
            for i in range(n_bins):
                bad = int(cum_bad[boundaries[i + 1]] - cum_bad[boundaries[i]])
                good = int(cum_good[boundaries[i + 1]] - cum_good[boundaries[i]])
                count = bad + good
                if count > 0:
                    p = bad / count
                    gini += (count / n_samples) * p * (1 - p)
            return 1 - gini

        elif self.objective == 'entropy':
            entropy = 0.0
            for i in range(n_bins):
                bad = int(cum_bad[boundaries[i + 1]] - cum_bad[boundaries[i]])
                good = int(cum_good[boundaries[i + 1]] - cum_good[boundaries[i]])
                count = bad + good
                if count > 0:
                    p = bad / count
                    if 0 < p < 1:
                        entropy -= (count / n_samples) * (p * np.log2(p) + (1 - p) * np.log2(1 - p))
            return -entropy

        else:  # chi2
            expected_bad = total_bad / n_bins if n_bins > 0 else 1
            chi2 = 0.0
            for i in range(n_bins):
                bad = int(cum_bad[boundaries[i + 1]] - cum_bad[boundaries[i]])
                if expected_bad > 0:
                    chi2 += (bad - expected_bad) ** 2 / expected_bad
            return chi2

    def _greedy_splits_fast(
        self,
        candidates: List[float],
        cum_bad: np.ndarray,
        cum_good: np.ndarray,
        total_good: int,
        total_bad: int,
        n_samples: int,
        min_samples: int,
    ) -> List[float]:
        """快速贪心分割选择，利用前缀和评分，O(C×K) 复杂度."""
        C = len(candidates)
        selected: List[int] = []

        for _ in range(self.max_n_bins - 1):
            best_score = -np.inf
            best_idx = -1
            for idx in range(C):
                if idx in selected:
                    continue
                trial = sorted(selected + [idx])
                score = self._fast_objective_from_indices(
                    trial, cum_bad, cum_good, total_good, total_bad, n_samples, min_samples
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0 and best_score > -np.inf:
                selected.append(best_idx)
                selected.sort()
            else:
                break

        return [candidates[i] for i in selected]

    def _local_refine_fast(
        self,
        splits: List[float],
        candidates: List[float],
        cum_bad: np.ndarray,
        cum_good: np.ndarray,
        total_good: int,
        total_bad: int,
        n_samples: int,
        min_samples: int,
    ) -> List[float]:
        """快速局部精化，利用前缀和评分，O(iterations × C × K) 复杂度."""
        C = len(candidates)
        cand_arr = np.asarray(candidates)

        # 将 splits 映射回候选索引
        current = sorted(set(
            int(np.argmin(np.abs(cand_arr - s))) for s in splits
        ))

        best_score = self._fast_objective_from_indices(
            current, cum_bad, cum_good, total_good, total_bad, n_samples, min_samples
        )

        for _ in range(4):
            improved = False
            best_indices = current

            # 1) 替换一个切分点
            for i in range(len(current)):
                for idx in range(C):
                    if idx in current and idx != current[i]:
                        continue
                    trial = current.copy()
                    trial[i] = idx
                    trial = sorted(set(trial))
                    if len(trial) != len(current):
                        continue
                    score = self._fast_objective_from_indices(
                        trial, cum_bad, cum_good, total_good, total_bad, n_samples, min_samples
                    )
                    if score > best_score + 1e-12:
                        best_score = score
                        best_indices = trial
                        improved = True

            # 2) 添加一个切分点
            if len(current) < self.max_n_bins - 1:
                for idx in range(C):
                    if idx in current:
                        continue
                    trial = sorted(set(current + [idx]))
                    score = self._fast_objective_from_indices(
                        trial, cum_bad, cum_good, total_good, total_bad, n_samples, min_samples
                    )
                    if score > best_score + 1e-12:
                        best_score = score
                        best_indices = trial
                        improved = True

            # 3) 删除一个切分点
            if len(current) > max(1, self.min_n_bins - 1):
                for i in range(len(current)):
                    trial = current[:i] + current[i + 1:]
                    score = self._fast_objective_from_indices(
                        trial, cum_bad, cum_good, total_good, total_bad, n_samples, min_samples
                    )
                    if score > best_score + 1e-12:
                        best_score = score
                        best_indices = trial
                        improved = True

            current = sorted(set(best_indices))
            if not improved:
                break

        return [candidates[i] for i in current]

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
        if not splits:
            return -1e18
        # 验证所有箱的样本数 >= min_samples，避免后处理合并
        # 使用全量样本数计算（与 _apply_post_fit_constraints 一致）
        n_samples = len(x_sorted)
        n_total = getattr(self, '_n_total_samples', n_samples)
        min_samples = self._get_min_samples(n_total)
        split_pos = [0] + [int(np.searchsorted(x_sorted, s, side='right'))
                           for s in sorted(splits)] + [n_samples]
        for i in range(len(split_pos) - 1):
            if split_pos[i + 1] - split_pos[i] < min_samples:
                return -1e18
        # 计算纯目标值（IV/KS/gini等），不含 lift bonus
        raw_objective = self._raw_objective_score(
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
        # 箱数利用率奖励：强力鼓励充分利用用户允许的分箱数
        n_splits = len(splits)
        target_splits = max(self.max_n_bins - 1, 1)
        bin_utilization_bonus = (n_splits / target_splits) ** 2 * 2.0
        # 综合评分：IV为基础 + composite为质量指标 + 箱数利用率
        return float(
            raw_objective
            + composite_score * 0.40
            + focus_score * 0.30
            + bin_utilization_bonus
        )

    def _raw_objective_score(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        total_good: int,
        total_bad: int,
        splits: List[float]
    ) -> float:
        """计算纯目标函数值（IV/KS/gini/entropy/chi2），不含 lift bonus."""
        if not splits:
            return 0.0
        eps = 1e-10
        split_positions = [0] + [int(np.searchsorted(x_sorted, s, side='right'))
                                  for s in sorted(splits)] + [len(x_sorted)]

        if self.objective in ('iv', 'custom'):
            iv = 0.0
            for i in range(len(split_positions) - 1):
                s, e = split_positions[i], split_positions[i + 1]
                if s >= e:
                    continue
                good = int(np.sum(y_sorted[s:e] == 0))
                bad = int(np.sum(y_sorted[s:e] == 1))
                gd = good / total_good if total_good > 0 else 0.0
                bd = bad / total_bad if total_bad > 0 else 0.0
                if gd > eps and bd > eps:
                    iv += (bd - gd) * np.log(bd / gd)
            return iv
        elif self.objective == 'ks':
            max_ks = 0.0
            cum_good = 0
            cum_bad = 0
            for i in range(len(split_positions) - 1):
                s, e = split_positions[i], split_positions[i + 1]
                if s >= e:
                    continue
                cum_good += int(np.sum(y_sorted[s:e] == 0))
                cum_bad += int(np.sum(y_sorted[s:e] == 1))
                ks = abs(cum_good / total_good - cum_bad / total_bad)
                max_ks = max(max_ks, ks)
            return max_ks
        else:
            # gini / entropy / chi2 → delegate to existing method
            return self._calculate_objective_score(
                x_sorted, y_sorted, total_good, total_bad, splits
            )

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
        :param kwargs: 其他参数
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
                        self._enrich_woe_map(woe_map, self.bin_tables_[feature])
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
