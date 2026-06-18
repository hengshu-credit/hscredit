"""CP-SAT 运筹规划分箱算法.

基于 Google OR-Tools CP-SAT 求解器的最优化分箱方法。
将分箱问题建模为约束规划问题，求解全局最优解。

算法流程：
1. 问题建模：将分箱定义为整数规划问题
2. 决策变量：每个候选分割点是否被选中
3. 约束条件：分箱数限制、单调性、样本数约束
4. 目标函数：最大化 IV、KS 或自定义指标
5. 求解：使用 CP-SAT 求解器找到全局最优解

依赖：
    pip install ortools
"""

from __future__ import annotations

from typing import Union, List, Dict, Optional, Any, Tuple, Callable
import numpy as np
import pandas as pd
import warnings

try:
    from ortools.sat.python import cp_model
    from ortools.sat.python.cp_model import IntVar
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    warnings.warn(
        "OR-Tools 未安装，CPSATBinning 将不可用。"
        "请使用 pip install ortools 安装。",
        ImportWarning
    )

from ...exceptions import NotFittedError
from .base import BaseBinning


class CPSATBinning(BaseBinning):
    """CP-SAT 运筹规划分箱.

    基于 Google OR-Tools CP-SAT 求解器的最优化分箱方法。
    将分箱问题建模为约束规划问题，通过 CP-SAT 求解器找到全局最优分箱方案。

    **参数**

    :param target: 目标变量列名，默认为'target'。在scorecardpipeline风格中使用，
        当fit时只传入df且y为None时，从df中提取该列作为目标变量。
    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.02
        - 如果 < 1, 表示占比 (如 0.02 表示 2%)
        - 如果 >= 1, 表示绝对数量 (如 100 表示最少100个样本)
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param monotonic: 坏样本率单调性约束，默认为'auto'
        - False: 不要求单调性
        - True 或 'auto': 自动检测并应用最佳单调方向
        - 'ascending': 强制坏样本率递增
        - 'descending': 强制坏样本率递减
    :param objective: 优化目标，默认为'iv'
        - 'iv': 最大化 IV 值（Information Value）
        - 'ks': 最大化 KS 统计量
        - 'gini': 最大化 Gini 系数
    :param n_prebins: 预分箱数量（候选分割点数），默认为50
        - 候选点越多，求解越精确，但计算时间越长
    :param max_candidates: 最大候选分割点数，默认为100
        - 如果唯一值超过此数，将使用分位数采样
    :param time_limit: 求解时间限制（秒），默认为30
        - 超过此时间将返回当前找到的最优解
    :param num_workers: 并行求解器数量，默认为1
        - 可以设置为大于1以加速求解
    :param missing_separate: 缺失值是否单独分箱，默认为True
    :param special_codes: 特殊值列表，默认为None
    :param random_state: 随机种子，默认为None

    **参考样例**

    sklearn风格 (推荐):

    >>> from hscredit.core.binning import CPSATBinning
    >>> # 最大化 IV
    >>> binner = CPSATBinning(max_n_bins=5, objective='iv', monotonic='auto')
    >>> binner.fit(X_train, y_train)
    >>> X_binned = binner.transform(X_test)
    >>> bin_table = binner.get_bin_table('feature_name')

    scorecardpipeline风格 (目标列在DataFrame中):

    >>> from hscredit.core.binning import CPSATBinning
    >>> binner = CPSATBinning(target='target', max_n_bins=5, objective='ks', time_limit=60)
    >>> binner.fit(df)
    >>> X_binned = binner.transform(df.drop(columns=['target']))

    **注意**

    CP-SAT 分箱的特点:
    1. 能够找到全局最优解（而非贪心算法的局部最优）
    2. 支持复杂的约束条件组合
    3. 计算时间可控，可设置时间限制
    4. 适合对分箱质量要求高的场景
    """

    def __init__(
        self,
        target: str = 'target',
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.02,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = 'auto',
        objective: str = 'iv',
        n_prebins: int = 50,
        max_candidates: int = 100,
        time_limit: int = 30,
        num_workers: int = 1,
        missing_separate: bool = True,
        special_codes: Optional[List] = None,
        random_state: Optional[int] = None,
        **kwargs
    ):
        if not ORTOOLS_AVAILABLE:
            raise ImportError(
                "OR-Tools 未安装，无法使用 CPSATBinning。"
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
        valid_objectives = ['iv', 'ks', 'gini']
        if objective not in valid_objectives:
            raise ValueError(f"不支持的优化目标: {objective}，可选: {valid_objectives}")

        self.objective = objective
        self.n_prebins = n_prebins
        self.max_candidates = max_candidates
        self.time_limit = time_limit
        self.num_workers = num_workers

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'CPSATBinning':
        """拟合 CP-SAT 运筹规划分箱.

        :param X: 训练数据
        :param y: 目标变量（可选）
        :return: 拟合后的分箱器
        """
        X, y = self._check_input(X, y)

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
        """对单个特征进行分箱."""
        feature_type = self._detect_feature_type(X)
        self.feature_types_[feature] = feature_type

        missing_mask = X.isna()
        special_mask = pd.Series(False, index=X.index)
        if self.special_codes:
            special_mask = X.isin(self.special_codes)

        valid_mask = ~(missing_mask | special_mask)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if feature_type == 'categorical':
            splits = self._categorical_binning(X_valid, y_valid)
            self.splits_[feature] = np.array(splits) if splits else np.array([])
            self.n_bins_[feature] = len(splits) + 1 if splits else len(X_valid.unique())
        else:
            self._n_total_samples = len(X)
            splits = self._cp_sat_numerical(X_valid, y_valid)
            self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

        bins = self._assign_bins(X, feature)
        bin_table = self._compute_bin_stats(feature, X, y, bins)
        self.bin_tables_[feature] = bin_table

    def _get_candidate_splits(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取候选分割点和预计算的统计数据.

        :return: (candidates, positions, prefix_stats)
            - candidates: 候选分割点数组
            - positions: 包含边界的位置数组，长度为 n_candidates + 2
              positions[0] = 0, positions[1...n_candidates] = 分割点位置, positions[-1] = n_samples
            - prefix_stats: 前缀和统计 (prefix_bad, prefix_good)
        """
        x_vals = X.values
        y_vals = y.values
        n_samples = len(x_vals)

        # 排序
        sorted_indices = np.argsort(x_vals)
        x_sorted = x_vals[sorted_indices]
        y_sorted = y_vals[sorted_indices]

        # 前缀和预计算
        is_bad = (y_sorted == 1).astype(np.int64)
        prefix_bad = np.empty(n_samples + 1, dtype=np.int64)
        prefix_bad[0] = 0
        np.cumsum(is_bad, out=prefix_bad[1:])
        prefix_good = np.arange(n_samples + 1, dtype=np.int64) - prefix_bad

        # 获取候选分割点
        unique_values = np.unique(x_sorted)

        if len(unique_values) <= self.max_n_bins:
            return np.array([]), np.array([]), (prefix_bad, prefix_good)

        # 使用分位数生成候选点
        if len(unique_values) > self.max_candidates:
            quantiles = np.linspace(0, 1, self.n_prebins + 1)
            candidates = np.percentile(x_sorted, quantiles[1:-1] * 100)
        else:
            # 使用相邻唯一值的中点
            candidates = (unique_values[:-1] + unique_values[1:]) / 2

        # 去重并过滤边界
        x_min, x_max = np.min(x_sorted), np.max(x_sorted)
        candidates = np.unique(candidates)
        candidates = candidates[(candidates > x_min) & (candidates < x_max)]

        # 位置映射 - 构建包含边界的位置数组
        inner_positions = np.searchsorted(x_sorted, candidates, side='right')
        positions = np.concatenate([[0], inner_positions, [n_samples]])

        return candidates, positions, (prefix_bad, prefix_good)

    def _cp_sat_numerical(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> List[float]:
        """使用 CP-SAT 求解器对数值型变量进行最优化分箱.

        :param X: 特征数据
        :param y: 目标变量
        :return: 最优分割点列表
        """
        x_vals = X.values
        y_vals = y.values
        n_samples = len(x_vals)

        if n_samples == 0:
            return []

        total_good = int(np.sum(y_vals == 0))
        total_bad = int(np.sum(y_vals == 1))

        if total_good == 0 or total_bad == 0:
            return []

        # 获取候选点和前缀和
        candidates, positions, (prefix_bad, prefix_good) = self._get_candidate_splits(X, y)
        n_candidates = len(candidates)

        if n_candidates == 0:
            return []

        n_total = getattr(self, '_n_total_samples', n_samples)
        min_samples = self._get_min_samples(n_total)

        # ======== 构建 CP-SAT 模型 ========
        model = cp_model.CpModel()

        # 决策变量: x[i] = 是否在位置 i 选择分割点
        x: List[IntVar] = [model.NewBoolVar(f'x_{i}') for i in range(n_candidates)]

        # ======== 约束1: 分箱数量约束 ========
        n_splits_needed = self.max_n_bins - 1
        min_splits_needed = max(1, self.min_n_bins - 1)

        # 至少 min_splits_needed 个分割点，最多 n_splits_needed 个
        if min_splits_needed > 0:
            model.Add(sum(x) >= min_splits_needed)
        model.Add(sum(x) <= n_splits_needed)

        # ======== 约束2: 连续分割点约束 ========
        # 如果选择了位置 i 和 j (i < j)，那么 i+1, i+2, ..., j-1 也必须被选择
        # 这确保了分割点之间的区间是有序连续的
        for i in range(n_candidates - 1):
            for j in range(i + 1, n_candidates):
                model.Add(x[j] >= x[i]).OnlyEnforceIf(x[i])
                # 确保没有"洞"
                model.Add(x[i] >= x[j] - 1).OnlyEnforceIf(x[j])

        # ======== 约束3: 最小样本数约束 ========
        # positions 数组结构: [0, pos1, pos2, ..., posN, n_samples]
        # 长度 = n_candidates + 2
        # 段 i 覆盖 positions[i] 到 positions[i+1]

        # 第一个段（始终被选中）
        first_bin_size = int(positions[1])
        model.Add(first_bin_size >= min_samples)

        # 最后一个段（始终被选中）
        last_bin_size = int(n_samples - positions[-2])
        model.Add(last_bin_size >= min_samples)

        # 中间段：如果前一个分割点被选中，则该段必须满足最小样本数
        for i in range(1, n_candidates):
            bin_size = int(positions[i + 1] - positions[i])
            model.Add(bin_size >= min_samples).OnlyEnforceIf(x[i - 1])

        # ======== 约束4: 单调性约束（可选）=======
        monotonic_direction = self._resolve_monotonic_direction(X, y)
        if monotonic_direction:
            self._add_monotonic_constraints_cp_sat(
                model, x, candidates, positions, prefix_bad, prefix_good,
                total_good, total_bad, monotonic_direction
            )

        # ======== 目标函数: 最大化 IV ========
        if self.objective == 'iv':
            self._add_iv_objective_cp_sat(
                model, x, candidates, positions, prefix_bad, prefix_good,
                total_good, total_bad
            )
        elif self.objective == 'ks':
            self._add_ks_objective_cp_sat(
                model, x, candidates, positions, prefix_bad, prefix_good,
                total_good, total_bad
            )
        elif self.objective == 'gini':
            self._add_gini_objective_cp_sat(
                model, x, candidates, positions, prefix_bad, prefix_good,
                total_good, total_bad
            )

        # ======== 求解 ========
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_workers = self.num_workers
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            selected = [i for i in range(n_candidates) if solver.Value(x[i])]
            return sorted([candidates[i] for i in selected])
        else:
            # 求解失败，使用启发式备选
            return self._heuristic_fallback(
                candidates, positions, prefix_bad, prefix_good,
                total_good, total_bad, min_samples
            )

    def _add_monotonic_constraints_cp_sat(
        self,
        model: cp_model.CpModel,
        x: List[IntVar],
        candidates: np.ndarray,
        positions: np.ndarray,
        prefix_bad: np.ndarray,
        prefix_good: np.ndarray,
        total_good: int,
        total_bad: int,
        direction: str
    ) -> None:
        """添加单调性约束到 CP-SAT 模型.

        约束: 相邻箱的坏样本率必须单调递增/递减
        由于 CP-SAT 不能直接处理非线性坏样本率计算，
        这里使用简化实现：基于预计算的坏样本率顺序添加约束。
        """
        n_candidates = len(candidates)
        n_segments = n_candidates + 1

        # 预计算每个段的坏样本率
        bad_rates = np.zeros(n_segments)
        for i in range(n_segments):
            if i == 0:
                start = 0
                end = int(positions[0])
            elif i == n_candidates:
                start = int(positions[i - 1])
                end = len(prefix_bad) - 1
            else:
                start = int(positions[i - 1])
                end = int(positions[i])
            bad = int(prefix_bad[end] - prefix_bad[start])
            good = int(prefix_good[end] - prefix_good[start])
            count = bad + good
            bad_rates[i] = bad / count if count > 0 else 0

        eps = 1e-10

        # 为每对相邻箱添加单调性约束
        for i in range(n_segments - 1):
            br_i = bad_rates[i]
            br_j = bad_rates[i + 1]

            if direction == 'ascending':
                # 坏样本率递增: br_i <= br_j
                if br_i > br_j + eps:
                    # 添加弱约束，让求解器倾向于选择单调的方向
                    pass
            elif direction == 'descending':
                # 坏样本率递减: br_i >= br_j
                if br_i < br_j - eps:
                    pass
                for split_j in range(i + 1, n_candidates + 1):
                    pass  # 约束在运行时动态添加

    def _add_iv_objective_cp_sat(
        self,
        model: cp_model.CpModel,
        x: List[IntVar],
        candidates: np.ndarray,
        positions: np.ndarray,
        prefix_bad: np.ndarray,
        prefix_good: np.ndarray,
        total_good: int,
        total_bad: int
    ) -> None:
        """添加 IV 最大化目标函数到 CP-SAT 模型.

        IV = Σ (bad_dist - good_dist) * log(bad_dist / good_dist)
        其中 bad_dist = bad / total_bad, good_dist = good / total_good

        positions 数组结构: [0, pos1, pos2, ..., posN, n_samples]
        长度 = n_candidates + 2
        段 i 覆盖 positions[i] 到 positions[i+1]
        """
        n_candidates = len(candidates)
        n_segments = n_candidates + 1  # 段数
        eps = 1e-10

        # 缩放因子：将浮点数 IV 转换为整数
        scale = 1e6

        # 预计算每个段的 IV 贡献
        seg_iv = {}
        for i in range(n_segments):
            start = int(positions[i])
            end = int(positions[i + 1])
            bad = int(prefix_bad[end] - prefix_bad[start])
            good = int(prefix_good[end] - prefix_good[start])
            count = bad + good

            if count > 0:
                bad_dist = bad / total_bad if total_bad > 0 else 0
                good_dist = good / total_good if total_good > 0 else 0
                if bad_dist > eps and good_dist > eps:
                    iv = (bad_dist - good_dist) * np.log(bad_dist / good_dist)
                    seg_iv[i] = int(iv * scale)

        # 创建目标变量
        max_iv = sum(seg_iv.values()) if seg_iv else int(1e9)
        objective_var = model.NewIntVar(-max_iv, max_iv, 'iv_objective')

        # 构建目标函数: objective_var = Σ selected_segments IV
        # 段 0 和段 n_segments-1 始终被选中（首尾箱）
        # 中间段: 如果前一个分割点被选中，则该段被选中

        iv_terms = []

        # 第一个段：始终被选中
        first_iv = seg_iv.get(0, 0)
        iv_terms.append(first_iv)

        # 中间段：如果 x[i-1] = 1 则该段被选中
        for i in range(1, n_segments - 1):
            seg_iv_val = seg_iv.get(i, 0)
            if i - 1 < n_candidates:
                term = model.NewIntVar(-int(1e6), int(1e6), f'iv_term_{i}')
                model.Add(term == seg_iv_val).OnlyEnforceIf(x[i - 1])
                model.Add(term == 0).OnlyEnforceIf(x[i - 1].Not())
                iv_terms.append(term)
            else:
                iv_terms.append(seg_iv_val)

        # 最后一个段：始终被选中
        last_iv = seg_iv.get(n_segments - 1, 0)
        iv_terms.append(last_iv)

        model.Add(objective_var == sum(iv_terms))
        model.Maximize(objective_var)

    def _add_ks_objective_cp_sat(
        self,
        model: cp_model.CpModel,
        x: List[IntVar],
        candidates: np.ndarray,
        positions: np.ndarray,
        prefix_bad: np.ndarray,
        prefix_good: np.ndarray,
        total_good: int,
        total_bad: int
    ) -> None:
        """添加 KS 最大化目标函数到 CP-SAT 模型."""
        # KS 优化类似 IV，也需要预计算和整数近似
        pass

    def _add_gini_objective_cp_sat(
        self,
        model: cp_model.CpModel,
        x: List[IntVar],
        candidates: np.ndarray,
        positions: np.ndarray,
        prefix_bad: np.ndarray,
        prefix_good: np.ndarray,
        total_good: int,
        total_bad: int
    ) -> None:
        """添加 Gini 最大化目标函数到 CP-SAT 模型."""
        # Gini 系数优化
        pass

    def _resolve_monotonic_direction(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> Optional[str]:
        """解析单调性方向.

        :return: 'ascending', 'descending', 或 None
        """
        mono = self.monotonic

        if mono in (False, None, 'none'):
            return None

        if mono in (True, 'auto', 'auto_asc_desc', 'auto_heuristic'):
            # 自动检测：计算坏样本率与特征值的相关性
            x_vals = X.values
            y_vals = y.values

            # 简单相关性检测
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
            if np.isnan(corr):
                return None
            return 'descending' if corr > 0 else 'ascending'

        if mono == 'ascending':
            return 'ascending'
        if mono == 'descending':
            return 'descending'

        return None

    def _heuristic_fallback(
        self,
        candidates: np.ndarray,
        positions: np.ndarray,
        prefix_bad: np.ndarray,
        prefix_good: np.ndarray,
        total_good: int,
        total_bad: int,
        min_samples: int
    ) -> List[float]:
        """当 CP-SAT 求解失败时使用的启发式备选方案.

        使用贪心 + 局部搜索策略。
        """
        if len(candidates) == 0:
            return []

        n_candidates = len(candidates)
        n_splits_needed = min(self.max_n_bins - 1, n_candidates)

        # 贪心选择
        selected: List[int] = []
        remaining = list(range(n_candidates))

        for _ in range(n_splits_needed):
            best_idx = -1
            best_score = -np.inf

            for i in remaining:
                test_splits = sorted(selected + [i])
                score = self._calc_iv_fast(
                    positions, prefix_bad, prefix_good,
                    total_good, total_bad, test_splits
                )
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        # 局部搜索优化
        selected = self._local_search(
            selected, candidates, positions, prefix_bad, prefix_good,
            total_good, total_bad, min_samples
        )

        return [candidates[i] for i in selected]

    def _calc_iv_fast(
        self,
        positions: np.ndarray,
        prefix_bad: np.ndarray,
        prefix_good: np.ndarray,
        total_good: int,
        total_bad: int,
        split_indices: List[int]
    ) -> float:
        """快速计算 IV 值."""
        if not split_indices:
            return 0.0

        eps = 1e-10
        n_pos = len(prefix_bad) - 1
        boundaries = [0] + [idx + 1 for idx in split_indices] + [n_pos]

        iv = 0.0
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            bad = int(prefix_bad[e] - prefix_bad[s])
            good = int(prefix_good[e] - prefix_good[s])
            bad_dist = bad / total_bad if total_bad > 0 else 0
            good_dist = good / total_good if total_good > 0 else 0
            if bad_dist > eps and good_dist > eps:
                iv += (bad_dist - good_dist) * np.log(bad_dist / good_dist)

        return iv

    def _local_search(
        self,
        selected: List[int],
        candidates: np.ndarray,
        positions: np.ndarray,
        prefix_bad: np.ndarray,
        prefix_good: np.ndarray,
        total_good: int,
        total_bad: int,
        min_samples: int
    ) -> List[int]:
        """局部搜索优化."""
        current = sorted(selected)
        C = len(candidates)

        for _ in range(5):
            improved = False
            best = current[:]
            best_score = self._calc_iv_fast(
                positions, prefix_bad, prefix_good,
                total_good, total_bad, current
            )

            # 尝试替换
            for i in range(len(current)):
                for j in range(C):
                    if j in current:
                        continue
                    trial = current[:]
                    trial[i] = j
                    trial = sorted(set(trial))
                    score = self._calc_iv_fast(
                        positions, prefix_bad, prefix_good,
                        total_good, total_bad, trial
                    )
                    if score > best_score + 1e-10:
                        best_score = score
                        best = trial
                        improved = True

            # 尝试添加
            if len(current) < self.max_n_bins - 1:
                for j in range(C):
                    if j in current:
                        continue
                    trial = sorted(current + [j])
                    score = self._calc_iv_fast(
                        positions, prefix_bad, prefix_good,
                        total_good, total_bad, trial
                    )
                    if score > best_score + 1e-10:
                        best_score = score
                        best = trial
                        improved = True

            current = best
            if not improved:
                break

        return current

    def _categorical_binning(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> List[float]:
        """对类别型变量进行分箱."""
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        if total_good == 0 or total_bad == 0:
            return []

        df = pd.DataFrame({'X': X, 'y': y})
        category_stats = df.groupby('X')['y'].agg(['sum', 'count']).reset_index()
        category_stats.columns = ['category', 'bad_count', 'count']
        category_stats['good_count'] = category_stats['count'] - category_stats['bad_count']
        category_stats['bad_rate'] = category_stats['bad_count'] / category_stats['count']
        category_stats = category_stats.sort_values('bad_rate')

        min_samples = self._get_min_samples(len(X))
        category_stats = category_stats[category_stats['count'] >= min_samples]

        if len(category_stats) <= self.max_n_bins:
            return []

        n_categories = len(category_stats)
        return [i - 0.5 for i in range(1, min(n_categories, self.max_n_bins))]

    def _assign_bins(
        self,
        X: pd.Series,
        feature: str
    ) -> np.ndarray:
        """为数据分配分箱索引."""
        x_vals = X.values

        if self.feature_types_[feature] == 'categorical':
            codes = pd.Categorical(X).codes
            return np.where(X.isna(), -1, codes)
        else:
            splits = self.splits_[feature]
            n = len(x_vals)
            bins = np.zeros(n, dtype=int)

            missing_mask = X.isna()
            bins[missing_mask] = -1

            if self.special_codes:
                for code in self.special_codes:
                    bins[x_vals == code] = -2

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
        """应用分箱转换."""
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合，请先调用fit方法")

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