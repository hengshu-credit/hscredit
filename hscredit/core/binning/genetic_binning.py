"""遗传算法分箱.

使用遗传算法进行全局最优分箱搜索，支持复杂约束条件。
适用于需要全局优化且约束复杂的场景。
"""

import logging
from typing import Callable, Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from ...exceptions import NotFittedError
from ...utils.parallel import ParallelWorkload
from .base import BaseBinning
from ._objectives import resolve_objective, evaluate_callback
from ._contracts import BinningConstraintViolation

logger = logging.getLogger(__name__)


class GeneticBinning(BaseBinning):
    """遗传算法分箱.

    使用遗传算法在候选切分点中搜索最优分箱方案，支持多种优化目标。
    通过交叉、变异、选择等遗传操作，逐步逼近全局最优解。

    **参数**

    :param custom_objective: 兼容回调入口，仅与 objective='custom' 配合；也可直接传 objective=函数。
    :type custom_objective: Optional[Callable]
    :param woe_clip: 可选的正数 WOE 截断阈值；统计表和自定义目标统计使用相同截断，默认 None 不截断。
    :type woe_clip: Optional[float]

    :param objective_params: 预定义目标的权重，或传给自定义目标函数的附加关键字参数，默认 None。
    :type objective_params: Optional[Dict]

    ``objective`` 可为原生目标名称、``CustomObjectives.names()`` 中的复合目标名称，
    或 ``callable(bin_stats, total_good, total_bad, **objective_params)``。回调须返回有限实数，越大越好。
    回调统计复用 ``compute_bin_stats``，``iv`` 为分档 IV，``woe`` 为 log(坏分布/好分布)。
    原 ``objective="custom", custom_objective=函数`` 写法继续可用。

    :param constraint_policy: 约束冲突策略，默认 ``"raise"`` 报错；``"legacy"`` 在约束不可行时
        保留原生尽力满足的规则（优化器无可行分区时可回退单箱）。不放宽输入校验或吞掉回调错误。
        实际未满足项记录于 ``constraint_violations_``，不增加业务统计表列。
    :type constraint_policy: str

    :param target: 目标变量列名，默认为'target'。在scorecardpipeline风格中使用，
        当fit时只传入df且y为None时，从df中提取该列作为目标变量。

    :type target: str
    :param cat_cutoff: 数值字段转为类别字段的识别阈值，默认 ``None`` （数值字段保持数值型）。
        ``0 < cat_cutoff < 1`` 时，非缺失唯一值数 / 非缺失样本数不超过该比例即按类别处理；
        ``cat_cutoff >= 1`` 时，唯一值数不超过 ``int(cat_cutoff)`` 即按类别处理。
        字符串、category 和布尔字段本身按类别处理；该参数不用于合并低频类别。

    :type cat_cutoff: Optional[Union[float, int]]
    :param category_order: 类别顺序，默认 ``None`` 按训练集坏样本率升序排列，同率按首次出现顺序。
        可传 ``{字段名: 完整类别序列}`` 或 ``callable(feature, x, y)`` ；
        顺序必须覆盖该字段全部非缺失训练类别，不能重复或包含未知类别。

    :param handle_unknown: 预测期未知类别策略，默认 ``-3`` ； ``"value"`` 等价于 ``-3`` ，
        返回未知箱（标签 ``"unknown"`` ，WOE 为 ``0.0`` ）； ``"raise"`` 抛出错误。
        也可传入已记录的整数箱号，将未知类别归入该箱；不接受布尔值。

    :type handle_unknown: Union[int, str]
    :param decimal: 数值型分箱标签显示精度，默认为4；实际切点不舍入
    :type decimal: int
    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :type n_jobs: Union[int, float]
    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :type parallel_backend: Optional[str]
    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :type parallel_config: Optional[Dict[str, Any]]
    :param user_splits: 自定义规则字典或返回规则的可调用对象，默认 ``None`` 。数值规则为严格递增的
        有限内部切点（不含正负无穷）；类别规则为完整覆盖训练类别的嵌套列表。
        缺失标记最多出现一次，可独立成箱或加入指定箱。

    :type user_splits: Optional[Dict[str, List]]
    :param user_splits_fixed: 固定用户节点，默认 ``None`` 表示不固定；可传全局布尔值或
        ``{字段名: 布尔值或布尔序列}`` 。数值序列长度须等于有限切点数；
        类别序列长度须等于用户分组数。 ``True`` 保留节点， ``False`` 允许合并。

    :type user_splits_fixed: Optional[Union[bool, Dict[str, Union[bool, List[bool]]]]]

    :param population_size: 种群大小，默认为50
    :type population_size: int
    :param generations: 迭代代数，默认为100
    :type generations: int
    :param mutation_rate: 变异率，默认为0.1
    :type mutation_rate: float
    :param crossover_rate: 交叉率，默认为0.8
    :type crossover_rate: float
    :param elitism_rate: 精英保留率（每代直接保留的最优个体比例），默认为0.1
    :type elitism_rate: float
    :param objective: 适应度（优化目标）函数，默认为 ``'iv'`` 。可取以下枚举值：

        - ``'iv'`` ：最大化 Information Value（信息价值），最常用，衡量整体预测力

        - ``'ks'`` ：最大化 KS 统计量，衡量好坏样本累积分布的最大区分度

        - ``'gini'`` ：最大化 Gini 系数（≈ 2×AUC−1），衡量排序区分能力

    :type objective: Union[str, Callable]

    :param max_n_bins: 最大分箱数，默认为5
    :type max_n_bins: int
    :param min_n_bins: 最小分箱数，默认为2
    :type min_n_bins: int
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :type min_bin_size: Union[float, int]
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :type max_bin_size: Optional[Union[float, int]]
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :type min_bad_rate: float
    :param monotonic: 是否要求坏样本率单调，默认为False
    :type monotonic: Union[bool, str]
    :param special_codes: 特殊值列表，默认为None
    :type special_codes: Optional[List]
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :type missing_separate: bool
    :param random_state: 随机种子，默认为None
    :type random_state: Optional[int]
    :param verbose: 是否输出详细信息，默认为False
    :type verbose: Union[bool, int]

    **参考样例**

    >>> from hscredit.core.binning import GeneticBinning
    >>> binner = GeneticBinning(objective='iv', max_n_bins=5, generations=100)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)

    **注意**

    遗传算法为随机全局优化，结果依赖 ``random_state`` ；相比贪心方法更可能跳出局部最优，
    但计算开销随 ``population_size × generations`` 增长，适合约束复杂、对分箱质量要求高的场景。

    **引用**

    遗传算法：Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems.*
    https://en.wikipedia.org/wiki/Genetic_algorithm
    """

    def __init__(
        self,
        target: str = "target",
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_rate: float = 0.1,
        objective: Union[str, Callable] = "iv",
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        cat_cutoff: Optional[Union[float, int]] = None,
        category_order=None,
        handle_unknown: Union[int, str] = -3,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
        decimal: int = 4,
        n_jobs: Union[int, float] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        user_splits: Optional[Dict[str, List]] = None,
        user_splits_fixed: Optional[Union[bool, Dict[str, Union[bool, List[bool]]]]] = None,
        objective_params: Optional[Dict[str, Any]] = None,
        custom_objective: Optional[Callable] = None,
        woe_clip: Optional[float] = None,
        constraint_policy: str = "raise",
    ):
        super().__init__(
            target=target,
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_bad_rate=min_bad_rate,
            monotonic=monotonic,
            special_codes=special_codes,
            missing_separate=missing_separate,
            cat_cutoff=cat_cutoff,
            user_splits=user_splits,
            user_splits_fixed=user_splits_fixed,
            category_order=category_order,
            handle_unknown=handle_unknown,
            random_state=random_state,
            verbose=verbose,
            decimal=decimal,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            constraint_policy=constraint_policy,
            woe_clip=woe_clip,
        )
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.objective = objective
        self.objective_params = objective_params
        self.custom_objective = custom_objective
        self._objective_name, self._objective_callback = resolve_objective(
            objective, objective_params, custom_objective, native_names=("iv", "ks", "gini")
        )

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None, **kwargs
    ) -> "GeneticBinning":
        """拟合遗传算法分箱。

        对每个特征以遗传算法（选择/交叉/变异/精英保留）在候选切分点中搜索使 ``objective``
        最大化的分箱方案。支持 sklearn 与 scorecardpipeline 两种调用风格，详见
        :meth:`BaseBinning.fit` 。

        :param X: 训练数据，shape ``(n_samples, n_features)`` ，DataFrame 或 ndarray
        :type X: Union[pd.DataFrame, np.ndarray]
        :param y: 默认值为 ``None`` 。二分类目标变量（0=好/1=坏）；scorecardpipeline 风格下可省略
        :type y: Optional[Union[pd.Series, np.ndarray]]
        :param kwargs: 透传给基类的其他参数
        :return: 拟合后的分箱器自身（便于链式调用）
        """
        X, y = self._check_input(X, y)
        self._n_total_samples = len(X)

        self._fit_features(X, y, "_fit_feature")

        self._finalize_categorical_fit()
        self._finalize_reserved_bins(X, y)
        self._is_fitted = True
        return self

    def _fit_feature(self, feature: str, X: pd.Series, y: pd.Series) -> None:
        """拟合单个特征。"""
        if self.verbose:
            logger.info(f"处理特征: {feature}")
        feature_type = self._detect_feature_type(X)
        self.feature_types_[feature] = feature_type
        if feature_type == "categorical":
            splits = self._fit_categorical(X, y)
            self.splits_[feature] = splits
        else:
            splits = self._fit_numerical(X, y)
            self.splits_[feature] = self._round_splits(splits)
        self.n_bins_[feature] = len(splits) + 1
        bins = self._apply_bins(X, splits)
        self.bin_tables_[feature] = self._compute_bin_stats(feature, X, y, bins)
        if self._objective_callback is not None:
            valid = np.asarray(bins) >= 0
            self.solver_info_ = getattr(self, "solver_info_", {})
            self.solver_info_[feature] = {
                "后端": "genetic",
                "求解状态": "HEURISTIC",
                "目标函数": self._objective_name,
                "目标值": (
                    evaluate_callback(
                        self._objective_callback, np.asarray(bins)[valid], np.asarray(y)[valid], woe_clip=self.woe_clip
                    )
                    if valid.any()
                    else None
                ),
            }

    def _fit_numerical(self, x: pd.Series, y: pd.Series) -> np.ndarray:
        """对数值型特征进行遗传算法分箱."""
        x_clean = x.copy()
        mask = x_clean.notna()

        if self.special_codes:
            for code in self.special_codes:
                mask = mask & (x_clean != code)

        x_valid = x_clean[mask]
        y_valid = y[mask]

        if len(x_valid) == 0:
            return np.array([])

        if x_valid.nunique() == 1:
            if self._objective_callback is not None and self._check_constraints(x_valid, y_valid, []):
                evaluate_callback(
                    self._objective_callback, np.zeros(len(x_valid), dtype=int), y_valid, woe_clip=self.woe_clip
                )
            return np.array([])

        # 生成候选切分点（分位数）
        candidate_splits = self._generate_candidates(x_valid)

        if len(candidate_splits) == 0:
            return np.array([])
        if len(candidate_splits) < self.min_n_bins - 1:
            return np.array([])

        # 运行遗传算法
        best_splits = self._genetic_algorithm(x_valid, y_valid, candidate_splits)

        return best_splits

    def _generate_candidates(self, x: pd.Series) -> np.ndarray:
        """生成候选切分点."""
        n_candidates = min(50, len(x) // 10)
        if n_candidates < self.max_n_bins - 1:
            n_candidates = self.max_n_bins * 2

        quantiles = np.linspace(0, 1, n_candidates + 2)[1:-1]
        candidates = np.percentile(x, quantiles * 100)
        return np.unique(candidates)

    def _genetic_algorithm(self, x: pd.Series, y: pd.Series, candidates: np.ndarray) -> np.ndarray:
        """运行遗传算法搜索最优分箱."""
        rng = np.random.RandomState(self.random_state)

        # 初始化种群
        population = self._initialize_population(candidates, rng)

        best_fitness = -np.inf
        best_individual = None
        no_improvement_count = 0
        max_no_improvement = 20

        for generation in range(self.generations):
            # 评估适应度
            tasks = [(individual, x, y, candidates) for individual in population]
            data_bytes = int(x.memory_usage(deep=True) + y.memory_usage(deep=True))
            data_bytes += int(sum(individual.nbytes for individual in population))
            fitness_scores = self._parallel_execute(
                self._evaluate_fitness_task,
                tasks,
                default_backend="threading",
                task_labels=[f"第{generation + 1}代个体{index + 1}" for index in range(len(tasks))],
                workload=ParallelWorkload(
                    task_count=len(tasks),
                    rows=len(x),
                    columns=len(tasks),
                    data_bytes=data_bytes,
                    cost_per_item=8.0,
                    capability="thread_safe",
                    releases_gil=True,
                    operation="遗传分箱适应度",
                ),
            )

            # 更新最优解
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if self.verbose and generation % 10 == 0:
                logger.info(f"  Generation {generation}: Best fitness = {best_fitness:.4f}")

            # 提前停止
            if no_improvement_count >= max_no_improvement:
                if self.verbose:
                    logger.info(f"  Early stopping at generation {generation}")
                break

            # 选择
            selected = self._selection(population, fitness_scores, rng)

            # 交叉
            offspring = self._crossover(selected, rng)

            # 变异
            offspring = self._mutation(offspring, candidates, rng)

            # 精英保留
            population = self._elitism(offspring, best_individual, fitness_scores)

        if best_individual is None:
            if self._objective_callback is not None:
                raise BinningConstraintViolation("遗传搜索未找到满足约束且具有有效自定义目标评分的候选")
            return np.array([])

        # 解码最优个体
        selected_indices = np.where(best_individual)[0]
        return candidates[selected_indices]

    def _evaluate_fitness_task(self, task) -> float:
        """执行一个只读遗传适应度任务。"""
        individual, x, y, candidates = task
        return self._evaluate_fitness(individual, x, y, candidates)

    def _initialize_population(self, candidates: np.ndarray, rng: np.random.RandomState) -> List[np.ndarray]:
        """初始化种群."""
        population = []
        n_candidates = len(candidates)

        for _ in range(self.population_size):
            # 随机选择切分点数量
            n_splits = rng.randint(self.min_n_bins - 1, min(self.max_n_bins - 1, n_candidates) + 1)
            # 随机选择切分点
            individual = np.zeros(n_candidates, dtype=bool)
            selected = rng.choice(n_candidates, n_splits, replace=False)
            individual[selected] = True
            population.append(individual)

        return population

    def _evaluate_fitness(self, individual, x, y, candidates):
        selected_indices = np.flatnonzero(individual)
        if len(selected_indices) < self.min_n_bins - 1 or len(selected_indices) > self.max_n_bins - 1:
            return -np.inf
        splits = candidates[selected_indices]
        if not self._check_constraints(x, y, splits):
            return -np.inf
        bins = np.searchsorted(splits, x, side="right")
        if self._objective_callback is not None:
            return evaluate_callback(self._objective_callback, bins, y, woe_clip=self.woe_clip)
        if self._objective_name == "iv":
            return self._calculate_iv(bins, y)
        if self._objective_name == "ks":
            return self._calculate_ks(bins, y)
        return self._calculate_gini(bins, y)

    def _check_constraints(self, x, y, splits):
        """在适应度评估前使用最终区间语义检查全部公共硬约束。"""
        from ._candidate_search import _is_monotonic

        bins = np.searchsorted(splits, x, side="right")
        counts = np.bincount(bins, minlength=len(splits) + 1)
        bad = np.bincount(bins, weights=y, minlength=len(counts))
        n_total = getattr(self, "_n_total_samples", len(x)) if self._objective_callback is not None else len(x)
        maximum = self._get_max_samples(n_total)
        if np.any(counts == 0) or np.any(counts < self._get_min_samples(n_total)):
            return False
        if maximum is not None and np.any(counts > maximum):
            return False
        rates = bad / counts
        return bool(np.all(rates >= self.min_bad_rate - 1e-12) and _is_monotonic(rates, self.monotonic, 0.0))

    def _calculate_iv(self, bins: pd.Series, y: pd.Series) -> float:
        """使用共享统计内核的计数平滑口径计算原始 IV。"""
        from ..metrics import compute_bin_stats

        table = compute_bin_stats(np.asarray(bins), np.asarray(y), round_digits=False)
        return float(table["分档IV值"].sum())

    def _calculate_ks(self, bins: pd.Series, y: pd.Series) -> float:
        """在完整分箱序列上计算好坏累计分布差，纯箱也参与累计。"""
        from ..metrics import compute_bin_stats

        table = compute_bin_stats(np.asarray(bins), np.asarray(y), round_digits=False)
        return float(table["分档KS值"].max())

    def _calculate_gini(self, bins: pd.Series, y: pd.Series) -> float:
        """以箱内坏率作为风险分数，按 2 * AUC - 1 计算排序 Gini。"""
        from ..metrics import compute_bin_stats, gini

        labels = np.asarray(y)
        if len(np.unique(labels)) < 2:
            return 0.0
        table = compute_bin_stats(np.asarray(bins), labels, round_digits=False)
        rates = table.set_index("分箱")["坏样本率"]
        scores = rates.reindex(np.asarray(bins)).to_numpy(dtype=float)
        return float(gini(labels, scores, score_direction="higher_risk"))

    def _selection(
        self, population: List[np.ndarray], fitness_scores: List[float], rng: np.random.RandomState
    ) -> List[np.ndarray]:
        """选择操作（锦标赛选择）."""
        selected = []
        tournament_size = 3

        for _ in range(len(population)):
            tournament_indices = rng.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())

        return selected

    def _crossover(self, population: List[np.ndarray], rng: np.random.RandomState) -> List[np.ndarray]:
        """交叉操作."""
        offspring = []

        for i in range(0, len(population), 2):
            parent1 = population[i]
            if i + 1 < len(population):
                parent2 = population[i + 1]
            else:
                offspring.append(parent1.copy())
                continue

            if len(parent1) > 1 and len(parent1) == len(parent2) and rng.random() < self.crossover_rate:
                # 单点交叉
                crossover_point = rng.randint(1, len(parent1))
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])

        return offspring[: len(population)]

    def _mutation(
        self, population: List[np.ndarray], candidates: np.ndarray, rng: np.random.RandomState
    ) -> List[np.ndarray]:
        """变异操作."""
        mutated = []

        for individual in population:
            if rng.random() < self.mutation_rate:
                # 翻转一个随机位
                mutation_point = rng.randint(len(individual))
                individual = individual.copy()
                individual[mutation_point] = not individual[mutation_point]
            mutated.append(individual)

        return mutated

    def _elitism(
        self, population: List[np.ndarray], best_individual: np.ndarray, fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """精英保留."""
        n_elites = int(self.elitism_rate * self.population_size)

        # 选择最优个体
        elite_indices = np.argsort(fitness_scores)[-n_elites:]
        elites = [population[i].copy() for i in elite_indices]

        # 替换最差个体
        new_population = population[:-n_elites] if n_elites > 0 else population
        new_population = elites + new_population[len(elites) :]

        return new_population

    def _fit_categorical(self, x: pd.Series, y: pd.Series) -> List:
        """对类别型特征进行分箱."""
        cat_stats = pd.DataFrame({"category": x, "target": y}).groupby("category")["target"].agg(["mean", "count"])

        min_samples = self._get_min_samples(len(x))
        cat_stats = cat_stats[cat_stats["count"] >= min_samples]
        cat_stats = cat_stats.sort_values("mean")

        # 限制分箱数
        if len(cat_stats) > self.max_n_bins:
            # 合并坏样本率相近的类别
            categories = self._merge_categories(cat_stats)
        else:
            categories = cat_stats.index.tolist()

        return categories

    def _merge_categories(self, cat_stats: pd.DataFrame) -> List:
        """合并类别以满足分箱数限制."""
        categories = cat_stats.index.tolist()

        while len(categories) > self.max_n_bins:
            bad_rates = cat_stats["mean"].values
            min_diff = float("inf")
            merge_idx = 0

            for i in range(len(bad_rates) - 1):
                diff = abs(bad_rates[i] - bad_rates[i + 1])
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = i

            # 合并
            cat1 = categories[merge_idx]
            cat2 = categories[merge_idx + 1]
            merged_cat = f"{cat1},{cat2}"

            categories.pop(merge_idx + 1)
            categories[merge_idx] = merged_cat

            # 更新统计
            merged_count = cat_stats.iloc[merge_idx]["count"] + cat_stats.iloc[merge_idx + 1]["count"]
            merged_bad = (
                cat_stats.iloc[merge_idx]["mean"] * cat_stats.iloc[merge_idx]["count"]
                + cat_stats.iloc[merge_idx + 1]["mean"] * cat_stats.iloc[merge_idx + 1]["count"]
            )
            merged_rate = merged_bad / merged_count

            cat_stats = cat_stats.drop([cat1, cat2])
            cat_stats.loc[merged_cat] = {"mean": merged_rate, "count": merged_count}
            cat_stats = cat_stats.sort_values("mean")

        return categories

    def _get_min_samples(self, n_total: int) -> int:
        """使用统一的最小样本数取整口径。"""
        return super()._get_min_samples(n_total)

    def _apply_bins(self, x: pd.Series, splits: Union[np.ndarray, List]) -> np.ndarray:
        """应用分箱."""
        feature = x.name
        if feature in self._cat_bins_ and self.feature_types_.get(feature) == "categorical":
            return self._assign_categorical_bins(feature, x)
        if isinstance(splits, list):
            bins = np.zeros(len(x), dtype=int)
            for i, cat in enumerate(splits):
                if "," in str(cat):
                    cats = str(cat).split(",")
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
        self, X: Union[pd.DataFrame, np.ndarray], metric: str = "indices", **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """应用分箱转换.

        将原始特征值转换为分箱索引、分箱标签或WOE值。

        :param X: 待转换数据, DataFrame或数组格式
        :type X: Union[pd.DataFrame, np.ndarray]
        :param metric: 默认值为 ``'indices'`` 。转换类型, 可选值:

            - 'indices': 返回分箱索引 (0, 1, 2, ...), 用于后续处理

            - 'bins': 返回分箱标签字符串, 用于可视化或报告

            - 'woe': 返回WOE值, 用于逻辑回归建模

        :type metric: str
        :param kwargs: 其他参数
        :return: 转换后的数据, 格式与输入X相同

        **参考样例**

        >>> binner = GeneticBinning()
        >>> binner.fit(X_train, y_train)
        >>>
        >>> # 获取分箱索引
        >>> X_binned = binner.transform(X_test, metric='indices')
        >>>
        >>> # 获取WOE编码 (用于建模)
        >>> X_woe = binner.transform(X_test, metric='woe')
        """
        if not self._is_fitted:
            raise NotFittedError("分箱器尚未拟合")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return self._transform_binning_features(
            X,
            metric,
            lambda feature: self._apply_bins(X[feature], self.splits_[feature]),
        )
