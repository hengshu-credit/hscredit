"""遗传算法分箱.

使用遗传算法进行全局最优分箱搜索，支持复杂约束条件。
适用于需要全局优化且约束复杂的场景。
"""

from typing import Union, List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from .base import BaseBinning


class GeneticBinning(BaseBinning):
    """遗传算法分箱.

    使用遗传算法在候选切分点中搜索最优分箱方案，支持多种优化目标。
    通过交叉、变异、选择等遗传操作，逐步逼近全局最优解。

    :param population_size: 种群大小，默认为50
    :param generations: 迭代代数，默认为100
    :param mutation_rate: 变异率，默认为0.1
    :param crossover_rate: 交叉率，默认为0.8
    :param elitism_rate: 精英保留率，默认为0.1
    :param objective: 优化目标，默认为'iv'，可选'iv', 'ks', 'gini'
    :param max_n_bins: 最大分箱数，默认为5
    :param min_n_bins: 最小分箱数，默认为2
    :param min_bin_size: 每箱最小样本数或占比，默认为0.01
    :param max_bin_size: 每箱最大样本数或占比，默认为None
    :param min_bad_rate: 每箱最小坏样本率，默认为0.0
    :param monotonic: 是否要求坏样本率单调，默认为False
    :param special_codes: 特殊值列表，默认为None
    :param missing_separate: 是否将缺失值单独分为一箱，默认为True
    :param random_state: 随机种子，默认为None
    :param verbose: 是否输出详细信息，默认为False

    **示例**

    >>> from hscredit.core.binning import GeneticBinning
    >>> binner = GeneticBinning(objective='iv', max_n_bins=5, generations=100)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    """

    def __init__(
        self,
        target: str = 'target',
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_rate: float = 0.1,
        objective: str = 'iv',
        max_n_bins: int = 5,
        min_n_bins: int = 2,
        min_bin_size: Union[float, int] = 0.01,
        max_bin_size: Optional[Union[float, int]] = None,
        min_bad_rate: float = 0.0,
        monotonic: Union[bool, str] = False,
        special_codes: Optional[List] = None,
        missing_separate: bool = True,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
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
            random_state=random_state,
            verbose=verbose,
        )
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.objective = objective

        if objective not in ['iv', 'ks', 'gini']:
            raise ValueError("objective必须是'iv', 'ks'或'gini'")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'GeneticBinning':
        """拟合遗传算法分箱."""
        X, y = self._check_input(X, y)

        for feature in X.columns:
            if self.verbose:
                print(f"处理特征: {feature}")

            feature_type = self._detect_feature_type(X[feature])
            self.feature_types_[feature] = feature_type

            if feature_type == 'categorical':
                splits = self._fit_categorical(X[feature], y)
                self.splits_[feature] = splits
            else:
                splits = self._fit_numerical(X[feature], y)
                self.splits_[feature] = self._round_splits(splits)
            self.n_bins_[feature] = len(splits) + 1

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

        # 生成候选切分点（分位数）
        candidate_splits = self._generate_candidates(x_valid)
        
        if len(candidate_splits) < self.min_n_bins - 1:
            return np.array([])

        # 运行遗传算法
        best_splits = self._genetic_algorithm(
            x_valid, y_valid, candidate_splits
        )

        return best_splits

    def _generate_candidates(
        self,
        x: pd.Series
    ) -> np.ndarray:
        """生成候选切分点."""
        n_candidates = min(50, len(x) // 10)
        if n_candidates < self.max_n_bins - 1:
            n_candidates = self.max_n_bins * 2

        quantiles = np.linspace(0, 1, n_candidates + 2)[1:-1]
        candidates = np.percentile(x, quantiles * 100)
        return np.unique(candidates)

    def _genetic_algorithm(
        self,
        x: pd.Series,
        y: pd.Series,
        candidates: np.ndarray
    ) -> np.ndarray:
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
            fitness_scores = [
                self._evaluate_fitness(individual, x, y, candidates)
                for individual in population
            ]

            # 更新最优解
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if self.verbose and generation % 10 == 0:
                print(f"  Generation {generation}: Best fitness = {best_fitness:.4f}")

            # 提前停止
            if no_improvement_count >= max_no_improvement:
                if self.verbose:
                    print(f"  Early stopping at generation {generation}")
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
            return np.array([])

        # 解码最优个体
        selected_indices = np.where(best_individual)[0]
        return candidates[selected_indices]

    def _initialize_population(
        self,
        candidates: np.ndarray,
        rng: np.random.RandomState
    ) -> List[np.ndarray]:
        """初始化种群."""
        population = []
        n_candidates = len(candidates)
        
        for _ in range(self.population_size):
            # 随机选择切分点数量
            n_splits = rng.randint(self.min_n_bins - 1, min(self.max_n_bins, n_candidates) + 1)
            # 随机选择切分点
            individual = np.zeros(n_candidates, dtype=bool)
            selected = rng.choice(n_candidates, n_splits, replace=False)
            individual[selected] = True
            population.append(individual)
        
        return population

    def _evaluate_fitness(
        self,
        individual: np.ndarray,
        x: pd.Series,
        y: pd.Series,
        candidates: np.ndarray
    ) -> float:
        """评估适应度."""
        selected_indices = np.where(individual)[0]
        
        if len(selected_indices) < self.min_n_bins - 1 or len(selected_indices) > self.max_n_bins - 1:
            return -np.inf

        splits = candidates[selected_indices]
        
        # 检查约束
        if not self._check_constraints(x, y, splits):
            return -np.inf

        # 计算目标函数
        bins = pd.cut(x, bins=[-np.inf] + splits.tolist() + [np.inf], labels=False)
        
        try:
            if self.objective == 'iv':
                return self._calculate_iv(bins, y)
            elif self.objective == 'ks':
                return self._calculate_ks(bins, y)
            elif self.objective == 'gini':
                return self._calculate_gini(bins, y)
        except:
            return -np.inf

    def _check_constraints(
        self,
        x: pd.Series,
        y: pd.Series,
        splits: np.ndarray
    ) -> bool:
        """检查约束条件."""
        if len(splits) == 0:
            return False

        bins = pd.cut(x, bins=[-np.inf] + splits.tolist() + [np.inf], labels=False)
        
        # 检查最小样本数
        min_samples = self._get_min_samples(len(x))
        bin_counts = pd.Series(bins).value_counts()
        if any(count < min_samples for count in bin_counts):
            return False

        # 检查单调性
        if self.monotonic:
            bin_stats = pd.DataFrame({'bin': bins, 'target': y}).groupby('bin')['target'].mean()
            bad_rates = bin_stats.values
            if self.monotonic == 'ascending':
                if not all(bad_rates[i] <= bad_rates[i+1] + 1e-10 for i in range(len(bad_rates)-1)):
                    return False
            elif self.monotonic == 'descending':
                if not all(bad_rates[i] >= bad_rates[i+1] - 1e-10 for i in range(len(bad_rates)-1)):
                    return False

        return True

    def _calculate_iv(self, bins: pd.Series, y: pd.Series) -> float:
        """计算IV值."""
        temp_df = pd.DataFrame({'bin': bins, 'target': y})
        bin_stats = temp_df.groupby('bin')['target'].agg(['sum', 'count'])
        bin_stats.columns = ['bad', 'total']
        bin_stats['good'] = bin_stats['total'] - bin_stats['bad']

        total_bad = bin_stats['bad'].sum()
        total_good = bin_stats['good'].sum()

        if total_bad == 0 or total_good == 0:
            return 0

        # 计算分布
        bin_stats['bad_rate'] = bin_stats['bad'] / total_bad
        bin_stats['good_rate'] = bin_stats['good'] / total_good

        # 平滑处理：将0替换为epsilon，避免log(0)和除零错误
        epsilon = 1e-10
        bad_rate_smooth = bin_stats['bad_rate'].replace(0, epsilon)
        good_rate_smooth = bin_stats['good_rate'].replace(0, epsilon)
        
        # IV公式：(bad_rate - good_rate) * log(bad_rate / good_rate)
        # 该公式理论上总是非负的
        iv = ((bad_rate_smooth - good_rate_smooth) * 
              np.log(bad_rate_smooth / good_rate_smooth)).sum()
        
        return iv

    def _calculate_ks(self, bins: pd.Series, y: pd.Series) -> float:
        """计算KS值."""
        temp_df = pd.DataFrame({'bin': bins, 'target': y}).sort_values('bin')
        cumsum_bad = temp_df[temp_df['target'] == 1].groupby('bin').size().cumsum()
        cumsum_good = temp_df[temp_df['target'] == 0].groupby('bin').size().cumsum()
        
        total_bad = (y == 1).sum()
        total_good = (y == 0).sum()
        
        if total_bad == 0 or total_good == 0:
            return 0

        cumsum_bad_rate = cumsum_bad / total_bad
        cumsum_good_rate = cumsum_good / total_good
        
        ks = abs(cumsum_bad_rate - cumsum_good_rate).max()
        return ks

    def _calculate_gini(self, bins: pd.Series, y: pd.Series) -> float:
        """计算Gini系数."""
        temp_df = pd.DataFrame({'bin': bins, 'target': y})
        bin_bad_rate = temp_df.groupby('bin')['target'].mean()
        bin_count = temp_df.groupby('bin').size()
        
        # 按坏样本率排序
        sorted_indices = bin_bad_rate.argsort()
        cumsum_count = bin_count.iloc[sorted_indices].cumsum()
        cumsum_bad = (bin_bad_rate * bin_count).iloc[sorted_indices].cumsum()
        
        total = len(y)
        total_bad = y.sum()
        
        if total == 0 or total_bad == 0:
            return 0

        # Lorentz曲线面积
        area = np.trapz(cumsum_count / total, cumsum_bad / total_bad)
        gini = 2 * (0.5 - area)
        return gini

    def _selection(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
        rng: np.random.RandomState
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

    def _crossover(
        self,
        population: List[np.ndarray],
        rng: np.random.RandomState
    ) -> List[np.ndarray]:
        """交叉操作."""
        offspring = []
        
        for i in range(0, len(population), 2):
            parent1 = population[i]
            if i + 1 < len(population):
                parent2 = population[i + 1]
            else:
                offspring.append(parent1.copy())
                continue

            if rng.random() < self.crossover_rate:
                # 单点交叉
                crossover_point = rng.randint(1, len(parent1))
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])

        return offspring[:len(population)]

    def _mutation(
        self,
        population: List[np.ndarray],
        candidates: np.ndarray,
        rng: np.random.RandomState
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
        self,
        population: List[np.ndarray],
        best_individual: np.ndarray,
        fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """精英保留."""
        n_elites = int(self.elitism_rate * self.population_size)
        
        # 选择最优个体
        elite_indices = np.argsort(fitness_scores)[-n_elites:]
        elites = [population[i].copy() for i in elite_indices]
        
        # 替换最差个体
        new_population = population[:-n_elites] if n_elites > 0 else population
        new_population = elites + new_population[len(elites):]
        
        return new_population

    def _fit_categorical(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> List:
        """对类别型特征进行分箱."""
        cat_stats = pd.DataFrame({
            'category': x,
            'target': y
        }).groupby('category')['target'].agg(['mean', 'count'])

        min_samples = self._get_min_samples(len(x))
        cat_stats = cat_stats[cat_stats['count'] >= min_samples]
        cat_stats = cat_stats.sort_values('mean')

        # 限制分箱数
        if len(cat_stats) > self.max_n_bins:
            # 合并坏样本率相近的类别
            categories = self._merge_categories(cat_stats)
        else:
            categories = cat_stats.index.tolist()

        return categories

    def _merge_categories(
        self,
        cat_stats: pd.DataFrame
    ) -> List:
        """合并类别以满足分箱数限制."""
        categories = cat_stats.index.tolist()
        
        while len(categories) > self.max_n_bins:
            bad_rates = cat_stats['mean'].values
            min_diff = float('inf')
            merge_idx = 0
            
            for i in range(len(bad_rates) - 1):
                diff = abs(bad_rates[i] - bad_rates[i+1])
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
            merged_count = cat_stats.iloc[merge_idx]['count'] + cat_stats.iloc[merge_idx + 1]['count']
            merged_bad = (cat_stats.iloc[merge_idx]['mean'] * cat_stats.iloc[merge_idx]['count'] +
                         cat_stats.iloc[merge_idx + 1]['mean'] * cat_stats.iloc[merge_idx + 1]['count'])
            merged_rate = merged_bad / merged_count
            
            cat_stats = cat_stats.drop([cat1, cat2])
            cat_stats.loc[merged_cat] = {'mean': merged_rate, 'count': merged_count}
            cat_stats = cat_stats.sort_values('mean')

        return categories

    def _get_min_samples(self, n_total: int) -> int:
        """获取最小样本数."""
        if self.min_bin_size < 1:
            return int(n_total * self.min_bin_size)
        return int(self.min_bin_size)

    def _apply_bins(
        self,
        x: pd.Series,
        splits: Union[np.ndarray, List]
    ) -> np.ndarray:
        """应用分箱."""
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
        :param kwargs: 其他参数(保留兼容性)
        :return: 转换后的数据, 格式与输入X相同
        
        :example:
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
            raise ValueError("分箱器尚未拟合")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        result = pd.DataFrame(index=X.index)

        for feature in X.columns:
            if feature not in self.splits_:
                raise KeyError(f"特征 '{feature}' 未在训练数据中找到")

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
                    woe_map[-1] = 0
                    woe_map[-2] = 0
                else:
                    raise ValueError(f"特征 '{feature}' 没有WOE映射信息")
                result[feature] = pd.Series(bins).map(woe_map).values
            else:
                raise ValueError(f"未知的metric: {metric}")

        return result
