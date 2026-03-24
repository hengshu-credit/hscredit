"""模型评估报告.

提供统一的风控模型评估报告生成，包括:
- 模型性能指标(KS、AUC、Gini等)
- 特征重要性分析
- 评分分布分析
- PSI稳定性检验
- ROC/Lift曲线
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

from ..metrics.classification import KS, AUC, Gini
from ..metrics.stability import PSI


def _lift_score(y_true, y_proba, top_ratio=0.1):
    """计算Lift值（内部辅助函数）."""
    n = len(y_true)
    n_top = int(n * top_ratio)
    
    # 按概率降序排序
    sorted_indices = np.argsort(-y_proba)
    y_sorted = y_true[sorted_indices]
    
    # 计算整体坏样本率和top_ratio的坏样本率
    overall_bad_rate = y_true.mean()
    top_bad_rate = y_sorted[:n_top].mean()
    
    if overall_bad_rate == 0:
        return 1.0
    
    return top_bad_rate / overall_bad_rate


class ModelReport:
    """风控模型评估报告.

    生成全面的模型评估报告，包括性能指标、特征分析、稳定性检验等。

    **参数**

    :param model: 训练好的模型
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param X_test: 测试集特征，可选
    :param y_test: 测试集标签，可选
    :param feature_names: 特征名称列表，可选
    """

    def __init__(
        self,
        model: 'BaseRiskModel',
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # 处理特征名称
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        elif hasattr(model, 'feature_names_in_'):
            self.feature_names = model.feature_names_in_
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # 计算预测结果
        self.y_train_pred = model.predict(X_train)
        self.y_train_proba = model.predict_proba(X_train)[:, 1]

        if X_test is not None and y_test is not None:
            self.y_test_pred = model.predict(X_test)
            self.y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            self.y_test_pred = None
            self.y_test_proba = None

        # 存储结果
        self._metrics = None
        self._feature_importance = None
        self._score_distribution = None

    def get_metrics(self) -> pd.DataFrame:
        """获取性能指标对比表.

        :return: 性能指标DataFrame
        """
        if self._metrics is not None:
            return self._metrics

        metrics = {}

        # 训练集指标
        train_metrics = self.model.evaluate(self.X_train, self.y_train)
        for k, v in train_metrics.items():
            metrics[f'train_{k}'] = v

        # 测试集指标
        if self.X_test is not None and self.y_test is not None:
            test_metrics = self.model.evaluate(self.X_test, self.y_test)
            for k, v in test_metrics.items():
                metrics[f'test_{k}'] = v

        # 转换为DataFrame
        self._metrics = pd.DataFrame([metrics]).T
        self._metrics.columns = ['Value']
        self._metrics.index.name = 'Metric'

        return self._metrics

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """获取特征重要性.

        :param top_n: 返回前N个重要特征，默认全部
        :return: 特征重要性DataFrame
        """
        if self._feature_importance is not None:
            return self._feature_importance.head(top_n) if top_n else self._feature_importance

        importances = self.model.get_feature_importances()

        # 创建DataFrame
        df = pd.DataFrame({
            'Feature': importances.index,
            'Importance': importances.values,
            'Importance_Ratio': importances.values / importances.sum()
        })

        # 计算排名
        df['Rank'] = range(1, len(df) + 1)

        # 添加累积重要性
        df['Cumulative_Importance'] = df['Importance_Ratio'].cumsum()

        self._feature_importance = df

        return df.head(top_n) if top_n else df

    def get_score_distribution(self, n_bins: int = 10, dataset: str = 'train') -> pd.DataFrame:
        """获取评分分布.

        :param n_bins: 分箱数，默认10
        :param dataset: 数据集，'train'或'test'
        :return: 评分分布DataFrame
        """
        if dataset == 'train':
            y = self.y_train
            proba = self.y_train_proba
        elif dataset == 'test' and self.y_test is not None:
            y = self.y_test
            proba = self.y_test_proba
        else:
            raise ValueError(f"数据集 {dataset} 不可用")

        scores = (1 - proba) * 1000  # 转换为0-1000分

        # 等频分箱
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(scores, quantiles)
        bin_edges[-1] += 1e-6  # 确保最大值包含在内

        # 分箱统计
        results = []
        for i in range(n_bins):
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])

            n_samples = mask.sum()
            n_bad = y[mask].sum()
            n_good = n_samples - n_bad
            bad_rate = n_bad / n_samples if n_samples > 0 else 0

            results.append({
                'Bin': i + 1,
                'Score_Min': bin_edges[i],
                'Score_Max': bin_edges[i + 1],
                'Sample_Count': n_samples,
                'Good_Count': n_good,
                'Bad_Count': n_bad,
                'Bad_Rate': bad_rate,
                'Sample_Ratio': n_samples / len(y)
            })

        return pd.DataFrame(results)

    def get_psi(self, n_bins: int = 10) -> float:
        """计算PSI稳定性指标.

        :param n_bins: 分箱数，默认10
        :return: PSI值
        """
        if self.y_test_proba is None:
            raise ValueError("需要提供测试集才能计算PSI")

        train_scores = (1 - self.y_train_proba) * 1000
        test_scores = (1 - self.y_test_proba) * 1000

        return PSI(train_scores, test_scores, bins=n_bins)

    def get_confusion_matrix(self, threshold: float = 0.5, dataset: str = 'train') -> pd.DataFrame:
        """获取混淆矩阵.

        :param threshold: 分类阈值，默认0.5
        :param dataset: 数据集，'train'或'test'
        :return: 混淆矩阵DataFrame
        """
        if dataset == 'train':
            y_true = self.y_train
            y_proba = self.y_train_proba
        elif dataset == 'test' and self.y_test is not None:
            y_true = self.y_test
            y_proba = self.y_test_proba
        else:
            raise ValueError(f"数据集 {dataset} 不可用")

        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        return pd.DataFrame(
            cm,
            index=['Actual_0', 'Actual_1'],
            columns=['Pred_0', 'Pred_1']
        )

    def get_roc_curve(self, dataset: str = 'train') -> Dict[str, np.ndarray]:
        """获取ROC曲线数据.

        :param dataset: 数据集，'train'或'test'
        :return: 包含fpr, tpr, thresholds的字典
        """
        if dataset == 'train':
            y_true = self.y_train
            y_proba = self.y_train_proba
        elif dataset == 'test' and self.y_test is not None:
            y_true = self.y_test
            y_proba = self.y_test_proba
        else:
            raise ValueError(f"数据集 {dataset} 不可用")

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': AUC(y_true, y_proba)
        }

    def get_lift_curve(self, dataset: str = 'train', n_bins: int = 10) -> pd.DataFrame:
        """获取Lift曲线数据.

        :param dataset: 数据集，'train'或'test'
        :param n_bins: 分箱数，默认10
        :return: Lift曲线DataFrame
        """
        if dataset == 'train':
            y_true = self.y_train
            y_proba = self.y_train_proba
        elif dataset == 'test' and self.y_test is not None:
            y_true = self.y_test
            y_proba = self.y_test_proba
        else:
            raise ValueError(f"数据集 {dataset} 不可用")

        # 按概率降序排序
        y_true_arr = np.asarray(y_true)
        y_proba_arr = np.asarray(y_proba)
        sorted_indices = np.argsort(-y_proba_arr)
        y_sorted = y_true_arr[sorted_indices]

        # 等频分箱
        bin_size = len(y_sorted) // n_bins
        results = []

        overall_bad_rate = y_true.mean()

        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(y_sorted)

            bin_y = y_sorted[start:end]
            bin_bad_rate = bin_y.mean()
            lift = bin_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0

            results.append({
                'Bin': i + 1,
                'Population_Ratio': (i + 1) / n_bins * 100,
                'Bad_Rate': bin_bad_rate,
                'Lift': lift,
                'Cumulative_Bad_Capture': y_sorted[:end].sum() / y_true.sum() * 100
            })

        return pd.DataFrame(results)

    def print_report(self):
        """打印完整报告."""
        print("=" * 60)
        print("风控模型评估报告")
        print("=" * 60)

        # 模型信息
        info = self.model.get_model_info()
        print(f"\n【模型信息】")
        print(f"模型类型: {info['model_type']}")
        print(f"目标函数: {info['objective']}")
        print(f"特征数量: {info['n_features']}")
        print(f"类别数量: {info['n_classes']}")

        # 性能指标
        print(f"\n【性能指标】")
        metrics = self.get_metrics()
        print(metrics.to_string())

        # 特征重要性
        print(f"\n【Top 10 特征重要性】")
        importance = self.get_feature_importance(top_n=10)
        print(importance.to_string(index=False))

        # 评分分布
        print(f"\n【评分分布 (训练集)】")
        dist = self.get_score_distribution(n_bins=5, dataset='train')
        print(dist.to_string(index=False))

        # PSI
        if self.y_test is not None:
            psi = self.get_psi()
            print(f"\n【稳定性检验】")
            print(f"PSI: {psi:.4f}")
            if psi < 0.1:
                print("评价: 模型稳定 (PSI < 0.1)")
            elif psi < 0.25:
                print("评价: 模型略有变化 (0.1 <= PSI < 0.25)")
            else:
                print("评价: 模型不稳定 (PSI >= 0.25)")

        print("\n" + "=" * 60)

    def to_dict(self) -> Dict[str, Any]:
        """将报告转换为字典."""
        return {
            'model_info': self.model.get_model_info(),
            'metrics': self.get_metrics().to_dict(),
            'feature_importance': self.get_feature_importance().to_dict(),
            'score_distribution_train': self.get_score_distribution(dataset='train').to_dict(),
            'psi': self.get_psi() if self.y_test is not None else None
        }
