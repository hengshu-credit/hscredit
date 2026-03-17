"""指标计算模块.

提供丰富的模型评估指标计算功能。

核心特性:
- 分类指标: KS, AUC, Gini, Accuracy, Precision, Recall, F1
- 稳定性指标: PSI, CSI
- 特征重要性: IV, Gini, Entropy
- 回归指标: MSE, MAE, R2
- 分桶统计: KS_bucket, PSI_table
- 分箱指标: WOE, IV_by_bin, KS_by_bin, Chi2_by_bin

所有指标计算支持:
- 单特征计算
- 批量计算
- 详细统计表输出
- 向量化高效计算
"""

from .classification import (
    KS,
    AUC,
    Gini,
    KS_bucket,
    ROC_curve,
    confusion_matrix,
    classification_report
)

from .stability import (
    PSI,
    CSI,
    PSI_table,
    CSI_table
)

from .importance import (
    IV,
    IV_table,
    gini_importance,
    entropy_importance
)

from .regression import (
    MSE,
    MAE,
    RMSE,
    R2
)

from .binning_metrics import (
    woe_iv_vectorized,
    compute_bin_stats,
    ks_by_bin,
    chi2_by_bin,
    divergence_by_bin,
    iv_for_splits,
    ks_for_splits,
    batch_iv,
    compare_splits_iv,
    compare_splits_ks,
)

__all__ = [
    # 分类指标
    "KS",
    "AUC",
    "Gini",
    "KS_bucket",
    "ROC_curve",
    "confusion_matrix",
    "classification_report",

    # 稳定性指标
    "PSI",
    "CSI",
    "PSI_table",
    "CSI_table",

    # 特征重要性
    "IV",
    "IV_table",
    "gini_importance",
    "entropy_importance",

    # 回归指标
    "MSE",
    "MAE",
    "RMSE",
    "R2",

    # 分箱指标
    "woe_iv_vectorized",
    "compute_bin_stats",
    "ks_by_bin",
    "chi2_by_bin",
    "divergence_by_bin",
    "iv_for_splits",
    "ks_for_splits",
    "batch_iv",
    "compare_splits_iv",
    "compare_splits_ks",
]
