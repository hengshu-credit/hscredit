"""指标计算模块 - 统一的模型评估指标入口.

提供分类、回归、特征评估、稳定性、金融风控等场景的评估指标。

指标分类:
- 分类指标: ks, auc, gini, accuracy, precision, recall, f1, ks_bucket
- 特征评估: iv, iv_table, chi2_test, cramers_v, feature_importance, bin_stats
- 稳定性: psi, psi_table, csi, csi_table, batch_psi
- 金融风控: lift, lift_table, lift_curve, badrate, badrate_by_group
- 回归指标: mse, mae, rmse, r2

使用示例:
    >>> from hscredit.core import metrics
    >>> metrics.ks(y_true, y_prob)
    0.45
    >>> metrics.iv(y_true, feature)
    0.23
    >>> metrics.psi(score_train, score_test)
    0.05

命名规范:
- 所有函数使用小写+下划线命名
- 分类指标: ks, auc, gini
- 特征指标: iv, iv_table
- 稳定性: psi, csi
- 金融指标: lift, badrate
"""

# 分类指标
from .classification import (
    ks, auc, gini,
    accuracy, precision, recall, f1,
    ks_bucket, roc_curve,
    confusion_matrix, classification_report,
    ks_2samps,
)

# 特征评估
from .feature import (
    iv, iv_table,
    chi2_test, cramers_v,
    feature_importance,
    feature_summary,
)

# 稳定性
from .stability import (
    psi, psi_table, psi_rating,
    csi, csi_table,
    batch_psi,
)

# 金融风控
from .finance import (
    lift, lift_at, lift_table, lift_curve, lift_monotonicity_check, rule_lift,
    badrate, badrate_by_group, badrate_trend, badrate_by_score_bin,
    score_stats, score_stability,
)

# 回归指标
from .regression import (
    mse, mae, rmse, r2,
)

# 分箱统计（供其他模块直接使用）
from ._binning import compute_bin_stats, add_margins, quadratic_curve_coefficient, composite_binning_quality

__all__ = [
    # 分类指标
    'ks', 'auc', 'gini',
    'accuracy', 'precision', 'recall', 'f1',
    'ks_bucket', 'roc_curve',
    'confusion_matrix', 'classification_report',
    'ks_2samps',
    
    # 特征评估
    'iv', 'iv_table',
    'chi2_test', 'cramers_v',
    'feature_importance',
    'feature_summary',
    
    # 稳定性
    'psi', 'psi_table', 'psi_rating',
    'csi', 'csi_table',
    'batch_psi',
    
    # 金融风控
    'lift', 'lift_at', 'lift_table', 'lift_curve', 'lift_monotonicity_check', 'rule_lift',
    'badrate', 'badrate_by_group', 'badrate_trend', 'badrate_by_score_bin',
    'score_stats', 'score_stability',
    
    # 回归指标
    'mse', 'mae', 'rmse', 'r2',
    
    # 分箱统计
    'compute_bin_stats', 'add_margins', 'quadratic_curve_coefficient', 'composite_binning_quality',
]
