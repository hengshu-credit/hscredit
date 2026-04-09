"""测试 ScoreCard.scorecard_points 的分箱标签修复.

验证 scorecard_points 的变量分箱列显示的是分箱切分点（区间标签），
而不是 WOE 的阈值切分值。包含缺失值分箱的测试。
"""

import unittest
import re
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

try:
    from hscredit.core.binning import OptimalBinning
    from hscredit.core.models import ScoreCard
    from sklearn.linear_model import LogisticRegression
    HSCREDIT_AVAILABLE = True
except ImportError:
    HSCREDIT_AVAILABLE = False


@unittest.skipUnless(HSCREDIT_AVAILABLE, "hscredit not available")
class TestScoreCardPointsBinLabels(unittest.TestCase):
    """测试评分卡 scorecard_points 的分箱标签."""
    
    def setUp(self):
        """设置测试数据（含缺失值）."""
        np.random.seed(42)
        n_samples = 500
        
        self.X = pd.DataFrame({
            'age': np.random.randint(18, 65, n_samples).astype(float),
            'income': np.random.randint(3000, 20000, n_samples).astype(float),
        })
        
        # 添加缺失值
        self.X.loc[0:10, 'age'] = np.nan
        self.X.loc[5:15, 'income'] = np.nan
        
        y_prob = 1 / (1 + np.exp(
            -(self.X['age'].fillna(35) - 35) / 10
            + (self.X['income'].fillna(10000) - 10000) / 5000
        ))
        self.y = (np.random.random(n_samples) < y_prob).astype(int)
        
        # 分箱
        self.binner = OptimalBinning(method='best_iv', max_n_bins=5)
        self.binner.fit(self.X, self.y)
        self.X_woe = self.binner.transform(self.X, metric='woe')
        
        # 训练 LR (使用 sklearn 的 LR 避免 multi_class 问题)
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(self.X_woe, self.y)
    
    def _is_interval_label(self, label):
        """检查标签是否为区间格式或缺失值/特殊值标签."""
        if label in ('-', '缺失值', '特殊值'):
            return True
        # 匹配区间格式: (-inf, x], (x, y], (x, +inf]
        return bool(re.match(
            r'\((-inf|[\d.e+-]+),\s*([\d.e+-]+|\+inf)\]',
            str(label)
        ))
    
    def _is_woe_value(self, label):
        """检查标签是否为 WOE 数值（不应出现在变量分箱列）."""
        try:
            val = float(label)
            # WOE 值通常是小数
            return True
        except (ValueError, TypeError):
            return False
    
    def test_with_binner_shows_bin_intervals(self):
        """测试有分箱器时，变量分箱列显示区间格式."""
        scorecard = ScoreCard(
            binner=self.binner, lr_model=self.lr,
            pdo=60, rate=2, base_odds=35, base_score=750
        )
        scorecard.fit(self.X_woe, self.y)
        points = scorecard.scorecard_points()
        
        # 检查变量分箱列
        for _, row in points.iterrows():
            label = row['变量分箱']
            self.assertTrue(
                self._is_interval_label(label),
                f"变量分箱 '{label}' 不是区间格式（变量: {row['变量名称']}）"
            )
    
    def test_with_binner_has_missing_bin(self):
        """测试有分箱器时，包含缺失值分箱."""
        scorecard = ScoreCard(
            binner=self.binner, lr_model=self.lr,
            pdo=60, rate=2, base_odds=35, base_score=750
        )
        scorecard.fit(self.X_woe, self.y)
        points = scorecard.scorecard_points()
        
        # 检查每个特征是否包含缺失值分箱
        for col in ['age', 'income']:
            feature_points = points[points['变量名称'] == col]
            missing_bins = feature_points[
                feature_points['变量分箱'] == '缺失值'
            ]
            self.assertGreater(
                len(missing_bins), 0,
                f"特征 {col} 应有缺失值分箱"
            )
    
    def test_without_binner_no_raw_woe_values(self):
        """测试没有分箱器时，变量分箱列不显示原始 WOE 值."""
        scorecard = ScoreCard(
            lr_model=self.lr,
            pdo=60, rate=2, base_odds=35, base_score=750
        )
        scorecard.fit(self.X_woe, self.y)
        points = scorecard.scorecard_points()
        
        # 检查非基础分的行，变量分箱不应为 WOE 数值
        feature_points = points[points['变量名称'] != '基础分']
        for _, row in feature_points.iterrows():
            label = row['变量分箱']
            woe = row['WOE值']
            # 变量分箱不应该等于 WOE 值的字符串表示
            if woe is not None:
                self.assertNotEqual(
                    str(label).strip(), str(woe).strip(),
                    f"变量分箱不应显示 WOE 值 '{woe}'（变量: {row['变量名称']}）"
                )
    
    def test_splits_fallback_generates_intervals(self):
        """测试当 bin_labels 不可用但 splits 可用时，生成区间标签."""
        import copy
        binner_no_labels = copy.deepcopy(self.binner)
        # 移除分箱标签列，只保留 splits
        for feat in binner_no_labels.bin_tables_:
            if '分箱标签' in binner_no_labels.bin_tables_[feat].columns:
                binner_no_labels.bin_tables_[feat] = (
                    binner_no_labels.bin_tables_[feat].drop(columns=['分箱标签'])
                )
        
        scorecard = ScoreCard(
            binner=binner_no_labels, lr_model=self.lr,
            pdo=60, rate=2, base_odds=35, base_score=750
        )
        scorecard.fit(self.X_woe, self.y)
        points = scorecard.scorecard_points()
        
        # 检查变量分箱列应为区间格式
        for _, row in points.iterrows():
            label = row['变量分箱']
            self.assertTrue(
                self._is_interval_label(label),
                f"变量分箱 '{label}' 不是区间格式（变量: {row['变量名称']}）"
            )
    
    def test_scorecard_points_columns(self):
        """测试 scorecard_points 输出列正确."""
        scorecard = ScoreCard(
            binner=self.binner, lr_model=self.lr,
            pdo=60, rate=2, base_odds=35, base_score=750
        )
        scorecard.fit(self.X_woe, self.y)
        points = scorecard.scorecard_points()
        
        expected_columns = ['变量名称', '变量含义', '变量分箱', '对应分数', 'WOE值']
        self.assertListEqual(list(points.columns), expected_columns)
    
    def test_scorecard_points_has_base_score(self):
        """测试 scorecard_points 包含基础分."""
        scorecard = ScoreCard(
            binner=self.binner, lr_model=self.lr,
            pdo=60, rate=2, base_odds=35, base_score=750
        )
        scorecard.fit(self.X_woe, self.y)
        points = scorecard.scorecard_points()
        
        base_rows = points[points['变量名称'] == '基础分']
        self.assertEqual(len(base_rows), 1)
        self.assertEqual(base_rows.iloc[0]['变量分箱'], '-')
    
    def test_scorecard_points_with_feature_map(self):
        """测试 scorecard_points 带特征字典."""
        scorecard = ScoreCard(
            binner=self.binner, lr_model=self.lr,
            pdo=60, rate=2, base_odds=35, base_score=750
        )
        scorecard.fit(self.X_woe, self.y)
        
        feature_map = {'age': '年龄', 'income': '收入'}
        points = scorecard.scorecard_points(feature_map=feature_map)
        
        age_rows = points[points['变量名称'] == 'age']
        self.assertTrue(all(age_rows['变量含义'] == '年龄'))


if __name__ == '__main__':
    unittest.main()
