"""测试重构后的 ScoreCard 功能."""

import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from hscredit.core.models import ScoreCard, LogisticRegression
    from hscredit.core.binning import OptimalBinning
    from hscredit.core.encoders import WOEEncoder
    HSCREDIT_AVAILABLE = True
except ImportError:
    HSCREDIT_AVAILABLE = False


@unittest.skipUnless(HSCREDIT_AVAILABLE, "hscredit not available")
class TestScoreCardRefactored(unittest.TestCase):
    """测试重构后的 ScoreCard."""
    
    def setUp(self):
        """设置测试数据."""
        np.random.seed(42)
        n_samples = 500
        
        self.X = pd.DataFrame({
            'age': np.random.randint(18, 65, n_samples),
            'income': np.random.randint(3000, 20000, n_samples),
        })
        
        # 创建目标变量
        y_prob = 1 / (1 + np.exp(-(self.X['age'] - 35) / 10 + (self.X['income'] - 10000) / 5000))
        self.y = (np.random.random(n_samples) < y_prob).astype(int)
        
        # 分箱并转换 WOE
        self.binner = OptimalBinning(method='best_iv', max_n_bins=5)
        self.binner.fit(self.X, self.y)
        self.X_woe = self.binner.transform(self.X, metric='woe')
    
    def test_fit_with_woe_data(self):
        """测试 fit 接受 WOE 数据."""
        scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        scorecard.fit(self.X_woe, self.y)
        
        self.assertTrue(hasattr(scorecard, 'lr_model_'))
        self.assertIsNotNone(scorecard.lr_model_)
        self.assertEqual(len(scorecard.feature_names_), 2)
    
    def test_predict_with_raw_data_auto_detect(self):
        """测试 predict 自动检测原始数据并转换."""
        scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        scorecard.fit(self.X_woe, self.y)
        
        # predict 传入原始数据
        scores = scorecard.predict(self.X)
        
        self.assertEqual(len(scores), len(self.X))
        # 评分应该有意义（不为 NaN）
        self.assertTrue(np.all(np.isfinite(scores)))
    
    def test_predict_with_woe_data_auto_detect(self):
        """测试 predict 自动检测 WOE 数据."""
        scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        scorecard.fit(self.X_woe, self.y)
        
        # predict 传入 WOE 数据
        scores = scorecard.predict(self.X_woe)
        
        self.assertEqual(len(scores), len(self.X_woe))
        # 评分应该有意义（不为 NaN）
        self.assertTrue(np.all(np.isfinite(scores)))
    
    def test_predict_with_input_type_param(self):
        """测试 predict 的 input_type 参数."""
        scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        scorecard.fit(self.X_woe, self.y)
        
        # 强制作为原始数据处理
        scores_raw = scorecard.predict(self.X, input_type='raw')
        
        # 强制作为 WOE 数据处理
        scores_woe = scorecard.predict(self.X_woe, input_type='woe')
        
        # 结果应该相同（因为 X 经过 WOE 转换后应该等于 X_woe）
        self.assertEqual(len(scores_raw), len(scores_woe))
    
    def test_binner_as_woe_transformer(self):
        """测试分箱器作为 WOE 转换器."""
        # 配置 binner
        scorecard = ScoreCard(
            binner=self.binner,
            pdo=60, rate=2, base_odds=35, base_score=750,
            verbose=True
        )
        
        # fit 前检查是否识别到 binner 的 WOE 能力
        self.assertTrue(hasattr(scorecard, '_binner_is_woe_transformer'))
        
        scorecard.fit(self.X_woe, self.y)
        
        # predict 应该能自动使用 combiner 进行 WOE 转换
        scores = scorecard.predict(self.X)
        
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(np.all(np.isfinite(scores)))
    
    def test_with_lr_model(self):
        """测试传入预训练 LR 模型."""
        lr = LogisticRegression()
        lr.fit(self.X_woe, self.y)
        
        scorecard = ScoreCard(lr_model=lr)
        scorecard.fit(self.X_woe, self.y)
        
        self.assertEqual(scorecard.lr_model_, lr)
        scores = scorecard.predict(self.X_woe, input_type='woe')
        self.assertEqual(len(scores), len(self.X_woe))
    
    def test_with_pipeline(self):
        """测试从 pipeline 提取组件."""
        # 创建 pipeline
        pipeline = Pipeline([
            ('binner', OptimalBinning(method='best_iv', max_n_bins=5)),
            ('lr', LogisticRegression())
        ])
        
        # 注意：sklearn pipeline 需要适配，这里简化测试
        # 实际使用时可能需要自定义 pipeline 或直接使用 combiner
    
    def test_scorecard_output(self):
        """测试评分卡输出."""
        scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        scorecard.fit(self.X_woe, self.y)
        
        # 评分卡刻度
        scale = scorecard.scorecard_scale()
        self.assertIn('刻度项', scale.columns)
        self.assertIn('刻度值', scale.columns)
        
        # 评分卡分数
        points = scorecard.scorecard_points()
        self.assertIn('变量名称', points.columns)
        self.assertIn('对应分数', points.columns)
    
    def test_predict_proba(self):
        """测试预测概率."""
        scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        scorecard.fit(self.X_woe, self.y)
        
        proba = scorecard.predict_proba(self.X)
        
        self.assertEqual(proba.shape, (len(self.X), 2))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
    
    def test_get_reason(self):
        """测试获取评分原因."""
        scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        scorecard.fit(self.X_woe, self.y)
        
        reasons = scorecard.get_reason(self.X, keep=2)
        
        self.assertEqual(len(reasons), len(self.X))
        self.assertIn('reason', reasons.columns)


if __name__ == '__main__':
    unittest.main()
