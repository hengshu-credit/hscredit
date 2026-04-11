"""
自定义损失函数模块测试

测试所有损失函数和评估指标的基本功能。
"""

import numpy as np
import pytest
from hscredit.core.models import (
    FocalLoss,
    AsymmetricFocalLoss,
    WeightedBCELoss,
    CostSensitiveLoss,
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
    OrdinalRankLoss,
    LiftFocusedLoss,
    KSMetric,
    GiniMetric,
    PSIMetric,
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
)


class TestFocalLoss:
    """测试Focal Loss"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        loss = FocalLoss(alpha=0.75, gamma=2.0)
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        
        # 计算损失
        loss_value = loss(y_true, y_pred)
        assert isinstance(loss_value, float)
        assert loss_value >= 0
        
        # 计算梯度
        grad = loss.gradient(y_true, y_pred)
        assert grad.shape == y_true.shape
        
        # 计算二阶导
        hess = loss.hessian(y_true, y_pred)
        assert hess.shape == y_true.shape
        assert np.all(hess > 0)  # 二阶导应该为正
    
    def test_perfect_prediction(self):
        """测试完美预测"""
        loss = FocalLoss()
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.01, 0.02, 0.98, 0.99])  # 接近完美预测
        
        loss_value = loss(y_true, y_pred)
        assert loss_value < 0.1  # 损失应该很小
    
    def test_xgboost_adapter(self):
        """测试XGBoost适配器"""
        loss = FocalLoss(alpha=0.75, gamma=2.0)
        adapter = XGBoostLossAdapter(loss)
        
        # 获取目标函数
        obj_fn = adapter.objective()
        assert callable(obj_fn)


class TestAsymmetricFocalLoss:
    """测试不对称Focal Loss"""

    def test_basic_functionality(self):
        """测试基本功能"""
        loss = AsymmetricFocalLoss(alpha=0.7, gamma_pos=2.5, gamma_neg=1.0)

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])

        loss_value = loss(y_true, y_pred)
        assert isinstance(loss_value, float)
        assert loss_value >= 0

        grad = loss.gradient(y_true, y_pred)
        hess = loss.hessian(y_true, y_pred)
        assert grad.shape == y_true.shape
        assert hess.shape == y_true.shape
        assert np.all(hess > 0)

    def test_asymmetric_focus_changes_penalty(self):
        """测试正负样本聚焦参数差异会影响梯度"""
        y_true = np.array([0, 1])
        y_pred = np.array([0.3, 0.7])

        loss_low = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=1.0)
        loss_high = AsymmetricFocalLoss(gamma_pos=4.0, gamma_neg=0.5)

        grad_low = np.abs(loss_low.gradient(y_true, y_pred))
        grad_high = np.abs(loss_high.gradient(y_true, y_pred))

        assert not np.allclose(grad_low, grad_high)

    def test_clip_value_stability(self):
        """测试clip_value在极端概率下的稳定性"""
        loss = AsymmetricFocalLoss(clip_value=0.05)

        y_true = np.array([0, 1])
        y_pred = np.array([1e-12, 1 - 1e-12])

        value = loss(y_true, y_pred)
        grad = loss.gradient(y_true, y_pred)
        hess = loss.hessian(y_true, y_pred)

        assert np.isfinite(value)
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))

    def test_adapter_compatibility(self):
        """测试与框架适配器兼容"""
        loss = AsymmetricFocalLoss()

        assert callable(XGBoostLossAdapter(loss).objective())
        assert callable(LightGBMLossAdapter(loss).objective())
        assert hasattr(CatBoostLossAdapter(loss).objective(), 'calc_ders_range')


class TestWeightedBCELoss:
    """测试加权BCE损失"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        loss = WeightedBCELoss(pos_weight=5.0, neg_weight=1.0)
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        
        loss_value = loss(y_true, y_pred)
        assert isinstance(loss_value, float)
        assert loss_value >= 0
        
        grad = loss.gradient(y_true, y_pred)
        assert grad.shape == y_true.shape
        
        hess = loss.hessian(y_true, y_pred)
        assert hess.shape == y_true.shape
    
    def test_auto_balance(self):
        """测试自动平衡权重"""
        loss = WeightedBCELoss(auto_balance=True)
        
        # 不平衡数据：90%负样本，10%正样本
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.random.rand(100)
        
        loss(y_true, y_pred)
        
        # 权重应该根据样本比例自动设置
        # 正样本权重应该约为 90/10 = 9
        assert abs(loss.pos_weight - 9.0) < 1.0


class TestCostSensitiveLoss:
    """测试成本敏感损失"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        loss = CostSensitiveLoss(fn_cost=100, fp_cost=1)
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        
        loss_value = loss(y_true, y_pred)
        assert isinstance(loss_value, float)
        
        grad = loss.gradient(y_true, y_pred)
        assert grad.shape == y_true.shape
        
        hess = loss.hessian(y_true, y_pred)
        assert hess.shape == y_true.shape
    
    def test_cost_ratio(self):
        """测试成本比例影响"""
        # 高成本比例
        loss_high = CostSensitiveLoss(fn_cost=1000, fp_cost=1)
        # 低成本比例
        loss_low = CostSensitiveLoss(fn_cost=10, fp_cost=1)
        
        y_true = np.array([1])  # 正样本
        y_pred = np.array([0.5])  # 预测概率
        
        # 高成本损失对正样本预测错误应该惩罚更重
        grad_high = loss_high.gradient(y_true, y_pred)
        grad_low = loss_low.gradient(y_true, y_pred)
        
        # 高成本的梯度绝对值应该更大
        assert abs(grad_high[0]) > abs(grad_low[0])


class TestRiskLosses:
    """测试风控业务损失"""
    
    def test_bad_debt_loss(self):
        """测试坏账率损失"""
        loss = BadDebtLoss(
            target_approval_rate=0.3,
            bad_debt_weight=1.0,
            approval_weight=0.5
        )
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        
        loss_value = loss(y_true, y_pred)
        assert isinstance(loss_value, float)
        
        grad = loss.gradient(y_true, y_pred)
        assert grad.shape == y_true.shape
    
    def test_approval_rate_loss(self):
        """测试通过率损失"""
        loss = ApprovalRateLoss(target_bad_debt_rate=0.05)
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        
        loss_value = loss(y_true, y_pred)
        assert isinstance(loss_value, float)
    
    def test_profit_max_loss(self):
        """测试利润最大化损失"""
        loss = ProfitMaxLoss(
            interest_income=100,
            bad_debt_loss=1000
        )
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        
        loss_value = loss(y_true, y_pred)
        assert isinstance(loss_value, float)
        
        grad = loss.gradient(y_true, y_pred)
        assert grad.shape == y_true.shape

    def test_ordinal_rank_loss_prefers_better_ordering(self):
        """测试排序损失更偏好正确排序"""
        loss = OrdinalRankLoss(rank_weight=2.0, bce_weight=1.0, max_pairs=1000)

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        y_pred_bad = np.array([0.8, 0.7, 0.6, 0.4, 0.3, 0.2])

        assert loss(y_true, y_pred_good) < loss(y_true, y_pred_bad)

        grad = loss.gradient(y_true, y_pred_good)
        hess = loss.hessian(y_true, y_pred_good)
        assert grad.shape == y_true.shape
        assert hess.shape == y_true.shape
        assert np.all(hess > 0)

    def test_ordinal_rank_loss_single_class_fallback(self):
        """测试单类样本下排序损失退化为稳定BCE"""
        loss = OrdinalRankLoss()

        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.6, 0.7, 0.8, 0.9])

        value = loss(y_true, y_pred)
        grad = loss.gradient(y_true, y_pred)
        hess = loss.hessian(y_true, y_pred)

        assert np.isfinite(value)
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))
        assert np.all(hess > 0)

    def test_lift_focused_loss_emphasizes_head_samples(self):
        """测试头部样本具有更高惩罚权重"""
        loss = LiftFocusedLoss(top_ratio=0.25, penalty_factor=4.0, positive_class_boost=2.0)

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.2, 0.5, 0.8, 0.9])

        weights = loss._get_sample_weights(y_true, y_pred)
        assert weights.shape == y_true.shape
        assert weights[np.argmax(y_pred)] > weights[np.argmin(y_pred)]

        value = loss(y_true, y_pred)
        grad = loss.gradient(y_true, y_pred)
        hess = loss.hessian(y_true, y_pred)

        assert np.isfinite(value)
        assert grad.shape == y_true.shape
        assert hess.shape == y_true.shape
        assert np.all(hess > 0)

    def test_lift_focused_loss_penalizes_head_bad_sample_more(self):
        """测试头部坏样本错误具有更大梯度惩罚"""
        loss = LiftFocusedLoss(top_ratio=0.5, penalty_factor=3.0, positive_class_boost=2.0)

        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.95, 0.85, 0.40, 0.10])

        grad = np.abs(loss.gradient(y_true, y_pred))
        assert grad[0] > grad[-1]


class TestMetrics:
    """测试评估指标"""
    
    def test_ks_metric(self):
        """测试KS指标"""
        metric = KSMetric()
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        
        ks_value = metric(y_true, y_pred)
        
        # KS值应该在[0, 1]范围内
        assert 0 <= ks_value <= 1
        
        # 对于这个预测，KS应该较高（因为预测较好）
        assert ks_value > 0.5
    
    def test_gini_metric(self):
        """测试Gini指标"""
        metric = GiniMetric()
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        
        gini_value = metric(y_true, y_pred)
        
        # Gini应该在[-1, 1]范围内
        assert -1 <= gini_value <= 1
        
        # 对于好的预测，Gini应该为正
        assert gini_value > 0
    
    def test_psi_metric(self):
        """测试PSI指标"""
        # 基准分布
        expected = np.random.randn(1000)
        
        metric = PSIMetric(expected=expected, n_bins=10)
        
        # 相同分布
        actual_same = expected.copy()
        psi_same = metric(np.zeros(1000), actual_same)
        assert psi_same < 0.01  # 应该接近0
        
        # 不同分布
        actual_diff = expected + 1  # 均值偏移
        psi_diff = metric(np.zeros(1000), actual_diff)
        assert psi_diff > psi_same  # 应该更大


class TestAdapters:
    """测试框架适配器"""
    
    def test_xgboost_adapter(self):
        """测试XGBoost适配器"""
        loss = FocalLoss(alpha=0.75, gamma=2.0)
        adapter = XGBoostLossAdapter(loss)
        
        # 获取目标函数
        obj_fn = adapter.objective()
        assert callable(obj_fn)
        
        # 获取评估指标
        ks_metric = KSMetric()
        metric_fn = adapter.metric(ks_metric)
        assert callable(metric_fn)
    
    def test_lightgbm_adapter(self):
        """测试LightGBM适配器"""
        loss = WeightedBCELoss(auto_balance=True)
        adapter = LightGBMLossAdapter(loss)
        
        obj_fn = adapter.objective()
        assert callable(obj_fn)
        
        ks_metric = KSMetric()
        metric_fn = adapter.metric(ks_metric)
        assert callable(metric_fn)
        
        # LightGBM metric应该返回(name, value, greater_is_better)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        result = metric_fn(y_true, y_pred)
        assert len(result) == 3
    
    def test_catboost_adapter(self):
        """测试CatBoost适配器"""
        loss = CostSensitiveLoss(fn_cost=100, fp_cost=1)
        adapter = CatBoostLossAdapter(loss)
        
        obj = adapter.objective()
        assert hasattr(obj, 'calc_ders_range')
        
        ks_metric = KSMetric()
        metric_obj = adapter.metric(ks_metric)
        assert hasattr(metric_obj, 'evaluate')
        assert hasattr(metric_obj, 'is_max_optimal')
        assert hasattr(metric_obj, 'get_final_error')


class TestNumericalStability:
    """测试数值稳定性"""
    
    def test_extreme_probabilities(self):
        """测试极端概率值"""
        loss = FocalLoss()
        
        y_true = np.array([0, 1])
        y_pred = np.array([1e-10, 1 - 1e-10])  # 极端概率
        
        # 应该不会出错
        loss_value = loss(y_true, y_pred)
        assert not np.isnan(loss_value)
        assert not np.isinf(loss_value)
        
        grad = loss.gradient(y_true, y_pred)
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))
        
        hess = loss.hessian(y_true, y_pred)
        assert not np.any(np.isnan(hess))
        assert not np.any(np.isinf(hess))
    
    def test_all_same_class(self):
        """测试所有样本属于同一类"""
        loss = WeightedBCELoss()
        
        y_true = np.array([1, 1, 1, 1])  # 全是正样本
        y_pred = np.array([0.3, 0.5, 0.7, 0.9])
        
        loss_value = loss(y_true, y_pred)
        assert not np.isnan(loss_value)
        
        grad = loss.gradient(y_true, y_pred)
        assert not np.any(np.isnan(grad))
    
    def test_empty_array(self):
        """测试空数组"""
        loss = FocalLoss()
        
        y_true = np.array([])
        y_pred = np.array([])
        
        # 空数组应返回nan或抛出异常，不崩溃即可
        try:
            result = loss(y_true, y_pred)
            # 如果返回了值，应该是nan
            assert np.isnan(result)
        except Exception:
            # 抛出异常也是可接受的行为
            pass


class TestIntegration:
    """集成测试"""
    
    def test_loss_metric_combination(self):
        """测试损失函数和评估指标组合"""
        loss = FocalLoss(alpha=0.75, gamma=2.0)
        adapter = LightGBMLossAdapter(loss)
        ks_metric = KSMetric()
        
        # 创建模拟数据
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.rand(100)
        
        # 计算损失
        loss_value = loss(y_true, y_pred)
        assert isinstance(loss_value, float)
        
        # 计算KS
        ks_value = ks_metric(y_true, y_pred)
        assert isinstance(ks_value, float)
        
        # 使用适配器
        obj_fn = adapter.objective()
        grad, hess = obj_fn(y_true, y_pred)
        assert grad.shape == y_true.shape
        assert hess.shape == y_true.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
