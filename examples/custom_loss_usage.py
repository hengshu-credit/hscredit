"""
自定义损失函数使用示例

演示如何在XGBoost、LightGBM、CatBoost、TabNet中使用自定义损失函数和评估指标。
"""

import sys
from pathlib import Path

# 添加项目路径到sys.path（使用绝对路径）
project_root = Path("/Users/xiaoxi/CodeBuddy/hscredit/hscredit")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 导入hscredit损失函数
from hscredit.core.models import (
    # 损失函数
    FocalLoss,
    WeightedBCELoss,
    CostSensitiveLoss,
    BadDebtLoss,
    ApprovalRateLoss,
    ProfitMaxLoss,
    # 评估指标
    KSMetric,
    GiniMetric,
    PSIMetric,
    # 适配器
    XGBoostLossAdapter,
    LightGBMLossAdapter,
    CatBoostLossAdapter,
    TabNetLossAdapter,
)


def prepare_data():
    """准备示例数据"""
    # 创建不平衡数据集（正样本占比10%）
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def example_xgboost():
    """XGBoost使用示例"""
    print("\n" + "="*60)
    print("XGBoost 自定义损失函数示例")
    print("="*60)
    
    X_train, X_test, y_train, y_test = prepare_data()
    
    import xgboost as xgb
    
    # 示例1: Focal Loss - 处理不平衡数据
    print("\n1. Focal Loss 示例:")
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    focal_adapter = XGBoostLossAdapter(focal_loss)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'seed': 42
    }
    
    bst_focal = xgb.train(
        params,
        dtrain,
        obj=focal_adapter.objective(),
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        verbose_eval=20
    )
    
    # 预测
    y_pred_focal = bst_focal.predict(dtest)
    ks_metric = KSMetric()
    ks_value = ks_metric(y_test.values, y_pred_focal)
    print(f"Focal Loss KS: {ks_value:.4f}")
    
    # 示例2: 成本敏感损失
    print("\n2. 成本敏感损失示例:")
    # 假设漏抓坏客户损失100元，误拒好客户损失10元
    cost_loss = CostSensitiveLoss(fn_cost=100, fp_cost=10)
    cost_adapter = XGBoostLossAdapter(cost_loss)
    
    bst_cost = xgb.train(
        params,
        dtrain,
        obj=cost_adapter.objective(),
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        verbose_eval=20
    )
    
    y_pred_cost = bst_cost.predict(dtest)
    ks_value = ks_metric(y_test.values, y_pred_cost)
    print(f"成本敏感损失 KS: {ks_value:.4f}")
    
    # 示例3: 使用自定义评估指标
    print("\n3. 自定义评估指标示例:")
    gini_metric = GiniMetric()
    
    bst_with_metrics = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        custom_metric=focal_adapter.metric(ks_metric),
        verbose_eval=20
    )


def example_lightgbm():
    """LightGBM使用示例"""
    print("\n" + "="*60)
    print("LightGBM 自定义损失函数示例")
    print("="*60)
    
    X_train, X_test, y_train, y_test = prepare_data()
    
    import lightgbm as lgb
    
    # 示例1: 加权BCE损失
    print("\n1. 加权BCE损失示例:")
    weighted_loss = WeightedBCELoss(auto_balance=True)
    weighted_adapter = LightGBMLossAdapter(weighted_loss)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'seed': 42,
        'verbose': -1
    }
    
    bst_weighted = lgb.train(
        params,
        train_data,
        fobj=weighted_adapter.objective(),
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[lgb.log_evaluation(20)]
    )
    
    y_pred = bst_weighted.predict(X_test)
    ks_metric = KSMetric()
    ks_value = ks_metric(y_test.values, y_pred)
    print(f"加权BCE损失 KS: {ks_value:.4f}")
    
    # 示例2: 坏账率优化损失
    print("\n2. 坏账率优化损失示例:")
    bad_debt_loss = BadDebtLoss(
        target_approval_rate=0.3,
        bad_debt_weight=1.0,
        approval_weight=0.5
    )
    bad_debt_adapter = LightGBMLossAdapter(bad_debt_loss)
    
    bst_bad_debt = lgb.train(
        params,
        train_data,
        fobj=bad_debt_adapter.objective(),
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[lgb.log_evaluation(20)]
    )
    
    y_pred = bst_bad_debt.predict(X_test)
    ks_value = ks_metric(y_test.values, y_pred)
    print(f"坏账率优化损失 KS: {ks_value:.4f}")
    
    # 示例3: 同时使用自定义损失和自定义指标
    print("\n3. 自定义损失 + 自定义指标示例:")
    
    bst_custom = lgb.train(
        params,
        train_data,
        fobj=weighted_adapter.objective(),
        feval=weighted_adapter.metric(ks_metric),
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[lgb.log_evaluation(20)]
    )


def example_catboost():
    """CatBoost使用示例"""
    print("\n" + "="*60)
    print("CatBoost 自定义损失函数示例")
    print("="*60)
    
    X_train, X_test, y_train, y_test = prepare_data()
    
    from catboost import CatBoostClassifier, Pool
    
    # 示例1: Focal Loss
    print("\n1. Focal Loss 示例:")
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    focal_adapter = CatBoostLossAdapter(focal_loss)
    
    model_focal = CatBoostClassifier(
        iterations=100,
        loss_function=focal_adapter.objective(),
        eval_metric='AUC',
        max_depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=20
    )
    
    model_focal.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=20
    )
    
    y_pred = model_focal.predict_proba(X_test)[:, 1]
    ks_metric = KSMetric()
    ks_value = ks_metric(y_test.values, y_pred)
    print(f"Focal Loss KS: {ks_value:.4f}")
    
    # 示例2: 利润最大化损失
    print("\n2. 利润最大化损失示例:")
    profit_loss = ProfitMaxLoss(
        interest_income=100,  # 每笔贷款利息收益100元
        bad_debt_loss=1000    # 每笔坏账损失1000元
    )
    profit_adapter = CatBoostLossAdapter(profit_loss)
    
    model_profit = CatBoostClassifier(
        iterations=100,
        loss_function=profit_adapter.objective(),
        eval_metric='AUC',
        max_depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=20
    )
    
    model_profit.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=20
    )
    
    y_pred = model_profit.predict_proba(X_test)[:, 1]
    ks_value = ks_metric(y_test.values, y_pred)
    print(f"利润最大化损失 KS: {ks_value:.4f}")


def example_tabnet():
    """TabNet使用示例（需要PyTorch）"""
    print("\n" + "="*60)
    print("TabNet 自定义损失函数示例")
    print("="*60)
    
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        print("TabNet未安装，跳过示例。安装方法: pip install pytorch-tabnet")
        return
    
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Focal Loss示例
    print("\n1. Focal Loss 示例:")
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    tabnet_adapter = TabNetLossAdapter(focal_loss)
    
    model = TabNetClassifier(
        n_d=8, n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2, n_shared=2,
        seed=42,
        verbose=10
    )
    
    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_test.values, y_test.values)],
        eval_name=['test'],
        eval_metric=['auc'],
        loss_fn=tabnet_adapter.loss_fn(),
        max_epochs=10,
        patience=5,
        batch_size=1024,
        virtual_batch_size=128
    )
    
    y_pred = model.predict_proba(X_test.values)[:, 1]
    ks_metric = KSMetric()
    ks_value = ks_metric(y_test.values, y_pred)
    print(f"TabNet + Focal Loss KS: {ks_value:.4f}")


def example_psi_monitoring():
    """PSI监控示例"""
    print("\n" + "="*60)
    print("PSI监控示例")
    print("="*60)
    
    X_train, X_test, y_train, y_test = prepare_data()
    
    import xgboost as xgb
    
    # 训练模型
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 42}
    bst = xgb.train(params, dtrain, num_boost_round=100)
    
    # 获取训练集和测试集的预测分数
    train_scores = bst.predict(dtrain)
    test_scores = bst.predict(xgb.DMatrix(X_test))
    
    # 计算PSI
    psi_metric = PSIMetric(expected=train_scores, n_bins=10)
    psi_value = psi_metric(y_test, test_scores)
    
    print(f"\n训练集样本数: {len(train_scores)}")
    print(f"测试集样本数: {len(test_scores)}")
    print(f"PSI值: {psi_value:.4f}")
    
    # PSI解读
    if psi_value < 0.1:
        print("✓ PSI < 0.1: 分布稳定")
    elif psi_value < 0.25:
        print("⚠ 0.1 ≤ PSI < 0.25: 分布有轻微变化")
    else:
        print("✗ PSI ≥ 0.25: 分布变化显著，需要关注")


def example_comparison():
    """不同损失函数对比"""
    print("\n" + "="*60)
    print("不同损失函数对比")
    print("="*60)
    
    X_train, X_test, y_train, y_test = prepare_data()
    
    import xgboost as xgb
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.1,
        'seed': 42
    }
    
    # 定义不同的损失函数
    losses = {
        'Standard BCE': None,  # 使用标准损失
        'Focal Loss': FocalLoss(alpha=0.75, gamma=2.0),
        'Weighted BCE': WeightedBCELoss(auto_balance=True),
        'Cost Sensitive': CostSensitiveLoss(fn_cost=100, fp_cost=10),
    }
    
    results = []
    ks_metric = KSMetric()
    
    for name, loss in losses.items():
        if loss is None:
            # 标准BCE
            bst = xgb.train(params, dtrain, num_boost_round=100)
        else:
            # 自定义损失
            adapter = XGBoostLossAdapter(loss)
            bst = xgb.train(params, dtrain, obj=adapter.objective(), num_boost_round=100)
        
        # 预测
        y_pred = bst.predict(dtest)
        
        # 计算指标
        ks_value = ks_metric(y_test.values, y_pred)
        
        results.append({
            'Loss Function': name,
            'KS': ks_value
        })
        
        print(f"{name:20s} KS: {ks_value:.4f}")
    
    # 结果对比
    results_df = pd.DataFrame(results)
    print("\n结果对比:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    # 运行所有示例
    print("\n" + "="*60)
    print("hscredit 自定义损失函数完整示例")
    print("="*60)
    
    # 示例对比
    example_comparison()
    
    # XGBoost示例
    example_xgboost()
    
    # LightGBM示例
    example_lightgbm()
    
    # CatBoost示例
    example_catboost()
    
    # TabNet示例
    example_tabnet()
    
    # PSI监控示例
    example_psi_monitoring()
    
    print("\n" + "="*60)
    print("示例运行完成!")
    print("="*60)
