"""GBMEncoder 使用示例.

展示如何使用 GBMEncoder 实现 GBDT + LR 等组合模型。
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

# 导入 hscredit 的 GBMEncoder
import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')
from hscredit.core.encoders import GBMEncoder


def generate_sample_data(n_samples=10000, n_features=20, random_state=42):
    """生成示例数据."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=random_state
    )
    
    # 添加一些类别特征
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    
    # 添加类别特征
    X_df['cat_1'] = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
    X_df['cat_2'] = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
    
    return X_df, pd.Series(y, name='target')


def example_1_xgboost_leaves():
    """示例1: XGBoost + 叶子节点特征."""
    print("=" * 60)
    print("示例1: XGBoost + 叶子节点特征")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"原始特征数: {X_train.shape[1]}")
    
    # 创建GBM编码器
    encoder = GBMEncoder(
        model_type='xgboost',
        n_estimators=30,
        max_depth=3,
        output_type='leaves',
        drop_origin=False,  # 保留原始特征
        random_state=42
    )
    
    # 拟合并转换
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)
    
    print(f"编码后特征数: {X_train_encoded.shape[1]}")
    print(f"新增树特征: {encoder.n_trees_}棵树的叶子节点")
    
    # 查看特征重要性
    importance = encoder.get_feature_importance()
    print("\nTop 10 重要特征:")
    print(importance.head(10))
    
    return encoder


def example_2_xgboost_lr_pipeline():
    """示例2: XGBoost + LR 组合 (GBDT + LR)."""
    print("\n" + "=" * 60)
    print("示例2: XGBoost + LR 组合 (GBDT + LR)")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_sample_data(n_samples=10000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 方案1: 使用Pipeline
    print("\n--- 方案1: 使用Pipeline ---")
    pipeline = Pipeline([
        ('gbm_encoder', GBMEncoder(
            model_type='xgboost',
            n_estimators=50,
            max_depth=4,
            output_type='leaves',
            drop_origin=True,  # 删除原始特征，只用叶子节点
            random_state=42
        )),
        ('lr', LogisticRegression(max_iter=1000, C=0.5))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))
    
    print(f"XGBoost+LR AUC: {auc:.4f}")
    print(f"XGBoost+LR Accuracy: {acc:.4f}")
    
    # 方案2: 手动分步
    print("\n--- 方案2: 手动分步 ---")
    encoder = GBMEncoder(
        model_type='xgboost',
        n_estimators=50,
        max_depth=4,
        output_type='onehot',  # 使用独热编码
        drop_origin=True,
        random_state=42
    )
    
    X_train_leaves = encoder.fit_transform(X_train, y_train)
    X_test_leaves = encoder.transform(X_test)
    
    print(f"叶子节点独热编码后维度: {X_train_leaves.shape[1]}")
    
    # 训练LR
    lr = LogisticRegression(max_iter=1000, C=0.5)
    lr.fit(X_train_leaves, y_train)
    
    y_pred = lr.predict_proba(X_test_leaves)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))
    
    print(f"XGBoost+LR (onehot) AUC: {auc:.4f}")
    print(f"XGBoost+LR (onehot) Accuracy: {acc:.4f}")


def example_3_lightgbm_lr():
    """示例3: LightGBM + LR."""
    print("\n" + "=" * 60)
    print("示例3: LightGBM + LR")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # LightGBM + LR
    encoder = GBMEncoder(
        model_type='lightgbm',
        n_estimators=40,
        max_depth=4,
        output_type='leaves',
        drop_origin=True,
        random_state=42
    )
    
    X_train_lgb = encoder.fit_transform(X_train, y_train)
    X_test_lgb = encoder.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_lgb, y_train)
    
    y_pred = lr.predict_proba(X_test_lgb)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"LightGBM+LR AUC: {auc:.4f}")
    
    # 对比纯LightGBM
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(n_estimators=40, max_depth=4, random_state=42)
    lgb_model.fit(X_train.drop(columns=['cat_1', 'cat_2']), y_train)
    y_pred_lgb = lgb_model.predict_proba(X_test.drop(columns=['cat_1', 'cat_2']))[:, 1]
    auc_lgb = roc_auc_score(y_test, y_pred_lgb)
    
    print(f"纯LightGBM AUC: {auc_lgb:.4f}")


def example_4_probability_features():
    """示例4: 使用预测概率作为特征."""
    print("\n" + "=" * 60)
    print("示例4: 使用预测概率作为特征")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 使用概率输出
    encoder = GBMEncoder(
        model_type='xgboost',
        n_estimators=50,
        max_depth=5,
        output_type='probability',
        drop_origin=False,  # 保留原始特征，添加概率
        random_state=42
    )
    
    X_train_prob = encoder.fit_transform(X_train, y_train)
    X_test_prob = encoder.transform(X_test)
    
    print(f"原始特征数: {X_train.shape[1]}")
    print(f"添加概率后特征数: {X_train_prob.shape[1]}")
    print(f"新增列: {encoder.feature_names_}")
    
    # 训练LR
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_prob, y_train)
    
    y_pred = lr.predict_proba(X_test_prob)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"原始特征+GBM概率 -> LR AUC: {auc:.4f}")


def example_5_embedding_output():
    """示例5: 使用embedding输出."""
    print("\n" + "=" * 60)
    print("示例5: 使用embedding输出")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 使用embedding输出
    encoder = GBMEncoder(
        model_type='xgboost',
        n_estimators=20,
        max_depth=3,
        output_type='embedding',
        drop_origin=True,
        random_state=42
    )
    
    X_train_emb = encoder.fit_transform(X_train, y_train)
    X_test_emb = encoder.transform(X_test)
    
    print(f"Embedding维度: {X_train_emb.shape[1]}")
    print(f"前5个样本的embedding:\n{X_train_emb.iloc[:5, :5]}")
    
    # 训练LR
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_emb, y_train)
    
    y_pred = lr.predict_proba(X_test_emb)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"GBM Embedding -> LR AUC: {auc:.4f}")


def example_6_catboost_with_categorical():
    """示例6: CatBoost处理类别特征."""
    print("\n" + "=" * 60)
    print("示例6: CatBoost处理类别特征")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"类别特征: cat_1, cat_2")
    
    # CatBoost自动处理类别特征
    encoder = GBMEncoder(
        model_type='catboost',
        n_estimators=30,
        max_depth=4,
        output_type='leaves',
        drop_origin=True,
        random_state=42
    )
    
    X_train_cb = encoder.fit_transform(X_train, y_train)
    X_test_cb = encoder.transform(X_test)
    
    print(f"CatBoost+LR特征数: {X_train_cb.shape[1]}")
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_cb, y_train)
    
    y_pred = lr.predict_proba(X_test_cb)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"CatBoost+LR AUC: {auc:.4f}")


def example_7_stacking_ensemble():
    """示例7: 使用多个GBM模型进行Stacking."""
    print("\n" + "=" * 60)
    print("示例7: 使用多个GBM模型进行Stacking")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 多个GBM编码器
    encoders = [
        ('xgb', GBMEncoder(model_type='xgboost', n_estimators=20, max_depth=3, 
                          output_type='probability', random_state=42)),
        ('lgb', GBMEncoder(model_type='lightgbm', n_estimators=20, max_depth=3,
                          output_type='probability', random_state=42)),
    ]
    
    # 生成stacking特征
    X_train_stack = X_train.copy()
    X_test_stack = X_test.copy()
    
    for name, encoder in encoders:
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
        
        # 重命名概率列
        X_train_enc = X_train_enc.rename(columns={'gbm_proba': f'{name}_proba'})
        X_test_enc = X_test_enc.rename(columns={'gbm_proba': f'{name}_proba'})
        
        X_train_stack[f'{name}_proba'] = X_train_enc[f'{name}_proba']
        X_test_stack[f'{name}_proba'] = X_test_enc[f'{name}_proba']
    
    print(f"Stacking特征数: {X_train_stack.shape[1]}")
    print(f"新增stacking特征: {[c for c in X_train_stack.columns if 'proba' in c]}")
    
    # 训练meta-learner
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_stack, y_train)
    
    y_pred = lr.predict_proba(X_test_stack)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"Stacking Ensemble (XGB+LGB+LR) AUC: {auc:.4f}")


def example_8_comparison_with_sklearn_gbdt():
    """示例8: 与sklearn的GradientBoosting对比."""
    print("\n" + "=" * 60)
    print("示例8: 与sklearn的GradientBoosting对比")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 移除类别特征用于sklearn
    X_train_num = X_train.drop(columns=['cat_1', 'cat_2'])
    X_test_num = X_test.drop(columns=['cat_1', 'cat_2'])
    
    # sklearn GBDT
    gbdt = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    gbdt.fit(X_train_num, y_train)
    y_pred_gbdt = gbdt.predict_proba(X_test_num)[:, 1]
    auc_gbdt = roc_auc_score(y_test, y_pred_gbdt)
    
    print(f"sklearn GBDT AUC: {auc_gbdt:.4f}")
    
    # XGBoost
    encoder = GBMEncoder(
        model_type='xgboost',
        n_estimators=50,
        max_depth=3,
        output_type='probability',
        drop_origin=True,
        random_state=42
    )
    X_train_xgb = encoder.fit_transform(X_train_num, y_train)
    X_test_xgb = encoder.transform(X_test_num)
    
    print(f"XGBoost输出维度: {X_train_xgb.shape[1]}")
    
    # XGBoost原始模型预测
    xgb_model = encoder.get_model()
    y_pred_xgb = xgb_model.predict_proba(X_test_num)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_pred_xgb)
    
    print(f"XGBoost AUC: {auc_xgb:.4f}")


if __name__ == '__main__':
    print("GBMEncoder 使用示例")
    print("=" * 60)
    
    # 运行所有示例
    try:
        example_1_xgboost_leaves()
    except Exception as e:
        print(f"示例1出错: {e}")
    
    try:
        example_2_xgboost_lr_pipeline()
    except Exception as e:
        print(f"示例2出错: {e}")
    
    try:
        example_3_lightgbm_lr()
    except Exception as e:
        print(f"示例3出错: {e}")
    
    try:
        example_4_probability_features()
    except Exception as e:
        print(f"示例4出错: {e}")
    
    try:
        example_5_embedding_output()
    except Exception as e:
        print(f"示例5出错: {e}")
    
    try:
        example_6_catboost_with_categorical()
    except Exception as e:
        print(f"示例6出错: {e}")
    
    try:
        example_7_stacking_ensemble()
    except Exception as e:
        print(f"示例7出错: {e}")
    
    try:
        example_8_comparison_with_sklearn_gbdt()
    except Exception as e:
        print(f"示例8出错: {e}")
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
