"""GBMEncoder 缺失值处理测试."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')
from hscredit.core.encoders import GBMEncoder


def generate_data_with_missing(n_samples=1000, n_features=10, missing_ratio=0.1, random_state=42):
    """生成带缺失值的示例数据."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=random_state
    )
    
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    X_df['cat_1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    
    # 引入缺失值
    np.random.seed(random_state)
    for col in X_df.columns:
        missing_mask = np.random.random(n_samples) < missing_ratio
        X_df.loc[missing_mask, col] = np.nan
    
    return X_df, pd.Series(y, name='target')


class TestGBMEncoderMissing:
    """测试GBMEncoder缺失值处理."""
    
    def test_xgboost_with_missing(self):
        """测试XGBoost处理缺失值."""
        X, y = generate_data_with_missing(n_samples=500, missing_ratio=0.15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 确认有缺失值
        assert X_train.isnull().sum().sum() > 0, "测试数据应该包含缺失值"
        
        # 创建编码器
        encoder = GBMEncoder(
            model_type='xgboost',
            n_estimators=10,
            max_depth=3,
            output_type='probability',
            random_state=42
        )
        
        # 拟合和转换（应该不报错）
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
        
        # 验证输出
        assert X_train_enc.shape[0] == X_train.shape[0]
        assert X_test_enc.shape[0] == X_test.shape[0]
        
        # 验证可以训练LR
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_enc, y_train)
        y_pred = lr.predict_proba(X_test_enc)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        assert auc > 0.5, "AUC应该大于0.5"
        print(f"✓ XGBoost with missing values: AUC={auc:.4f}")
    
    def test_lightgbm_with_missing(self):
        """测试LightGBM处理缺失值."""
        try:
            import lightgbm
        except ImportError:
            pytest.skip("LightGBM未安装")
        
        X, y = generate_data_with_missing(n_samples=500, missing_ratio=0.15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        encoder = GBMEncoder(
            model_type='lightgbm',
            n_estimators=10,
            max_depth=3,
            output_type='probability',
            random_state=42
        )
        
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_enc, y_train)
        y_pred = lr.predict_proba(X_test_enc)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        assert auc > 0.5
        print(f"✓ LightGBM with missing values: AUC={auc:.4f}")
    
    def test_catboost_with_missing(self):
        """测试CatBoost处理缺失值."""
        try:
            import catboost
        except ImportError:
            pytest.skip("CatBoost未安装")
        
        X, y = generate_data_with_missing(n_samples=500, missing_ratio=0.15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        encoder = GBMEncoder(
            model_type='catboost',
            n_estimators=10,
            max_depth=3,
            output_type='probability',
            random_state=42
        )
        
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_enc, y_train)
        y_pred = lr.predict_proba(X_test_enc)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        assert auc > 0.5
        print(f"✓ CatBoost with missing values: AUC={auc:.4f}")
    
    def test_missing_stats(self):
        """测试缺失值统计功能."""
        X, y = generate_data_with_missing(n_samples=500, missing_ratio=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        encoder = GBMEncoder(
            model_type='xgboost',
            n_estimators=5,
            max_depth=2,
            random_state=42
        )
        
        encoder.fit(X_train, y_train)
        
        # 获取缺失值统计
        missing_stats = encoder.get_missing_stats()
        
        assert isinstance(missing_stats, pd.DataFrame)
        assert 'feature' in missing_stats.columns
        assert 'missing_count' in missing_stats.columns
        assert 'missing_ratio' in missing_stats.columns
        
        if len(missing_stats) > 0:
            assert missing_stats['missing_count'].sum() > 0
        
        print(f"✓ Missing stats: {len(missing_stats)} features with missing values")
    
    def test_no_missing(self):
        """测试无缺失值时正常工作."""
        X, y = generate_data_with_missing(n_samples=500, missing_ratio=0.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        assert X_train.isnull().sum().sum() == 0, "测试数据应该无缺失值"
        
        encoder = GBMEncoder(
            model_type='xgboost',
            n_estimators=10,
            max_depth=3,
            output_type='probability',
            random_state=42
        )
        
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_enc, y_train)
        y_pred = lr.predict_proba(X_test_enc)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        assert auc > 0.5
        print(f"✓ No missing values: AUC={auc:.4f}")
    
    def test_high_missing_ratio(self):
        """测试高缺失率情况."""
        X, y = generate_data_with_missing(n_samples=500, missing_ratio=0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        encoder = GBMEncoder(
            model_type='xgboost',
            n_estimators=10,
            max_depth=3,
            output_type='probability',
            random_state=42
        )
        
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_enc, y_train)
        y_pred = lr.predict_proba(X_test_enc)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        # 即使30%缺失率，模型仍应能工作
        assert auc > 0.5
        print(f"✓ High missing ratio (30%): AUC={auc:.4f}")


if __name__ == '__main__':
    test = TestGBMEncoderMissing()
    
    print("Running GBMEncoder missing value tests...\n")
    
    try:
        test.test_xgboost_with_missing()
    except Exception as e:
        print(f"✗ XGBoost test failed: {e}")
    
    try:
        test.test_lightgbm_with_missing()
    except Exception as e:
        print(f"✗ LightGBM test failed: {e}")
    
    try:
        test.test_catboost_with_missing()
    except Exception as e:
        print(f"✗ CatBoost test failed: {e}")
    
    try:
        test.test_missing_stats()
    except Exception as e:
        print(f"✗ Missing stats test failed: {e}")
    
    try:
        test.test_no_missing()
    except Exception as e:
        print(f"✗ No missing test failed: {e}")
    
    try:
        test.test_high_missing_ratio()
    except Exception as e:
        print(f"✗ High missing ratio test failed: {e}")
    
    print("\nAll tests completed!")
