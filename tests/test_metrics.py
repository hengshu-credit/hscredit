#!/usr/bin/env python3
"""
测试所有指标模块
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from hscredit.core.metrics.classification import KS, AUC, Gini
from hscredit.core.metrics.importance import IV
from hscredit.core.metrics.stability import PSI, CSI
from hscredit.core.metrics.regression import MSE, MAE, RMSE, R2

def test_metrics():
    print('=== 分类指标测试 ===')
    # 生成分类测试数据
    y_true_cls = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred_cls = np.array([0.1, 0.3, 0.6, 0.8, 0.2, 0.9, 0.4, 0.7, 0.5, 0.1])

    print('KS:', KS(y_true_cls, y_pred_cls))
    print('AUC:', AUC(y_true_cls, y_pred_cls))
    print('Gini:', Gini(y_true_cls, y_pred_cls))

    print('\n=== 稳定性指标测试 ===')
    # 生成稳定性测试数据
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(0.1, 1.1, 1000)

    print('PSI:', PSI(data1, data2))
    print('CSI:', CSI(data1, data2))

    print('\n=== 特征重要性测试 ===')
    # 生成特征重要性测试数据
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100)
    })
    y = np.random.randint(0, 2, 100)

    # IV函数计算单个特征的IV值
    iv_value = IV(y, X['feature1'])
    print('IV for feature1:', iv_value)

    print('\n=== 回归指标测试 ===')
    # 生成回归测试数据
    y_true_reg = np.array([1, 2, 3, 4, 5])
    y_pred_reg = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

    print('MSE:', MSE(y_true_reg, y_pred_reg))
    print('MAE:', MAE(y_true_reg, y_pred_reg))
    print('RMSE:', RMSE(y_true_reg, y_pred_reg))
    print('R2:', R2(y_true_reg, y_pred_reg))

    print('\n=== 所有指标模块测试通过! ===')

if __name__ == '__main__':
    test_metrics()