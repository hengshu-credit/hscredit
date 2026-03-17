# hscredit 测试套件

本目录包含 hscredit 包的所有测试代码，按功能模块分类组织。

## 测试目录结构

```
tests/
├── test_binning/            # 分箱功能测试（11个测试文件）
├── test_visualization/      # 可视化功能测试（5个测试文件）
├── test_feature_selection/  # 特征选择测试（5个测试文件）
├── test_modeling/           # 建模功能测试（4个测试文件）
├── test_encoding/           # 编码功能测试（1个测试文件）
├── test_reports/            # 报告功能测试（2个测试文件）
└── test_utils/              # 工具函数测试（8个测试文件）
```

## 测试分类说明

### test_binning/ - 分箱功能测试
测试各种分箱算法和功能：
- `test_binning.py`: 基础分箱功能测试
- `test_binning_detailed.py`: 详细分箱测试
- `test_binning_fixes.py`: 分箱bug修复验证
- `test_binning_optimization.py`: 分箱优化测试
- `test_binning_review.py`: 分箱复查测试
- `test_categorical_binning_complete.py`: 类别型分箱完整测试
- `test_categorical_binning_examples.py`: 类别型分箱示例测试
- `test_categorical_rules.py`: 类别规则测试
- `test_categorical_rules_simple.py`: 简单类别规则测试
- `test_monotonic_binning.py`: 单调分箱测试
- `test_uniform_binning.py`: 均匀分箱测试

### test_visualization/ - 可视化功能测试
测试各种可视化功能：
- `test_bin_plot_fix.py`: 分箱图修复测试
- `test_bin_plot_modes.py`: 分箱图模式测试
- `test_bin_table_display.py`: 分箱表格显示测试
- `test_plot_title_fix.py`: 图表标题修复测试
- `test_psi_plot.py`: PSI图测试

### test_feature_selection/ - 特征选择测试
测试特征选择和筛选功能：
- `test_all_selectors.py`: 所有选择器综合测试
- `test_stepwise_selector.py`: 逐步回归选择器测试
- `test_vif_fix.py`: VIF修复测试
- `test_feature_bin_stats_grey.py`: 特征分箱统计灰度测试
- `test_feature_bin_stats_update.py`: 特征分箱统计更新测试

### test_modeling/ - 建模功能测试
测试模型构建和评估功能：
- `test_logistic_regression_singular.py`: 逻辑回归奇异值测试
- `test_lr_multicollinearity_warning.py`: 逻辑回归多重共线性警告测试
- `test_losses.py`: 损失函数测试
- `test_metrics.py`: 评估指标测试

### test_encoding/ - 编码功能测试
测试特征编码功能：
- `test_gbm_encoder_missing.py`: GBM编码器缺失值处理测试

### test_reports/ - 报告功能测试
测试报告生成功能：
- `test_excel_writer.py`: Excel写入器测试
- `test_bin_stats_formatting.py`: 分箱统计格式化测试

### test_utils/ - 工具函数测试
测试各种工具函数和通用功能：
- `test_feature_type.py`: 特征类型测试
- `test_feature_type_edge_cases.py`: 特征类型边界情况测试
- `test_default_behavior.py`: 默认行为测试
- `test_final_verification.py`: 最终验证测试
- `test_iv_comprehensive.py`: IV值综合测试
- `test_iv_negative_fix.py`: IV值负值修复测试
- `test_notebook_fix.py`: Notebook修复测试
- `verify_iv_formula.py`: IV公式验证脚本

## 运行测试

### 运行所有测试
```bash
# 使用 pytest
pytest tests/

# 使用 pytest 并显示详细输出
pytest tests/ -v

# 使用 pytest 并生成覆盖率报告
pytest tests/ --cov=hscredit --cov-report=html
```

### 运行特定模块的测试
```bash
# 运行分箱功能测试
pytest tests/test_binning/

# 运行可视化功能测试
pytest tests/test_visualization/

# 运行特征选择测试
pytest tests/test_feature_selection/

# 运行建模功能测试
pytest tests/test_modeling/
```

### 运行单个测试文件
```bash
pytest tests/test_binning/test_binning.py -v
```

### 运行特定测试函数
```bash
pytest tests/test_binning/test_binning.py::test_binning_basic -v
```

## 测试规范

### 测试文件命名
- 测试文件以 `test_` 开头
- 验证脚本以 `verify_` 开头
- 每个测试文件对应一个或一类功能

### 测试函数命名
- 测试函数以 `test_` 开头
- 函数名应清晰描述测试内容
- 例如：`test_binning_basic()`, `test_iv_calculation()`

### 测试组织原则
1. **按功能分类**: 测试文件按功能模块组织在不同的子目录中
2. **独立性**: 每个测试应独立运行，不依赖其他测试的结果
3. **可重复性**: 测试结果应该可重复，不受外部环境影响
4. **清晰性**: 测试代码应清晰易懂，包含适当的注释

## 编写新测试

添加新测试时，请遵循以下步骤：

1. 确定测试所属的功能模块
2. 在相应的子目录中创建测试文件
3. 按照测试规范编写测试代码
4. 运行测试确保通过
5. 更新本 README 文档

## 测试覆盖范围

当前测试覆盖以下功能：
- ✓ 分箱算法和统计
- ✓ 特征编码
- ✓ 特征选择和筛选
- ✓ 模型构建和评估
- ✓ 可视化功能
- ✓ 报告生成
- ✓ 工具函数

## 注意事项

1. 运行测试前请确保已安装所有依赖包
2. 部分测试可能需要较长时间，建议使用 `-v` 参数查看进度
3. 如果测试失败，请检查错误信息并修复相关代码
4. 提交代码前请确保所有测试通过

## 相关文档

- [项目主文档](../README.md)
- [目录结构说明](../DIRECTORY_STRUCTURE.md)
- [示例代码](../examples/)
