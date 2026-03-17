# hscredit 项目结构设计

## 设计原则

1. **模块化**: 各功能模块独立，便于维护和扩展
2. **中文优先**: 所有用户可见的命名、文档、报告均以中文为主
3. **向后兼容**: 保持现有API不变，避免破坏性变更

## 推荐项目结构

```
hscredit/
├── core/                      # 核心算法模块
│   ├── binning/              # 分箱算法
│   │   ├── __init__.py
│   │   ├── base.py          # 基类
│   │   ├── optimal_binning.py  # 最优分箱统一接口
│   │   ├── optimal_iv_binning.py
│   │   ├── optimal_ks_binning.py
│   │   ├── quantile_binning.py
│   │   ├── tree_binning.py
│   │   └── ...              # 其他分箱方法
│   ├── metrics/             # 指标计算
│   │   ├── __init__.py
│   │   ├── binning_metrics.py  # 分箱指标
│   │   └── model_metrics.py   # 模型指标
│   └── __init__.py
│
├── preprocessing/             # 数据预处理模块
│   ├── encoders/            # 特征编码器
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── woe_encoder.py   # WOE编码
│   │   ├── target_encoder.py
│   │   ├── count_encoder.py
│   │   ├── one_hot_encoder.py
│   │   ├── ordinal_encoder.py
│   │   ├── quantile_encoder.py
│   │   └── catboost_encoder.py
│   ├── selection/           # 特征筛选
│   │   ├── __init__.py
│   │   ├── filter_method.py
│   │   ├── wrapper_method.py
│   │   └── embedded_method.py
│   └── __init__.py
│
├── feature_engineering/      # 特征工程
│   ├── __init__.py
│   ├── expression.py        # 特征表达式
│   └── transformer.py       # 特征转换
│
├── model/                   # 模型模块
│   ├── __init__.py
│   ├── losses/             # 损失函数
│   └── trainer.py          # 训练器
│
├── rules/                   # 规则集
│   ├── __init__.py
│   └── rule.py             # 规则定义
│
├── report/                  # 报告输出
│   ├── __init__.py
│   ├── excel/              # Excel报告
│   │   └── __init__.py
│   └── base.py             # 报告基类
│
├── analysis/                # 分析模块
│   ├── __init__.py
│   ├── feature_analyzer.py # 特征分析
│   └── score_analyzer.py   # 评分分析
│
├── utils/                   # 公共工具
│   ├── __init__.py
│   ├── io.py               # IO工具
│   ├── misc.py             # 杂项工具
│   ├── display.py          # 展示工具
│   └── datasets.py         # 数据集工具
│
└── __init__.py             # 包入口
```

## 模块说明

### core/ - 核心算法
- **binning/**: 提供18种分箱算法，统一通过OptimalBinning接口访问
- **metrics/**: 计算WOE、IV、KS、LIFT等风控指标

### preprocessing/ - 预处理
- **encoders/**: 7种特征编码器（WOE、Target、Count、OneHot、Ordinal、Quantile、CatBoost）
- **selection/**: 特征筛选方法（过滤法、包裹法、嵌入法）

### feature_engineering/ - 特征工程
- 特征表达式解析
- 特征转换

### model/ - 模型
- 损失函数定义
- 模型训练器

### rules/ - 规则集
- 规则定义和执行

### report/ - 报告输出
- Excel报告生成
- 报告模板

### analysis/ - 分析
- 特征分析
- 评分分析

### utils/ - 公共工具
- IO操作
- 辅助函数
- 数据展示

## 中文本地化规范

1. **类名/函数名**: 使用中文或中英结合（如OptimalBinning、WOEEncoder）
2. **文档字符串**: 中文优先，部分术语可保留英文
3. **报告输出**: 
   - 列名使用中文（如"坏样本率"、"IV值"）
   - 图表标题使用中文
   - 单位使用国际单位或常用缩写
4. **异常信息**: 中文提示
5. **日志输出**: 中文优先

## 现有模块整合

| 现有目录 | 目标位置 | 说明 |
|---------|---------|------|
| core/binning | core/binning | 保持不变 |
| core/metrics | core/metrics | 保持不变 |
| core/selection | preprocessing/selection | 移动到preprocessing下 |
| preprocessing/encoders | core/encoders | 已迁移 |
| preprocessing/selection.py | preprocessing/selection | 移动到selection目录 |
| analysis | analysis | 保持不变 |
| feature_engineering | feature_engineering | 保持不变 |
| model | model | 保持不变 |
| rules | rules | 保持不变 |
| report | report | 保持不变 |
| utils | utils | 保持不变 |

## 迁移注意事项

1. 保持现有导入路径兼容（如from hscredit.core.binning import OptimalBinning）
2. 更新__all__导出列表
3. 添加deprecation警告（可选）
