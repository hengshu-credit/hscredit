# 安装

`hscredit` 需要 **Python 3.9+**。基础安装保持轻量，重型依赖（Boosting、深度学习、
调参、解释、PMML）按需安装。

## 基础安装

```bash
pip install hscredit
```

基础依赖包含 `numpy`、`pandas`、`scipy`、`scikit-learn`、`statsmodels`、`matplotlib`、
`seaborn`、`openpyxl` 等，足以覆盖数据探索、分箱、编码、特征筛选、评分卡建模、指标计算、
可视化与 Excel 报告。

## 可选能力

| 安装命令 | 适用场景 |
|:---|:---|
| `pip install hscredit[boost]` | XGBoost、LightGBM、CatBoost、NGBoost 等 Boosting 模型 |
| `pip install hscredit[net]` | PyTorch 与 TabNet 深度学习模型 |
| `pip install hscredit[tune]` | Optuna 参数调优和调参看板 |
| `pip install hscredit[explain]` | SHAP 模型解释 |
| `pip install hscredit[pmml]` | PMML 导出相关能力 |
| `pip install hscredit[docs]` | 构建本文档所需的 Sphinx 工具链 |
| `pip install hscredit[all]` | 安装全部可选能力 |

```{note}
PMML 导出（`hscredit[pmml]`）依赖 `sklearn2pmml`，运行时需要 **Java 11+**。
若本机 Java 版本过低，调用 `ScoreCard.export_pmml(...)` 会抛出
`UnsupportedClassVersionError`，这属于运行环境问题而非库本身缺陷。
```

## 开发模式安装

参与开发或本地验证：

```bash
git clone https://github.com/hscredit/hscredit.git
cd hscredit
pip install -e ".[dev]"
pytest tests/ -v
```

## 验证安装

```python
import hscredit

hscredit.info()          # 打印包信息与模块概览
print(hscredit.__version__)
```

## 构建文档

```bash
pip install hscredit[docs]
cd docs && make html
# 产物位于 docs/_build/html/index.html
```
