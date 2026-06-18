# 打包、编译与发布指南

本文说明 hscredit 的本地安装、依赖分组、构建校验和发布流程。

## 元数据来源

项目采用 PEP 517 / PEP 621 打包方式：

| 文件 | 作用 |
|------|------|
| `pyproject.toml` | 唯一的项目元数据、运行依赖、可选依赖和构建后端配置来源 |
| `setup.py` | 兼容旧版安装工具的最小入口，不维护依赖与版本信息 |
| `MANIFEST.in` | 源码包文件清单，控制 README、LICENSE、模板文件、docs/examples 的收录与排除 |
| `requirements.txt` | 本地全量开发环境辅助文件，不作为发布包依赖来源 |

## 安装方式

```bash
# 最小运行环境
pip install .

# 可编辑开发环境
pip install -e ".[dev]"

# 全功能环境
pip install -e ".[all]"
```

## 可选依赖分组

| 分组 | 说明 |
|------|------|
| `boost` | XGBoost / LightGBM / CatBoost / NGBoost |
| `xgboost` | 仅安装 XGBoost |
| `lightgbm` | 仅安装 LightGBM |
| `catboost` | 仅安装 CatBoost |
| `net` | Torch / TabNet |
| `pmml` | PMML 导出相关依赖 |
| `tune` | Optuna 调参与面板 |
| `explain` | SHAP 解释 |
| `dev` | pytest / black / flake8 / mypy / build / twine / check-manifest |
| `docs` | Sphinx / nbsphinx 文档构建 |
| `all` | 全部分组集合 |

## 构建发布包

```bash
# 清理旧产物
rm -rf dist/ build/

# 构建 sdist 和 wheel
python -m build

# 校验包元数据和 README 渲染
python -m twine check dist/*
```

Windows PowerShell 可使用：

```powershell
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
```

## 源码包清单校验

```bash
check-manifest
```

若新增包内模板、静态资源或示例文件，需要同步更新 `MANIFEST.in` 或 `pyproject.toml` 的 `tool.setuptools.package-data`。

## 发布流程

```bash
# TestPyPI
python -m twine upload --repository testpypi dist/*

# PyPI
python -m twine upload dist/*
```

发布前建议执行：

```bash
python -m build
python -m twine check dist/*
pytest tests/ -m "not slow and not integration"
```

## 注意事项

- 不要在 `setup.py` 中重复维护依赖、版本、作者或 classifiers。
- 最小运行依赖只放入 `project.dependencies`。
- 大型或平台敏感依赖应放入 `project.optional-dependencies`。
- `requirements.txt` 面向开发者本地全量环境，不能作为发布包依赖的权威来源。
- 包内必需资源应通过 `tool.setuptools.package-data` 和 `MANIFEST.in` 同时覆盖。
