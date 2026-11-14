# 安装

`hscredit` 当前源码支持 **Python 3.9–3.15**。基础安装包含常用风控分析和 SHAP 解释；
Boosting、深度学习、调参、PMML、数据库驱动与风控平台按需安装。
[PyPI 0.1.6](https://pypi.org/project/hscredit/0.1.6/) 支持同一 Python 版本范围。

## 基础安装

Python 3.9–3.15 可直接从 PyPI 安装：

```bash
python -m pip install hscredit
```

基础依赖包含 `numpy`、`pandas`、`scipy`、`scikit-learn`、`statsmodels`、`matplotlib`、
`seaborn`、`openpyxl`、`shap`、`pyarrow` 等，Python 3.15 之前还包含 `ortools`，覆盖数据探索、分箱、编码、
特征筛选、评分卡建模、指标计算、可视化与 Excel 报告。懒加载减少导入时开销，不代表减少安装体积。

## 可选能力

以下命令适用于 PyPI 安装。从源码安装可在仓库根目录执行 `python -m pip install ".[扩展名]"`。Python 3.15 的 `boost` 和 `pmml` 兼容依赖准备见下文。

| 安装命令 | 适用场景 |
|:---|:---|
| `pip install hscredit[boost]` | XGBoost、LightGBM、CatBoost、NGBoost 等 Boosting 模型 |
| `pip install hscredit[net]` | ResNet、FTTransformer、TabM、TabPFN 全部 net 模型依赖及既有 TabNet 依赖 |
| `pip install hscredit[tune]` | Optuna 参数调优和调参看板 |
| `pip install "hscredit[web]"` | 用 `python -m hscredit web` 启动[风控平台](platform.md) |
| `pip install hscredit` | 已内置 SHAP 模型解释、特征重要性和单样本分析 |
| `pip install hscredit[pmml]` | PMML 导出相关能力 |
| `pip install hscredit[db-mysql]` | MySQL / MariaDB 连接池与流式读写 |
| `pip install hscredit[db-hive]` | HiveServer2 连接池与流式读写 |
| `pip install hscredit[db-impala]` | Impala / Kudu 连接池与流式读写 |
| `pip install hscredit[db-oracle]` | Oracle 原生连接池与流式读写 |
| `pip install hscredit[db-starrocks]` | StarRocks MySQL 协议与 Stream Load |
| `pip install hscredit[db-clickhouse]` | ClickHouse 原生 DataFrame 流式读写 |
| `pip install hscredit[db-maxcompute]` | MaxCompute DB-API、原生写入与元数据 |
| `pip install hscredit[database-all]` | 安装全部数据库适配器 |
| `pip install hscredit[docs]` | 构建本文档所需的 Sphinx 工具链 |
| `pip install hscredit[all]` | 安装全部可选能力 |

```{note}
PMML 导出（`hscredit[pmml]`）依赖 `sklearn2pmml`，运行时需要 **Java 11+**。
若本机 Java 版本过低，调用 `ScoreCard.export_pmml(...)` 会抛出
`UnsupportedClassVersionError`，这属于运行环境问题而非库本身缺陷。
```

```{note}
数据库驱动全部为可选依赖。基础安装仍可安全导入 `Database`、`QueryStream` 和适配器
注册接口；只有创建对应数据库连接时才检查驱动。完整指南见 {doc}`database`。
```

<a id="python-315"></a>

## Python 3.15

### 安装当前源码

在已激活的 Python 3.15 虚拟环境中，进入包含 3.15 适配的当前源码目录执行：

```bash
python -m pip install --upgrade pip
python -m pip install .
```

Python 3.15 使用 pandas 3.0.6 及以上、SHAP 0.54 系列和标准库 `compression.zstd`。历史 Zstandard
文件仍可读取，`save_pickle`、`load_pickle` 的调用方式相同。

`or_tools`、`cp_sat` 分箱及约束修复优先使用 OR-Tools。环境中缺少 OR-Tools 时，
在相同候选边界上执行动态规划或有时限枚举，继续检查箱数、样本量、坏率、单调性及固定切点。
`solver_info_` 中的“后端”和“求解状态”反映实际使用的方法。达到时间限制时返回已经验证的
可行分区并发出提示，尚未找到可行分区时抛出 `TimeoutError`。

NumExpr、Bottleneck、LZ4 等原生扩展在缺少对应 wheel 时需要从源码编译。
Windows 需安装 Visual Studio C++ 构建工具及 Windows SDK，CatBoost 构建还需要 C++ Clang tools for Windows。
也可设置 `LLVM_ROOT`，指向包含 `bin` 和 `lib` 的独立 LLVM 根目录。Linux 需提供 C/C++ 编译器和
Python 开发头文件。PMML 的 JPype 源码构建需要 JDK 11+ 和 Apache Ant，并将 Ant 的 `bin` 目录加入 `PATH`。

### 使用 Windows x64 兼容包

[0.1.4 Release Assets](https://github.com/hengshu-credit/hscredit/releases/tag/v0.1.4) 提供以下已验证的依赖 wheel。
它们是 hscredit 基于上游源码构建的兼容包；该 Release 原有的 hscredit 主包仍要求 Python `<3.15`。

| 依赖包 | 下载 | 构建范围 |
|:---|:---|:---|
| CatBoost 1.2.10 | [wheel](https://github.com/hengshu-credit/hscredit/releases/download/v0.1.4/catboost-1.2.10-cp315-cp315-win_amd64.whl) | CPython 3.15、Windows x64；CPU 训练与预测，不含 Jupyter 交互式训练组件 |
| NGBoost 0.6.0 | [wheel](https://github.com/hengshu-credit/hscredit/releases/download/v0.1.4/ngboost-0.6.0-py3-none-any.whl) | 纯 Python 包，放宽 Python 版本约束以支持 3.15 |
| lifelines 0.30.3 | [wheel](https://github.com/hengshu-credit/hscredit/releases/download/v0.1.4/lifelines-0.30.3-py3-none-any.whl) | NGBoost 间接依赖，适配 Python 3.15 下的 pandas 3 |
| scikit-learn 1.8.0 | [wheel](https://github.com/hengshu-credit/hscredit/releases/download/v0.1.4/scikit_learn-1.8.0-cp315-cp315-win_amd64.whl) | CPython 3.15、Windows x64；兼容 PMML 所需的 `<1.9` 版本范围 |

将四个文件下载至源码目录下的 `native-wheels/`，在上述虚拟环境中安装 Boosting 能力。
先安装兼容 wheel，再由源码安装补齐其余依赖；不要省略第二步。

```bash
python -m pip install --no-index --no-deps --force-reinstall --find-links=./native-wheels catboost==1.2.10 ngboost==0.6.0 lifelines==0.30.3 scikit-learn==1.8.0
python -m pip install ".[boost]"
python -m pip check
```

需要 PMML 时，准备好 JDK 和 Apache Ant 后执行 `python -m pip install ".[pmml]"`。
这两个 Windows 原生 wheel 不适用于 Linux、macOS、Windows ARM64 或自由线程版 CPython；其他环境需自行构建。

### 从上游源码构建依赖

CatBoost 1.2.10 的构建依赖锁定了旧版 Cython。仓库提供下列构建入口，对官方源码包执行
SHA-256 校验，再将构建用的 Cython 调整至 3.3 系列，生成用于 CPU 训练和预测的 wheel。
构建使用 `--no-widget`，Jupyter 交互式训练组件需要单独适配。
NGBoost 0.6 的 Python 版本声明上限为 3.15，构建入口将这一上限调整为 3.16，
并保留其余依赖要求和算法源码。
其间接依赖 lifelines 需要同步调整 pandas 版本约束，避免安装程序降级至在 Python 3.15 上
存在日期数组崩溃问题的 pandas 2.3。
以下命令在仓库根目录和 Python 3.15 虚拟环境中运行。

```bash
python -m pip install "setuptools>=77,<83" wheel "cython>=3.3,<3.4" "cmake>=3.24,<4" "conan>=2.4.1,<3" "numpy<3" scipy ninja meson-python
python scripts/build_python315_wheels.py --package catboost --wheel-dir native-wheels --parallel 2
python scripts/build_python315_wheels.py --package ngboost --wheel-dir native-wheels
python scripts/build_python315_wheels.py --package lifelines --wheel-dir native-wheels
python -m pip install --no-index --no-deps --force-reinstall --find-links=native-wheels catboost==1.2.10 ngboost==0.6.0 lifelines==0.30.3
python -m pip install -e ".[boost]"
```

安装 PMML 可选能力时，准备 JDK 11+ 和 Apache Ant，并将 Ant 的 `bin` 目录加入当前终端的
`PATH`，以便构建 JPype 的 Java 组件。先使用支持 Python 3.15 的构建依赖生成 scikit-learn 1.8 的 wheel。
构建入口会使用独立的短路径目录，兼容 Windows 编译器的输出路径限制。

```bash
python scripts/build_python315_wheels.py --package scikit-learn --wheel-dir native-wheels --parallel 2
python -m pip install --no-index --no-deps --find-links=native-wheels scikit-learn==1.8.0
python -m pip install -e ".[pmml]"
```

以上兼容构建已在 Windows x64、CPython 3.15.2 上验证。构建时可通过
`--work-dir` 指定独立目录，保留源码和中间产物以便增量构建。

## 开发模式安装

参与开发或本地验证：

```bash
git clone https://github.com/hengshu-credit/hscredit.git
cd hscredit
pip install -e ".[dev]"
pytest tests/ -v
```

## 安装工具兼容

hscredit 支持 setuptools 77 至当前最新版，标准安装通过 PEP 517 隔离构建完成。setuptools 82
及以上已经移除 `pkg_resources`，但 hscredit 的源码、构建入口和运行时均不依赖该模块，因此无需
额外安装或恢复它。

若安装过程中出现 `ModuleNotFoundError: No module named 'pkg_resources'`，请根据完整堆栈确认
实际失败的第三方源码包，并优先选择支持当前 Python 的 wheel 或新版依赖。不要为此在 hscredit
环境中全局设置 `setuptools<82`；这会掩盖第三方包的构建兼容问题，也会阻止验证最新版
setuptools。

## 验证安装

```python
import hscredit

hscredit.info()          # 打印包信息与模块概览
print(hscredit.__version__)
```

## Agent Skills

hscredit 还提供可供 AI Agent 直接调用的分箱分析和完整报告 Skills。Skills 与 Python 包分别安装，
支持从仓库复制、由 Agent 从 GitHub 安装，或上传到 OpenAI 项目。

完整安装、调用示例和环境隔离说明见 {doc}`skills`。

## 构建文档

```bash
pip install hscredit[docs]
cd docs && make html
# 产物位于 docs/_build/html/index.html
```
