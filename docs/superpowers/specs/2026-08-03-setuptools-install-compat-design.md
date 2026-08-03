# hscredit 多版本 setuptools 安装兼容设计

## 背景

setuptools 82.0.0 已移除 `pkg_resources`。hscredit 源码与兼容入口 `setup.py` 当前均未引用
`pkg_resources`，并且已经在 Python 3.14、setuptools 83.0.0、`pkg_resources` 不存在的环境中通过
wheel 构建和 pip/PEP 517 隔离构建。

当前构建配置仍存在三项需要收敛的问题：

1. `[build-system]` 包含 hscredit 构建不需要的 `cython`、`wheel` 和 `build`。
2. `hscredit._compat` 在运行时直接使用 `packaging.version`，但 `packaging` 仅由 `boost` extra
   间接声明，基础安装无法保证其存在。
3. `project.license = {text = "MIT"}` 在新版 setuptools 中已弃用，继续使用会产生构建警告，
   且存在未来被移除的风险。

## 目标

- hscredit 的源码包和 wheel 构建不依赖 `pkg_resources`。
- 支持 setuptools 77 至当前最新版；覆盖仍提供 `pkg_resources` 的版本以及已经移除它的版本。
- Python 3.9 使用该 Python 版本可安装的最新 setuptools，Python 3.10 及以上使用当前最新版。
- 默认 PEP 517 构建隔离与显式 `--no-build-isolation` 构建均可工作。
- `packaging` 作为 hscredit 的直接运行依赖声明，不依赖 setuptools、wheel 或其他库的传递安装。
- 不通过 `try/except` 判断 setuptools 或 `pkg_resources` 版本分支。
- 不为规避第三方旧包问题而全局限制 `setuptools<82`。

## 非目标

- 不保证任意第三方历史源码包都能在 setuptools 82+ 下构建。第三方项目的 PEP 517 构建环境由
  其自身的 `pyproject.toml` 决定，hscredit 无法从自身元数据覆盖。
- 不在导入 hscredit 时安装、升级或降级 setuptools。
- 不把 setuptools、wheel、build 作为 hscredit 运行依赖。
- 本次不修改 Pandas 及其关联库的最终版本策略；该部分沿用独立的依赖兼容设计。

## 构建配置

`pyproject.toml` 的构建后端调整为：

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"
```

选择 setuptools 77 作为构建下界，是因为项目同时采用 PEP 639 的 SPDX license 字符串：

```toml
[project]
license = "MIT"
```

setuptools 77 至 81 代表仍包含 `pkg_resources` 的构建路径，setuptools 82 及以上代表不再包含
`pkg_resources` 的构建路径。hscredit 对两条路径使用同一份 `pyproject.toml`，不维护条件分支。

从 `[build-system].requires` 移除：

- `cython`：仓库没有 `.pyx`、`.pxd` 或需要本地编译的扩展源码。
- `wheel`：现代 setuptools 可直接执行 wheel 构建。
- `build`：它是构建前端，应由开发或发布环境安装，而不是安装进 PEP 517 后端隔离环境。

保留 `setup.py` 作为旧安装工具的兼容入口。该文件只调用 `setuptools.setup()`，不读取项目元数据、
不导入 hscredit，也不访问 `pkg_resources`。

## 运行依赖

在 `[project].dependencies` 和开发用 `requirements.txt` 中直接加入不指定版本下限的
`packaging`：

```toml
dependencies = [
    "packaging",
]
```

`hscredit._compat` 继续通过 `packaging.version.Version` 执行明确版本号比较。setuptools 是否包含
`pkg_resources` 不参与运行时判断。

## 第三方依赖处理

若完整安装在某个第三方包的构建阶段出现
`ModuleNotFoundError: No module named 'pkg_resources'`，处理顺序固定为：

1. 根据完整堆栈确认失败包，不能将错误归因于 hscredit 的构建后端。
2. 优先选择该包支持目标 Python 的 wheel 或已迁移到现代构建接口的版本。
3. 非基础功能依赖保留在对应 extra 中，避免阻断 hscredit 基础安装。
4. 无法替换的历史源码包只能由安装方使用单独的 build constraint 临时处理；不得把
   `setuptools<82` 写入 hscredit 的全局构建或运行依赖。

## 验证矩阵

自动化构建测试至少覆盖：

| Python | setuptools | `pkg_resources` 状态 | 构建方式 |
| --- | --- | --- | --- |
| 3.9 | 77.x | 存在 | `python -m build --no-isolation` |
| 3.9 | 82.0.1 | 不存在 | `pip wheel --use-pep517 --no-deps` |
| 3.14 | 当前最新版 | 不存在 | `pip wheel --use-pep517 --no-deps` |

仓库测试负责验证静态元数据：

- `[build-system].requires` 只包含 `setuptools>=77`。
- 项目 license 使用 SPDX 字符串 `MIT`。
- 核心依赖包含无版本限制的 `packaging`。
- 核心依赖不包含 `setuptools`、`wheel`、`build` 或 `pkg_resources`。
- 源码、`setup.py` 和构建配置不导入 `pkg_resources`。
- 构建出的 wheel 元数据包含 `Requires-Dist: packaging`，不包含运行时 setuptools 依赖。

## 成功标准

1. setuptools 77.x 和 Python 3.9 可构建 hscredit wheel。
2. setuptools 82.0.1 且 `pkg_resources` 不存在时可构建 hscredit wheel。
3. Python 3.14 与当前最新版 setuptools 可通过 PEP 517 隔离构建。
4. 构建过程不再出现 `project.license` 表格式弃用警告。
5. 安装后的 hscredit 不依赖 `pkg_resources`，版本兼容逻辑继续使用 `packaging.version`。
6. 现有测试通过，且不修改工作区中与 EDA 并行功能相关的未提交内容。
