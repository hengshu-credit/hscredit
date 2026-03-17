# Sphinx文档生成完整指南

## 📚 概述

hscredit使用Sphinx生成专业的HTML文档，支持自动API文档、Markdown格式、多种部署方式。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Users/xiaoxi/CodeBuddy/hscredit/hscredit/docs/sphinx
pip install -r requirements.txt
```

### 2. 生成文档

```bash
# 方式1: 使用Makefile
make html

# 方式2: 使用Python脚本
python build_docs.py html

# 方式3: 使用测试脚本
./test_build.sh
```

### 3. 预览文档

```bash
# 方式1: 直接打开
open _build/html/index.html

# 方式2: 实时预览（需要sphinx-autobuild）
make livehtml
```

访问 http://localhost:8000 查看实时更新的文档。

---

## 📁 文档结构

```
docs/sphinx/
├── conf.py                 # Sphinx配置文件
├── index.rst               # 文档首页
├── Makefile                # Make构建命令
├── build_docs.py           # Python构建脚本
├── test_build.sh           # 测试脚本
├── requirements.txt        # Python依赖
├── README.md               # 说明文档
├── DEPLOYMENT_GUIDE.md     # 部署指南
│
├── _static/                # 静态资源
│   ├── custom.css          # 自定义样式
│   └── custom.js           # 自定义脚本
│
├── api/                    # API参考
│   ├── report.rst          # 报告模块API
│   ├── model.rst           # 模型模块API
│   └── core.rst            # 核心模块API
│
├── user_guide/             # 用户指南
│   ├── introduction.md
│   ├── excel_writer.md
│   └── losses.md
│
├── examples/               # 示例文档
│   └── index.rst
│
├── development/            # 开发文档
│   ├── contributing.md
│   └── migration.md
│
└── _build/                 # 构建输出
    ├── html/               # HTML文档
    ├── latex/              # LaTeX文档
    └── linkcheck/          # 链接检查结果
```

---

## 🔧 配置说明

### conf.py 主要配置

```python
# 项目信息
project = 'hscredit'
copyright = '2024, hscredit Team'
author = 'hscredit Team'
release = '0.1.0'

# 扩展
extensions = [
    'sphinx.ext.autodoc',      # 自动API文档
    'sphinx.ext.napoleon',     # 支持Google/NumPy docstring
    'sphinx.ext.viewcode',     # 源代码链接
    'myst_parser',             # Markdown支持
]

# 主题
html_theme = 'pydata_sphinx_theme'

# Markdown支持
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
```

---

## 📝 编写文档

### 1. API文档（自动生成）

使用autodoc从Python代码自动生成：

```rst
.. autoclass:: hscredit.report.excel.ExcelWriter
   :members:
   :undoc-members:
   :show-inheritance:
```

### 2. 编写Markdown文档

使用MyST Markdown语法：

```markdown
# 标题

## 代码示例

```python
from hscredit import ExcelWriter
writer = ExcelWriter()
```

## 警告框

```{note}
这是一个提示。
```

```{warning}
这是一个警告。
```

## 交叉引用

- {doc}`链接到其他文档 <quickstart>`
- {meth}`链接到API <hscredit.ExcelWriter.save>`
```

### 3. Python Docstring风格

使用Google风格：

```python
def function(arg1: str, arg2: int) -> bool:
    """
    函数说明
    
    Args:
        arg1: 参数1说明
        arg2: 参数2说明
    
    Returns:
        返回值说明
    
    Raises:
        ValueError: 异常说明
    
    Examples:
        >>> result = function("test", 123)
        True
    """
    return True
```

---

## 🌐 部署方式

### 方式1: GitHub Pages（推荐）

#### 自动部署

推送到GitHub后自动部署：

```bash
git add .
git commit -m "docs: 更新文档"
git push
```

GitHub Actions会自动构建和部署。

#### 手动部署

```bash
# 构建并部署
make deploy
# 或
python build_docs.py deploy
```

访问地址: `https://yourusername.github.io/hscredit/`

### 方式2: Read the Docs

1. 访问 https://readthedocs.org/
2. 导入GitHub仓库
3. 配置构建设置
4. 自动构建和部署

### 方式3: Netlify

1. 访问 https://www.netlify.com/
2. 连接GitHub仓库
3. 配置构建命令: `cd docs/sphinx && make html`
4. 发布目录: `docs/sphinx/_build/html`

### 方式4: 自托管

```bash
# 1. 构建文档
make html

# 2. 上传到服务器
rsync -avz _build/html/ user@server:/var/www/docs/

# 3. 配置Nginx
# 见 DEPLOYMENT_GUIDE.md
```

---

## 🛠️ 常用命令

### Makefile命令

```bash
make html        # 生成HTML文档
make clean       # 清理构建文件
make livehtml    # 实时预览（需要sphinx-autobuild）
make pdf         # 生成PDF文档（需要LaTeX）
make linkcheck   # 检查外部链接
make deploy      # 部署到GitHub Pages
make coverage    # 查看文档覆盖率
```

### Python脚本命令

```bash
python build_docs.py install   # 安装依赖
python build_docs.py html      # 生成HTML
python build_docs.py live      # 实时预览
python build_docs.py pdf       # 生成PDF
python build_docs.py deploy    # 部署
python build_docs.py check     # 检查链接
```

### 测试脚本

```bash
./test_build.sh    # 快速测试文档构建
```

---

## ✨ 主要特性

### 1. 自动API文档

- ✅ 从Python docstring自动生成
- ✅ 支持Google和NumPy风格
- ✅ 自动生成类、方法、函数文档
- ✅ 包含示例代码

### 2. Markdown支持

- ✅ 使用MyST-Parser
- ✅ 支持所有Markdown语法
- ✅ 支持代码高亮
- ✅ 支持数学公式

### 3. 现代化主题

- ✅ pydata-sphinx-theme
- ✅ 响应式设计
- ✅ 暗色/亮色模式
- ✅ 搜索功能

### 4. 扩展功能

- ✅ 代码复制按钮
- ✅ 外部链接标记
- ✅ 返回顶部按钮
- ✅ 目录高亮

### 5. 自动部署

- ✅ GitHub Actions集成
- ✅ 推送后自动构建
- ✅ 自动部署到GitHub Pages

---

## 🔄 文档更新流程

### 1. 修改代码

更新Python代码和docstring：

```python
def new_function():
    """
    新功能说明
    
    Args:
        param: 参数说明
    
    Returns:
        返回值说明
    
    Examples:
        >>> new_function()
        'result'
    """
    return 'result'
```

### 2. 更新文档

修改对应的 `.rst` 或 `.md` 文件。

### 3. 本地预览

```bash
make livehtml
```

在浏览器中查看效果。

### 4. 提交更改

```bash
git add docs/sphinx/
git commit -m "docs: 更新API文档"
git push
```

### 5. 自动部署

GitHub Actions自动构建和部署新文档。

---

## 🐛 故障排除

### 问题1: Sphinx未安装

```bash
# 安装依赖
pip install -r docs/sphinx/requirements.txt
```

### 问题2: 构建失败

```bash
# 清理缓存
make clean

# 查看详细日志
python -m sphinx -b html . _build/html -v
```

### 问题3: API文档为空

检查 `conf.py` 中的路径配置：

```python
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

### 问题4: 样式不生效

```bash
# 清理浏览器缓存
# Chrome/Edge: Ctrl+Shift+R (Windows) / Cmd+Shift+R (macOS)

# 重新构建
make clean && make html
```

### 问题5: 部署后404

检查GitHub Pages设置：
1. Settings → Pages
2. Source: gh-pages branch
3. Custom domain: （可选）

---

## 📖 参考资源

### 官方文档

- [Sphinx文档](https://www.sphinx-doc.org/)
- [MyST-Parser文档](https://myst-parser.readthedocs.io/)
- [pydata-sphinx-theme文档](https://pydata-sphinx-theme.readthedocs.io/)

### 部署平台

- [GitHub Pages文档](https://docs.github.com/pages)
- [Read the Docs文档](https://docs.readthedocs.io/)
- [Netlify文档](https://docs.netlify.com/)

### 扩展文档

- [sphinx-autodoc文档](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [sphinx-napoleon文档](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

---

## 📋 检查清单

### 文档质量

- [ ] 所有API都有文档
- [ ] 示例代码可运行
- [ ] 链接都有效
- [ ] 没有拼写错误
- [ ] 格式统一

### 构建检查

- [ ] HTML构建成功
- [ ] 没有警告或错误
- [ ] 样式正确显示
- [ ] 搜索功能正常

### 部署检查

- [ ] 文档可访问
- [ ] 所有页面正常
- [ ] 移动端显示正常
- [ ] 加载速度可接受

---

## 💡 最佳实践

1. **及时更新文档** - 代码变更时同步更新文档
2. **使用版本控制** - 文档与代码一起提交
3. **编写完整示例** - 示例代码要完整可运行
4. **保持简洁** - 避免冗长的说明
5. **使用图表** - 复杂流程用图表展示
6. **定期检查链接** - 使用 `make linkcheck`

---

**更新时间**: 2024-01
**文档版本**: 1.0
