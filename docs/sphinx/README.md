# hscredit Sphinx文档

本目录包含hscredit项目的Sphinx文档配置和源文件。

## 快速开始

### 1. 安装依赖

```bash
cd /Users/xiaoxi/CodeBuddy/hscredit/hscredit/docs/sphinx
pip install -r requirements.txt
```

### 2. 构建文档

```bash
# 生成HTML文档
make html

# 或使用Python脚本
python build_docs.py html
```

### 3. 预览文档

```bash
# 打开生成的文档
open _build/html/index.html

# 或启动实时预览
make livehtml
```

## 目录结构

```
sphinx/
├── _static/              # 静态资源（CSS、JS）
├── _templates/           # 自定义模板
├── api/                  # API参考文档
│   ├── report.rst        # 报告模块API
│   ├── model.rst         # 模型模块API
│   └── core.rst          # 核心模块API
├── examples/             # 示例文档
├── user_guide/           # 用户指南
├── development/          # 开发文档
├── conf.py               # Sphinx配置文件
├── index.rst             # 文档首页
├── Makefile              # Make构建命令
├── build_docs.py         # Python构建脚本
├── requirements.txt      # Python依赖
├── DEPLOYMENT_GUIDE.md   # 部署指南
└── README.md             # 本文件
```

## 常用命令

### Makefile命令

```bash
make html        # 生成HTML文档
make clean       # 清理构建文件
make livehtml    # 启动实时预览
make pdf         # 生成PDF文档
make linkcheck   # 检查外部链接
make deploy      # 部署到GitHub Pages
```

### Python脚本命令

```bash
python build_docs.py install   # 安装依赖
python build_docs.py html      # 生成HTML
python build_docs.py live      # 实时预览
python build_docs.py deploy    # 部署
python build_docs.py check     # 检查链接
```

## 部署文档

详细部署指南请查看 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)。

### GitHub Pages部署

```bash
# 一键部署
make deploy
```

访问地址: `https://yourusername.github.io/hscredit/`

## 编写文档

### API文档

使用 `autodoc` 自动生成：

```rst
.. autoclass:: hscredit.module.ClassName
   :members:
   :undoc-members:
   :show-inheritance:
```

### Markdown文档

使用 MyST Markdown：

```markdown
# 标题

正文内容。

```python
print("Hello, World!")
```
```

## 文档风格

- 使用Google风格或NumPy风格的docstring
- 代码示例要完整可运行
- 包含必要的注释和说明
- 使用警告框突出重要信息

## 更新文档

1. 修改源代码和docstring
2. 更新对应的 `.rst` 或 `.md` 文件
3. 本地预览确认效果
4. 提交更改并部署

## 故障排除

详见 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) 的故障排除章节。

## 参考链接

- [Sphinx文档](https://www.sphinx-doc.org/)
- [MyST-Parser](https://myst-parser.readthedocs.io/)
- [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/)
