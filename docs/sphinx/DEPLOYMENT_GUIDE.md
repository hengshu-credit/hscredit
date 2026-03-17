# 文档部署指南

本文档说明如何生成和部署hscredit的HTML文档。

## 快速开始

### 1. 安装依赖

```bash
cd /Users/xiaoxi/CodeBuddy/hscredit/hscredit/docs/sphinx
pip install -r requirements.txt
```

或使用构建脚本：

```bash
python build_docs.py install
```

### 2. 生成HTML文档

```bash
# 使用Makefile
make html

# 或使用Python脚本
python build_docs.py html
```

生成的文档位于 `_build/html/index.html`。

### 3. 本地预览

```bash
# 打开生成的HTML文件
open _build/html/index.html

# 或启动实时预览服务器
make livehtml
# 或
python build_docs.py live
```

访问 http://localhost:8000 查看文档。

---

## 部署选项

### 选项1: GitHub Pages（推荐）

#### 自动部署

```bash
# 构建并部署
make deploy
# 或
python build_docs.py deploy
```

#### 手动部署

1. 构建HTML文档：
   ```bash
   make html
   ```

2. 部署到gh-pages分支：
   ```bash
   pip install ghp-import
   ghp-import -n -p -f _build/html
   ```

3. 在GitHub仓库设置中启用GitHub Pages：
   - Settings → Pages → Source: gh-pages branch

4. 访问文档：`https://yourusername.github.io/hscredit/`

#### 使用GitHub Actions自动部署

创建 `.github/workflows/docs.yml`：

```yaml
name: Deploy Docs

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r docs/sphinx/requirements.txt
        pip install -e .
    
    - name: Build documentation
      run: |
        cd docs/sphinx
        make html
    
    - name: Deploy to GitHub Pages
      if: github.event_name == 'push'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/sphinx/_build/html
```

### 选项2: Read the Docs

1. 在 [Read the Docs](https://readthedocs.org/) 注册账号
2. 导入GitHub仓库
3. 配置构建设置：
   - Python配置文件：`docs/sphinx/conf.py`
   - Requirements文件：`docs/sphinx/requirements.txt`
4. 自动构建和部署

### 选项3: Netlify

1. 在 [Netlify](https://www.netlify.com/) 注册账号
2. 创建新站点，连接GitHub仓库
3. 配置构建设置：
   - Build command: `cd docs/sphinx && make html`
   - Publish directory: `docs/sphinx/_build/html`
4. 自动构建和部署

### 选项4: 自托管服务器

1. 构建HTML文档：
   ```bash
   make html
   ```

2. 上传 `_build/html` 目录到服务器：
   ```bash
   # 使用rsync
   rsync -avz _build/html/ user@server:/var/www/hscredit-docs/
   
   # 或使用scp
   scp -r _build/html/* user@server:/var/www/hscredit-docs/
   ```

3. 配置Nginx：
   ```nginx
   server {
       listen 80;
       server_name docs.hscredit.com;
       
       root /var/www/hscredit-docs;
       index index.html;
       
       location / {
           try_files $uri $uri/ =404;
       }
   }
   ```

---

## 文档结构

```
docs/sphinx/
├── _static/              # 静态文件
│   ├── custom.css        # 自定义样式
│   └── custom.js         # 自定义脚本
├── _templates/           # 模板文件
├── _build/               # 构建输出
│   ├── html/             # HTML文档
│   ├── latex/            # LaTeX文档
│   └── linkcheck/        # 链接检查结果
├── api/                  # API文档
│   ├── report.rst
│   ├── model.rst
│   └── core.rst
├── conf.py               # Sphinx配置
├── index.rst             # 文档首页
├── installation.md       # 安装指南
├── quickstart.md         # 快速开始
├── Makefile              # Make命令
├── requirements.txt      # 依赖列表
└── build_docs.py         # 构建脚本
```

---

## 常用命令

### 构建命令

```bash
# 生成HTML文档
make html

# 清理构建文件
make clean

# 实时预览
make livehtml

# 生成PDF
make pdf

# 检查链接
make linkcheck

# 部署到GitHub Pages
make deploy
```

### Python脚本命令

```bash
# 安装依赖
python build_docs.py install

# 生成HTML
python build_docs.py html

# 启动实时预览
python build_docs.py live

# 部署到GitHub Pages
python build_docs.py deploy

# 检查链接
python build_docs.py check
```

---

## 文档更新流程

### 1. 更新源代码

修改Python代码和docstring：

```python
def example_function(param1, param2):
    """
    示例函数
    
    Parameters
    ----------
    param1 : str
        参数1说明
    param2 : int
        参数2说明
    
    Returns
    -------
    bool
        返回值说明
    
    Examples
    --------
    >>> result = example_function("test", 123)
    >>> print(result)
    True
    """
    return True
```

### 2. 更新RST文件

修改 `api/*.rst` 文件，添加新的API文档：

```rst
.. autoclass:: hscredit.module.ClassName
   :members:
   :undoc-members:
   :show-inheritance:
```

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

如果配置了GitHub Actions，文档会自动构建和部署。

---

## 文档风格指南

### Python Docstring风格

使用Google风格或NumPy风格：

```python
# Google风格
def function(arg1, arg2):
    """
    函数说明
    
    Args:
        arg1 (str): 参数1说明
        arg2 (int): 参数2说明
    
    Returns:
        bool: 返回值说明
    
    Examples:
        >>> function("test", 123)
        True
    """
    pass

# NumPy风格
def function(arg1, arg2):
    """
    函数说明
    
    Parameters
    ----------
    arg1 : str
        参数1说明
    arg2 : int
        参数2说明
    
    Returns
    -------
    bool
        返回值说明
    
    Examples
    --------
    >>> function("test", 123)
    True
    """
    pass
```

### Markdown文档风格

使用MyST Markdown语法：

```markdown
# 标题

正文内容。

## 二级标题

### 代码块

```python
print("Hello, World!")
```

### 警告框

```{note}
这是一个提示。
```

```{warning}
这是一个警告。
```

### 链接和引用

- [外部链接](https://example.com)
- {doc}`内部文档链接 <quickstart>`
- {meth}`API链接 <hscredit.module.Class.method>`
```

---

## 故障排除

### Q1: Sphinx构建失败

**检查Python版本**：

```bash
python --version  # 需要 >= 3.8
```

**检查依赖**：

```bash
pip list | grep sphinx
```

**清理缓存**：

```bash
make clean
```

### Q2: API文档为空

**检查模块路径**：

```python
# 在conf.py中
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

**检查autodoc配置**：

```python
# 在conf.py中
extensions = ['sphinx.ext.autodoc']
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
}
```

### Q3: 样式不生效

**检查静态文件路径**：

```python
# 在conf.py中
html_static_path = ['_static']
```

**清理浏览器缓存**：

Ctrl+Shift+R (Windows/Linux) 或 Cmd+Shift+R (macOS)

### Q4: 部署后404错误

**检查index.html**：

确保 `_build/html/index.html` 存在。

**检查GitHub Pages设置**：

Settings → Pages → Source: gh-pages branch

---

## 参考资源

- [Sphinx官方文档](https://www.sphinx-doc.org/)
- [MyST-Parser文档](https://myst-parser.readthedocs.io/)
- [pydata-sphinx-theme文档](https://pydata-sphinx-theme.readthedocs.io/)
- [GitHub Pages文档](https://docs.github.com/pages)
- [Read the Docs文档](https://docs.readthedocs.io/)

---

**更新时间**: 2024-01
**文档版本**: 1.0
