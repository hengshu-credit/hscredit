# hscredit Makefile
# 简化常用操作

.PHONY: help install dev test validate clean jupyter build build-check check-manifest publish-test publish docs push-docs clone-docs

# 默认目标
.DEFAULT_GOAL := help

# 帮助信息
help: ## 显示帮助信息
	@echo "hscredit 常用命令:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

# 安装（生产环境）
install: ## 安装生产环境依赖
	pip install -e .
	@echo "✅ 安装完成"

# 安装（开发环境）
dev: ## 安装开发环境依赖
	pip install -e ".[dev]"
	pip install jupyter notebook ipykernel
	python -m ipykernel install --user --name=hscredit --display-name="hscredit"
	@echo "✅ 开发环境安装完成"

# 运行测试
test: ## 运行单元测试
	pytest tests/ -v --tb=short
	@echo "✅ 测试完成"

# 测试覆盖率
coverage: ## 生成测试覆盖率报告
	pytest tests/ --cov=hscredit --cov-report=html --cov-report=term
	@echo "✅ 覆盖率报告已生成: htmlcov/index.html"

# 环境验证
validate: ## 验证开发环境
	python scripts/validate_environment.py

# 启动Jupyter
jupyter: ## 启动Jupyter Notebook
	cd examples && jupyter notebook

# 运行notebook验证
notebook-test: ## 执行notebook验证
	python examples/00_quickstart.py
	@echo "✅ 快速开始示例验证完成"

# 代码格式化
format: ## 格式化代码
	black hscredit tests
	@echo "✅ 代码格式化完成"

# 代码检查
lint: ## 检查代码质量
	flake8 hscredit tests --select=E9,F63,F7,F82,F601
	@echo "✅ 代码检查完成"

# 类型检查
type-check: ## 类型检查
	mypy hscredit/core/model_selection.py hscredit/utils/serialization.py hscredit/utils/parallel.py --follow-imports=skip --ignore-missing-imports
	@echo "✅ 类型检查完成"

# 清理
clean: ## 清理临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf outputs/*.xlsx
	rm -rf docs/_build/
	@echo "✅ 清理完成"

# 构建文档
docs: ## 构建文档
	cd docs && make clean && make html
	@echo "✅ 文档已生成: docs/_build/html/index.html"

# 构建发布包（包含文档构建）
build: docs ## 构建发布包（同时构建文档）
	python -m pip install --upgrade build twine
	rm -rf dist/ build/
	python -m build
	@echo "✅ 发布包已生成: dist/"

build-check: build ## 构建并校验发布包元数据
	python -m twine check dist/*
	@echo "✅ 发布包元数据校验通过"

check-manifest: ## 校验源码包文件清单
	python -m pip install --upgrade check-manifest
	check-manifest
	@echo "✅ MANIFEST 校验通过"

# 发布到PyPI（测试）
publish-test: ## 发布到TestPyPI
	python -m twine upload --repository testpypi dist/*
	@echo "✅ 已发布到TestPyPI"

# 发布到PyPI
publish: ## 发布到PyPI
	python -m twine upload dist/*
	@echo "✅ 已发布到PyPI"

# 多版本测试（需要tox）
tox-test: ## 多版本测试
	tox
	@echo "✅ 多版本测试完成"

# 完整检查
check: format lint type-check test ## 完整检查（格式化+lint+类型+测试）
	@echo "✅ 完整检查通过"

# 快速开始
quickstart: dev validate ## 快速开始（安装+验证）
	@echo ""
	@echo "🎉 环境准备完成！"
	@echo ""
	@echo "下一步:"
	@echo "  1. 运行 'make jupyter' 启动Jupyter"
	@echo "  2. 运行 'python examples/00_quickstart.py' 验证完整流程"
	@echo ""

# 克隆文档仓库
DOCS_REPO ?= https://github.com/hscredit/hscredit-docs.git
DOCS_DIR ?= _deploy_docs
clone-docs: ## 克隆 hscredit-docs 仓库
	rm -rf $(DOCS_DIR)
	git clone $(DOCS_REPO) $(DOCS_DIR)

# 推送文档到 hscredit-docs 仓库
push-docs: clone-docs ## 推送文档到 hscredit-docs 仓库
	rm -rf $(DOCS_DIR)/*
	cp -r docs/_build/html/* $(DOCS_DIR)/
	cd $(DOCS_DIR) && \
	git add -A && \
	git commit -m "docs: update $(shell date +%Y-%m-%d)" && \
	git push origin main || git push origin master
	rm -rf $(DOCS_DIR)
	@echo "✅ 文档已推送到 hscredit-docs 仓库"
