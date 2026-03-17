# hscredit Makefile
# 简化常用操作

.PHONY: help install dev test validate clean jupyter

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
	jupyter nbconvert --to notebook --execute examples/00_project_overview.ipynb --output-dir outputs/
	@echo "✅ Notebook验证完成"

# 代码格式化
format: ## 格式化代码
	black hscredit tests
	@echo "✅ 代码格式化完成"

# 代码检查
lint: ## 检查代码质量
	flake8 hscredit tests
	@echo "✅ 代码检查完成"

# 类型检查
type-check: ## 类型检查
	mypy hscredit --ignore-missing-imports
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
	@echo "✅ 清理完成"

# 构建文档
docs: ## 构建文档
	cd docs && make html
	@echo "✅ 文档已生成: docs/_build/html/index.html"

# 构建发布包
build: ## 构建发布包
	python -m build
	@echo "✅ 发布包已生成: dist/"

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
	@echo "  2. 打开 examples/00_project_overview.ipynb"
	@echo "  3. 执行notebook进行验证"
	@echo ""
