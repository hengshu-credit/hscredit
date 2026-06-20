# hscredit 文档 Makefile（Sphinx）
# 使用方式：
#   make html      # 构建 HTML 到 _build/html
#   make clean     # 清理构建产物（sphinx-build -M clean）
#
# 采用 Sphinx 官方 Makefile 的「catch-all 转发」写法：除 help 外的目标（html/clean/
# latexpdf 等）统一转发给 sphinx-build -M。注意 html/clean 不要放进 .PHONY，否则会被
# 当作无配方的伪目标而跳过 catch-all 规则。

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# 默认目标：显示帮助
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# catch-all：把未知目标转发给 sphinx-build
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
