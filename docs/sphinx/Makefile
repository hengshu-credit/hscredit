# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# 自定义命令

# 生成HTML文档
html:
	@echo "生成HTML文档..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "✅ 文档已生成: $(BUILDDIR)/html/index.html"

# 清理构建文件
clean:
	@echo "清理构建文件..."
	@rm -rf $(BUILDDIR)/*
	@echo "✅ 清理完成"

# 实时预览（需要sphinx-autobuild）
livehtml:
	@echo "启动实时预览..."
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --open-browser

# 生成PDF（需要安装latex）
pdf:
	@echo "生成PDF文档..."
	@$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	cd $(BUILDDIR)/latex && make all-pdf
	@echo "✅ PDF已生成: $(BUILDDIR)/latex/*.pdf"

# 检查链接
linkcheck:
	@echo "检查外部链接..."
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# 部署到GitHub Pages
deploy:
	@echo "部署到GitHub Pages..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	ghp-import -n -p -f $(BUILDDIR)/html
	@echo "✅ 部署完成"

# 查看文档覆盖率
coverage:
	@echo "检查文档覆盖率..."
	@$(SPHINXBUILD) -M coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@cat $(BUILDDIR)/coverage/python.txt
