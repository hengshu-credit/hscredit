// hscredit文档自定义JavaScript

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 添加返回顶部按钮
    addBackToTopButton();
    
    // 添加代码复制功能增强
    enhanceCodeCopy();
    
    // 添加目录高亮
    highlightToc();
    
    // 添加外部链接提示
    markExternalLinks();
});

// 返回顶部按钮
function addBackToTopButton() {
    const button = document.createElement('button');
    button.innerHTML = '↑';
    button.className = 'back-to-top';
    button.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: #3f1dba;
        color: white;
        border: none;
        cursor: pointer;
        font-size: 24px;
        display: none;
        z-index: 1000;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    `;
    
    document.body.appendChild(button);
    
    // 滚动监听
    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            button.style.display = 'block';
        } else {
            button.style.display = 'none';
        }
    });
    
    // 点击事件
    button.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// 增强代码复制功能
function enhanceCodeCopy() {
    // 为代码块添加语言标签
    const codeBlocks = document.querySelectorAll('.highlight');
    codeBlocks.forEach(block => {
        const language = block.className.split(' ').find(c => c.startsWith('highlight-'));
        if (language) {
            const langName = language.replace('highlight-', '');
            const label = document.createElement('div');
            label.className = 'code-language-label';
            label.textContent = langName.toUpperCase();
            label.style.cssText = `
                position: absolute;
                top: 0;
                right: 0;
                background-color: #3f1dba;
                color: white;
                padding: 2px 8px;
                font-size: 12px;
                border-radius: 0 4px 0 4px;
            `;
            block.style.position = 'relative';
            block.appendChild(label);
        }
    });
}

// 目录高亮
function highlightToc() {
    const tocLinks = document.querySelectorAll('.toc a');
    const sections = document.querySelectorAll('h2, h3');
    
    window.addEventListener('scroll', function() {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (scrollY >= sectionTop - 100) {
                current = section.getAttribute('id');
            }
        });
        
        tocLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    });
}

// 标记外部链接
function markExternalLinks() {
    const links = document.querySelectorAll('a[href^="http"]');
    links.forEach(link => {
        if (!link.href.includes(window.location.hostname)) {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
            
            // 添加外部链接图标
            if (!link.querySelector('.external-link-icon')) {
                const icon = document.createElement('span');
                icon.className = 'external-link-icon';
                icon.innerHTML = ' ↗';
                icon.style.cssText = 'font-size: 0.8em;';
                link.appendChild(icon);
            }
        }
    });
}

// 搜索功能增强
function enhanceSearch() {
    const searchInput = document.querySelector('input[type="search"]');
    if (searchInput) {
        // 添加快捷键
        document.addEventListener('keydown', function(e) {
            if (e.key === '/' && document.activeElement !== searchInput) {
                e.preventDefault();
                searchInput.focus();
            }
            
            if (e.key === 'Escape' && document.activeElement === searchInput) {
                searchInput.blur();
            }
        });
        
        // 添加搜索提示
        searchInput.setAttribute('placeholder', '搜索文档... (按 / 快速聚焦)');
    }
}

// 代码行号
function addLineNumbers() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        const lines = block.textContent.split('\n');
        if (lines.length > 3) {
            const numberedLines = lines.map((line, i) => {
                return `<span class="line-number">${i + 1}</span> ${line}`;
            }).join('\n');
            block.innerHTML = numberedLines;
        }
    });
}

// 工具提示
function addTooltips() {
    const apiElements = document.querySelectorAll('.py.class, .py.method, .py.function');
    apiElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            this.style.boxShadow = '0 2px 8px rgba(63, 29, 186, 0.1)';
        });
        
        element.addEventListener('mouseleave', function() {
            this.style.boxShadow = 'none';
        });
    });
}

// 平滑滚动
function smoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// 初始化所有增强功能
smoothScroll();
enhanceSearch();
addTooltips();
