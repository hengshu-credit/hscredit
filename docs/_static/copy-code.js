"use strict";

(function () {
    function fallbackCopy(text) {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();

        try {
            if (!document.execCommand("copy")) {
                throw new Error("浏览器拒绝复制操作");
            }
        } finally {
            textarea.remove();
        }
    }

    async function copyText(text) {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
            return;
        }
        fallbackCopy(text);
    }

    function getCopyableCode(codeBlock) {
        return codeBlock.textContent.replace(/^(?:>>>|\.\.\.)(?: )?/gm, "");
    }

    function showStatus(button, text, className) {
        window.clearTimeout(button.copyStatusTimer);
        button.textContent = text;
        button.classList.remove("is-copied", "is-error");
        button.classList.add(className);
        button.copyStatusTimer = window.setTimeout(function () {
            button.textContent = "复制";
            button.classList.remove("is-copied", "is-error");
        }, 1600);
    }

    function addCopyButtons() {
        document.querySelectorAll(".rst-content div.highlight > pre").forEach(function (codeBlock) {
            const container = codeBlock.parentElement;
            if (container.querySelector(":scope > .hs-copy-code")) {
                return;
            }

            const button = document.createElement("button");
            button.type = "button";
            button.className = "hs-copy-code";
            button.textContent = "复制";
            button.setAttribute("aria-label", "复制示例代码");
            button.setAttribute("title", "复制代码");
            button.addEventListener("click", async function () {
                try {
                    await copyText(getCopyableCode(codeBlock));
                    showStatus(button, "已复制", "is-copied");
                } catch (error) {
                    showStatus(button, "复制失败", "is-error");
                }
            });
            container.appendChild(button);
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", addCopyButtons);
    } else {
        addCopyButtons();
    }
})();
