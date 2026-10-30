/** 到底后继续下滚，按文档顺序进入下一页。 */
(function () {
    "use strict";

    function init() {
        var content = document.querySelector(".wy-nav-content");
        var next = content && content.querySelector("footer a[rel='next']");
        if (!next) { return; }

        var navigating = false;
        var bottomSince = null;
        var wheelDistance = 0;
        var lastWheel = 0;
        var touch = null;
        var hint = document.createElement("p");
        hint.className = "hs-next-page-hint";
        hint.textContent = "继续向下滚动，阅读下一页：" + (next.title || next.textContent.trim());
        next.closest(".rst-footer-buttons").insertAdjacentElement("afterend", hint);

        function atBottom() {
            var root = document.scrollingElement || document.documentElement;
            return root.scrollHeight - window.innerHeight - window.scrollY <= 2;
        }

        function updateBottom() {
            if (atBottom()) {
                if (bottomSince === null) { bottomSince = performance.now(); }
            } else {
                bottomSince = null;
                wheelDistance = 0;
            }
        }

        function readingContent(target) {
            var element = target instanceof Element ? target : target.parentElement;
            if (!element || !content.contains(element) || document.querySelector(".wy-nav-side.shift") ||
                element.isContentEditable || element.closest("input, textarea, select")) {
                return false;
            }
            // 表格、代码及其他嵌套滚动区域到底时仍由该区域处理。
            for (var node = element; node && node !== content; node = node.parentElement) {
                var style = getComputedStyle(node);
                if ((/(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight) ||
                    (/(auto|scroll)/.test(style.overflowX) && node.scrollWidth > node.clientWidth)) {
                    return false;
                }
            }
            return true;
        }

        function advance() {
            if (navigating) { return; }
            navigating = true;
            window.location.assign(next.href);
        }

        content.addEventListener("wheel", function (event) {
            if (navigating || event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey ||
                !readingContent(event.target) || Math.abs(event.deltaX) >= Math.abs(event.deltaY)) {
                wheelDistance = 0;
                return;
            }
            updateBottom();
            var now = performance.now();
            if (event.deltaY <= 0 || bottomSince === null || now - bottomSince < 350) {
                wheelDistance = 0;
                return;
            }
            // 不在首次触底时翻页，也不把轻微触控板抖动当成继续阅读。
            if (now - lastWheel > 250) { wheelDistance = 0; }
            lastWheel = now;
            var unit = event.deltaMode === 1 ? 16 : (event.deltaMode === 2 ? window.innerHeight : 1);
            wheelDistance += event.deltaY * unit;
            if (wheelDistance >= 100) {
                event.preventDefault();
                advance();
            }
        }, { passive: false });

        content.addEventListener("touchstart", function (event) {
            touch = null;
            if (event.touches.length === 1 && atBottom() && readingContent(event.target)) {
                var point = event.touches[0];
                touch = { id: point.identifier, x: point.clientX, y: point.clientY };
            }
        }, { passive: true });
        content.addEventListener("touchend", function (event) {
            var start = touch;
            touch = null;
            if (!start || event.touches.length || !atBottom()) { return; }
            var point = Array.prototype.find.call(event.changedTouches, function (item) { return item.identifier === start.id; });
            if (!point) { return; }
            var distance = start.y - point.clientY;
            if (distance >= 80 && distance > Math.abs(start.x - point.clientX)) { advance(); }
        }, { passive: true });
        content.addEventListener("touchcancel", function () { touch = null; }, { passive: true });
        window.addEventListener("scroll", updateBottom, { passive: true });
        window.addEventListener("resize", updateBottom);
        // 后退缓存会保留 JS 状态，重新进入页面时允许再次翻页。
        window.addEventListener("pageshow", function () {
            navigating = false;
            bottomSince = null;
            wheelDistance = 0;
            touch = null;
            updateBottom();
        });
        updateBottom();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
