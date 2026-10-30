/**
 * Keep RTD's sidebar navigation in sync with the content viewport.
 *
 * Expansion, reading position and scroll position are independent. RTD's
 * default reset collapses sibling branches on every hash change; its scroll
 * handler also couples the sidebar to the document. Keep both under user control.
 */
(function () {
    "use strict";

    var activeLink = null;
    var scrollTicking = false;
    var restored = false;
    var pageLinks = null;
    var pendingLink = null;

    function keepVisible(link, alignStart) {
        var scroller = link.closest(".wy-side-scroll");
        if (!scroller || !link.getClientRects().length) { return; }
        var brand = scroller.querySelector(".wy-side-nav-search");
        var scrollerRect = scroller.getBoundingClientRect();
        var linkRect = link.getBoundingClientRect();
        var visibleTop = brand
            ? Math.max(scrollerRect.top, brand.getBoundingClientRect().bottom)
            : scrollerRect.top;
        var visibleBottom = scrollerRect.bottom;
        var spacing = 8;

        if (alignStart || linkRect.top < visibleTop) {
            scroller.scrollTop -= visibleTop - linkRect.top + spacing;
        } else if (linkRect.bottom > visibleBottom) {
            scroller.scrollTop += linkRect.bottom - visibleBottom + spacing;
        }
    }

    function branchLink(item) {
        return item.querySelector(":scope > a");
    }

    function positionNewPage(link) {
        var scroller = link.closest(".wy-side-scroll");
        keepVisible(link, true);
        var expectedTop = scroller.scrollTop;
        // 字体和 Logo 载入会改变品牌栏及菜单高度；只补齐首次定位，
        // 用户已滚动侧栏或开始阅读时不再移动目录。
        function settle() {
            if (activeLink !== link || Math.abs(scroller.scrollTop - expectedTop) > 2) { return; }
            keepVisible(link, true);
            expectedTop = scroller.scrollTop;
        }
        window.addEventListener("load", settle, { once: true });
        if (document.fonts) { document.fonts.ready.then(settle); }
    }

    function branches() {
        return Array.prototype.filter.call(document.querySelectorAll(".wy-menu-vertical li"), function (item) {
            return item.querySelector(":scope > ul") && branchLink(item);
        });
    }

    function linkKey(link) {
        var url = makeUrl(link.getAttribute("href"));
        return url ? url.pathname + url.hash : "";
    }

    function storageKey() {
        var root = document.documentElement.getAttribute("data-content_root") ||
            (window.DOCUMENTATION_OPTIONS && window.DOCUMENTATION_OPTIONS.URL_ROOT) || "./";
        return "hscredit.sidebar:" + new URL(root || "./", window.location.href).pathname;
    }

    function setExpanded(item, expanded) {
        item.classList.toggle("current", expanded);
        item.setAttribute("aria-expanded", String(expanded));
        var link = branchLink(item);
        link.classList.remove("current");
        link.setAttribute("aria-expanded", String(expanded));
        var button = link.querySelector(".toctree-expand");
        if (button) { button.setAttribute("aria-expanded", String(expanded)); }
    }

    function savePosition(link) {
        var scroller = document.querySelector(".wy-side-scroll");
        if (!scroller) { return; }
        var state = {
            expanded: branches().filter(function (item) { return item.classList.contains("current"); })
                .map(function (item) { return linkKey(branchLink(item)); }),
            scrollTop: scroller.scrollTop,
            page: window.location.pathname + window.location.hash,
            anchor: link ? linkKey(link) : null,
            offset: link ? link.getBoundingClientRect().top - scroller.getBoundingClientRect().top : null
        };
        try { sessionStorage.setItem(storageKey(), JSON.stringify(state)); } catch (error) { /* Storage is optional. */ }
    }

    function reveal(link) {
        for (var item = link.closest("li"); item; item = item.parentElement.closest("li")) {
            if (item.querySelector(":scope > ul")) { setExpanded(item, true); }
        }
    }

    function resetNavigation() {
        var entries = getCurrentPageLinks();
        var current = entries.find(function (entry) {
            return makeUrl(entry.link.getAttribute("href")).hash === window.location.hash;
        }) || entries[0];
        var scroller = document.querySelector(".wy-side-scroll");
        var state = null;
        if (!restored) {
            try { state = JSON.parse(sessionStorage.getItem(storageKey())); } catch (error) { /* No saved state. */ }
            if (state && Array.isArray(state.expanded)) {
                branches().forEach(function (item) {
                    setExpanded(item, state.expanded.indexOf(linkKey(branchLink(item))) !== -1);
                });
            }
        }
        var previousTop = scroller.scrollTop;
        if (current) {
            reveal(current.link);
            setActiveLink(current.link);
        }
        if (!restored && state && Number.isFinite(state.scrollTop)) {
            scroller.scrollTop = state.scrollTop;
            var anchor = Array.prototype.find.call(document.querySelectorAll(".wy-menu-vertical a[href]"), function (link) {
                return linkKey(link) === state.anchor && link.getClientRects().length;
            });
            if (anchor && Number.isFinite(state.offset)) {
                scroller.scrollTop += anchor.getBoundingClientRect().top - scroller.getBoundingClientRect().top - state.offset;
            }
            if (current && !anchor && state.page !== window.location.pathname + window.location.hash) {
                // 自动续读和页脚导航没有侧栏点击锚点，应定位新页面，
                // 而不是沿用上一页的目录滚动位置。手动点击与刷新仍保留位置。
                positionNewPage(current.link);
            }
        } else if (!restored) {
            if (current) { keepVisible(current.link); }
        } else {
            scroller.scrollTop = previousTop;
        }
        restored = true;
    }

    function decoupleDocumentScroll() {
        var navigation = window.SphinxRtdTheme && window.SphinxRtdTheme.Navigation;
        if (!navigation) {
            return;
        }

        navigation.onScroll = function () {
            this.winScroll = false;
            this.winPosition = this.win.scrollTop();
        };
        navigation.reset = resetNavigation;
        navigation.toggleCurrent = function (anchor) {
            var link = anchor[0];
            var item = link.closest("li");
            var scroller = link.closest(".wy-side-scroll");
            var top = scroller.scrollTop;
            setExpanded(item, !item.classList.contains("current"));
            scroller.scrollTop = top;
            savePosition();
        };
    }

    function samePath(url) {
        var currentPath = window.location.pathname.replace(/\/index\.html$/, "/");
        var targetPath = url.pathname.replace(/\/index\.html$/, "/");
        return currentPath === targetPath;
    }

    function makeUrl(href) {
        try {
            return new URL(href, window.location.href);
        } catch (error) {
            return null;
        }
    }

    function getTarget(hash) {
        if (!hash || hash === "#") {
            return document.querySelector(".rst-content .document") || document.body;
        }

        return document.getElementById(decodeURIComponent(hash.slice(1)));
    }

    function getTop(element) {
        if (element === document.body) {
            return 0;
        }
        return element.getBoundingClientRect().top + window.pageYOffset;
    }

    function getCurrentPageLinks() {
        if (pageLinks) { return pageLinks; }
        var menu = document.querySelector(".wy-menu-vertical");
        if (!menu) {
            return [];
        }

        pageLinks = Array.prototype.slice.call(menu.querySelectorAll("a.reference.internal[href]"))
            .map(function (link) {
                var url = makeUrl(link.getAttribute("href"));
                if (!url || !samePath(url)) {
                    return null;
                }

                var target = getTarget(url.hash);
                if (!target) {
                    return null;
                }

                return {
                    link: link,
                    target: target,
                };
            })
            .filter(Boolean)
            .sort(function (a, b) { return getTop(a.target) - getTop(b.target); });
        return pageLinks;
    }

    function setActiveLink(link) {
        if (!link || link === activeLink) {
            return;
        }

        document.querySelectorAll(".wy-menu-vertical li.hs-scrollspy-current").forEach(function (item) {
            item.classList.remove("hs-scrollspy-current");
        });
        document.querySelectorAll(".wy-menu-vertical a[aria-current='location']").forEach(function (item) {
            item.removeAttribute("aria-current");
        });

        var item = link.closest("li");
        if (item) {
            item.classList.add("hs-scrollspy-current");
        }
        link.setAttribute("aria-current", "location");
        activeLink = link;
    }

    function syncCurrentMenuItem() {
        var links = getCurrentPageLinks();
        if (!links.length) {
            return;
        }

        var readLine = window.pageYOffset + Math.max(120, window.innerHeight * 0.18);
        var bottomGap = document.documentElement.scrollHeight - window.innerHeight - window.pageYOffset;
        var current = links[0];

        if (window.pageYOffset <= 1) {
            current = links[0];
        } else if (bottomGap < 4) {
            current = links[links.length - 1];
        } else {
            links.forEach(function (entry) {
                if (getTop(entry.target) <= readLine) {
                    current = entry;
                }
            });
        }

        setActiveLink(current.link);
    }

    function requestSync() {
        if (scrollTicking) {
            return;
        }

        scrollTicking = true;
        window.requestAnimationFrame(function () {
            scrollTicking = false;
            syncCurrentMenuItem();
        });
    }

    function initScrollSpy() {
        // Normal links navigate without also toggling their branch. Only the
        // expand button changes expansion; RTD still handles that button.
        var menu = document.querySelector(".wy-menu-vertical");
        menu.addEventListener("click", function (event) {
            var link = event.target.closest("a[href]");
            if (!link || event.target.closest(".toctree-expand")) { return; }
            event.stopPropagation();
            if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) { return; }
            pendingLink = samePath(makeUrl(link.getAttribute("href"))) ? null : link;
            savePosition(link);
            document.querySelectorAll("[data-toggle='wy-nav-shift']").forEach(function (item) {
                item.classList.remove("shift");
            });
        }, true);
        window.addEventListener("pagehide", function () {
            savePosition(pendingLink);
        });
        window.addEventListener("scroll", requestSync, { passive: true });
        window.addEventListener("resize", requestSync);
        window.addEventListener("hashchange", requestSync);
    }

    // Patch before the theme registers its ready/hashchange callbacks.
    decoupleDocumentScroll();
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
            decoupleDocumentScroll();
            initScrollSpy();
        });
    } else {
        decoupleDocumentScroll();
        initScrollSpy();
    }
})();
