/**
 * Keep RTD's sidebar navigation in sync with the content viewport.
 *
 * RTD calls scrollIntoView() on the active menu link after every hash change.
 * With a sticky brand header this may also move the page viewport, so sidebar
 * links are handled inside .wy-side-scroll only. A lightweight scrollspy then
 * marks the exact menu item for the heading/signature currently being read.
 */
(function () {
    "use strict";

    var nativeScrollIntoView = Element.prototype.scrollIntoView;
    var activeLink = null;
    var scrollTicking = false;

    Element.prototype.scrollIntoView = function (options) {
        var menu = this.closest && this.closest(".wy-menu-vertical");
        if (!menu) {
            return nativeScrollIntoView.call(this, options);
        }

        var scroller = this.closest(".wy-side-scroll");
        if (!scroller) {
            return;
        }

        var brand = scroller.querySelector(".wy-side-nav-search");
        var scrollerRect = scroller.getBoundingClientRect();
        var linkRect = this.getBoundingClientRect();
        var visibleTop = brand
            ? Math.max(scrollerRect.top, brand.getBoundingClientRect().bottom)
            : scrollerRect.top;
        var visibleBottom = scrollerRect.bottom;
        var spacing = 8;

        if (linkRect.top < visibleTop) {
            scroller.scrollTop -= visibleTop - linkRect.top + spacing;
        } else if (linkRect.bottom > visibleBottom) {
            scroller.scrollTop += linkRect.bottom - visibleBottom + spacing;
        }
    };

    function decoupleDocumentScroll() {
        var navigation = window.SphinxRtdTheme && window.SphinxRtdTheme.Navigation;
        if (!navigation) {
            return;
        }

        navigation.onScroll = function () {
            this.winScroll = false;
            this.winPosition = this.win.scrollTop();
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
        var menu = document.querySelector(".wy-menu-vertical");
        if (!menu) {
            return [];
        }

        return Array.prototype.slice.call(menu.querySelectorAll("a.reference.internal[href]"))
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
                    top: getTop(target),
                };
            })
            .filter(Boolean)
            .sort(function (a, b) {
                return a.top - b.top;
            });
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
        link.scrollIntoView({ block: "nearest" });
    }

    function syncCurrentMenuItem() {
        var links = getCurrentPageLinks();
        if (!links.length) {
            return;
        }

        var readLine = window.pageYOffset + Math.max(120, window.innerHeight * 0.18);
        var bottomGap = document.documentElement.scrollHeight - window.innerHeight - window.pageYOffset;
        var current = links[0];

        if (bottomGap < 4) {
            current = links[links.length - 1];
        } else {
            links.forEach(function (entry) {
                if (entry.top <= readLine) {
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
        syncCurrentMenuItem();
        window.addEventListener("scroll", requestSync, { passive: true });
        window.addEventListener("resize", requestSync);
        window.addEventListener("hashchange", requestSync);
    }

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
