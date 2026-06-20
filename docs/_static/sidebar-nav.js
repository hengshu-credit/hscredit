/**
 * Keep RTD's sidebar navigation from changing the document scroll position.
 *
 * RTD calls scrollIntoView() on the active menu link after every hash change.
 * With a sticky brand header this may also move the page viewport, so sidebar
 * links are handled inside .wy-side-scroll only.
 */
(function () {
    "use strict";

    var nativeScrollIntoView = Element.prototype.scrollIntoView;

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

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", decoupleDocumentScroll);
    } else {
        decoupleDocumentScroll();
    }
})();
