// README 中标为滚动查看的原图仍可用鼠标拖动、键盘或触屏浏览。
for (const image of document.querySelectorAll('#readme-showcase img[src^="_static/showcase/"]')) {
    const pane = image.closest('div[style*="overflow"]');
    if (pane && pane.querySelectorAll("img").length === 1) pane.classList.add("hs-gallery-pane");
}
for (const pane of document.querySelectorAll(".hs-gallery-pane")) {
    const image = pane.querySelector("img");
    pane.tabIndex = 0;
    pane.setAttribute("role", "region");
    pane.setAttribute("aria-label", image.alt);
    image.draggable = false;

    let start;
    pane.addEventListener("pointerdown", event => {
        if (event.pointerType !== "mouse" || event.button !== 0) return;
        start = { x: event.clientX, y: event.clientY, left: pane.scrollLeft, top: pane.scrollTop };
        pane.setPointerCapture(event.pointerId);
        pane.classList.add("is-dragging");
    });
    pane.addEventListener("pointermove", event => {
        if (!start || !pane.hasPointerCapture(event.pointerId)) return;
        pane.scrollLeft = start.left + start.x - event.clientX;
        pane.scrollTop = start.top + start.y - event.clientY;
    });
    pane.addEventListener("lostpointercapture", () => {
        start = null;
        pane.classList.remove("is-dragging");
    });
}
