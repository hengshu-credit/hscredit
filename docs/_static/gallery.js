// 触摸屏和键盘使用原生滚动；鼠标可按住拖动当前面板的原图。
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
