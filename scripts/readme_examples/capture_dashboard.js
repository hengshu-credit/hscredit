// 通过 Playwright CLI 的 run-code 执行，保存 README 的真实 Dashboard 长图。
// locator.screenshot 会滚动到元素；切换路由后也可能保留滚动位置。
// 必须先回到文档顶部，否则 fixed 导航栏会出现在长图中间。
async (page, options = {}) => {
  const output = options.path || "docs/assets/readme/showcase/tuning-dashboard.png";
  const viewport = options.viewport || { width: 1440, height: 1000 };
  if (!/\/dashboard\/studies\/\d+\/?$/.test(new URL(page.url()).pathname)) {
    throw new Error("请先打开需要截取的 Study 概览页面。");
  }
  await page.setViewportSize(viewport);
  await page.waitForFunction(() => {
    const graphs = Array.from(document.querySelectorAll(".js-plotly-plot"));
    return graphs.length > 0 && graphs.every(graph => graph._fullLayout && graph.data?.length > 0);
  });
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => window.scrollTo({ top: 0, left: 0, behavior: "instant" }));
  await page.waitForFunction(() => window.scrollY === 0 && window.scrollX === 0);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const geometry = await page.evaluate(() => {
    const bar = document.querySelector("header.MuiAppBar-root");
    if (!bar) return null;
    const bounds = bar.getBoundingClientRect();
    const firstHeading = document.querySelector("main h6");
    return {
      scrollY: window.scrollY,
      top: bounds.top,
      bottom: bounds.bottom,
      position: getComputedStyle(bar).position,
      firstHeadingTop: firstHeading?.getBoundingClientRect().top,
    };
  });
  if (!geometry || geometry.scrollY !== 0 || Math.abs(geometry.top) > 1 || !Number.isFinite(geometry.firstHeadingTop) || geometry.firstHeadingTop < geometry.bottom) {
    throw new Error("看板截图位置异常：" + JSON.stringify(geometry));
  }
  await page.screenshot({ path: output, fullPage: true, animations: "disabled" });
  return { output, viewport, geometry };
};
