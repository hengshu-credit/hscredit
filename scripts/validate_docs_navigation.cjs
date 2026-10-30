/** 在真实浏览器中校验章节目录、展开状态和滚动位置。 */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const http = require("node:http");
const path = require("node:path");
const { chromium } = require("playwright");

async function main() {
    const root = path.resolve(process.argv[2] || "docs/_build/html");
    const server = http.createServer(async (request, response) => {
        try {
            const file = path.resolve(root, "." + decodeURIComponent(new URL(request.url, "http://localhost").pathname));
            assert(file.startsWith(root + path.sep));
            const types = {
                ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
                ".svg": "image/svg+xml", ".woff2": "font/woff2", ".ttf": "font/ttf",
            };
            response.setHeader("Content-Type", types[path.extname(file)] || "application/octet-stream");
            response.end(await fs.readFile(file));
        } catch (error) {
            response.statusCode = 404;
            response.end();
        }
    });
    await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
    let browser;
    try {
        browser = await chromium.launch({
            headless: true,
            ...(process.env.HSCREDIT_BROWSER_EXECUTABLE ? { executablePath: process.env.HSCREDIT_BROWSER_EXECUTABLE } : {}),
        });
        const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
        const errors = [];
        page.on("pageerror", error => errors.push(error.message));
        const base = `http://127.0.0.1:${server.address().port}`;
        const nav = ".wy-menu-vertical";
        async function ready() {
            await page.evaluate(() => document.fonts.ready);
            await page.waitForTimeout(150);
        }
        async function currentPageInView(targetPage = page) {
            const position = await targetPage.locator(`${nav} a[aria-current='location']`).evaluateAll(links => {
                const link = links[0];
                if (!link) { return { count: links.length }; }
                const scroller = link.closest(".wy-side-scroll");
                const brand = scroller.querySelector(".wy-side-nav-search");
                const bounds = link.getBoundingClientRect();
                return {
                    count: links.length,
                    page: new URL(link.href).pathname,
                    expected: location.pathname,
                    top: bounds.top,
                    bottom: bounds.bottom,
                    visibleTop: brand.getBoundingClientRect().bottom,
                    visibleBottom: scroller.getBoundingClientRect().bottom,
                    remaining: scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop,
                    highlight: getComputedStyle(link).borderImageSource,
                };
            });
            assert.equal(position.count, 1, "当前阅读项不唯一");
            assert.equal(position.page, position.expected, "高亮仍指向上一页");
            assert(position.top >= position.visibleTop && position.bottom <= position.visibleBottom, JSON.stringify(position));
            assert(position.top <= position.visibleTop + 40 || position.remaining <= 2,
                "新页面目录未定位到品牌栏下方：" + JSON.stringify(position));
            assert(position.highlight.includes("linear-gradient"), "当前页缺少独立的定位高亮");
        }
        async function state(link) {
            return link.evaluate(a => ({
                top: a.getBoundingClientRect().top,
                left: a.getBoundingClientRect().left,
                side: document.querySelector(".wy-side-scroll").scrollTop,
                body: window.scrollY,
            }));
        }
        function stable(before, after, keys) {
            for (const key of keys) {
                assert(Math.abs(before[key] - after[key]) <= 2, `${key} 跳动：${before[key]} -> ${after[key]}`);
            }
        }
        async function prepare(link) {
            await link.scrollIntoViewIfNeeded();
            // sticky 品牌栏不属于目录的可点击区域。
            await link.evaluate(a => {
                const scroller = a.closest(".wy-side-scroll");
                const brand = scroller.querySelector(".wy-side-nav-search");
                const target = Math.max(brand.getBoundingClientRect().bottom + 40, window.innerHeight / 2);
                scroller.scrollTop += a.getBoundingClientRect().top - target;
            });
        }
        async function toggle(link) {
            const button = link.locator(":scope > button");
            await prepare(link);
            const before = await state(link);
            await button.click();
            await ready();
            stable(before, await state(link), ["top", "left", "side", "body"]);
        }
        await page.goto(base + "/quickstart.html");
        await ready();
        for (const name of ["installation", "analysis_parameters", "database", "skills", "multiway_tree"]) {
            assert(await page.locator(`${nav} a[href^="${name}.html#"]`).count() > 0, `${name} 缺少小标题`);
        }
        const install = page.locator(`${nav} a[href="installation.html"]`);
        const analysis = page.locator(`${nav} a[href="analysis_parameters.html"]`);
        await toggle(install);
        await toggle(analysis);
        assert(await install.evaluate(a => a.closest("li").classList.contains("current")), "相邻目录被自动收起");
        await toggle(analysis); // 收起也应保持位置。
        const section = page.locator(`${nav} a[href^="#"]`).filter({ hasNot: page.locator("button") }).first();
        await prepare(section);
        const before = await state(section);
        await section.click();
        await ready();
        stable(before, await state(section), ["top", "left", "side"]);
        await page.evaluate(() => window.scrollTo(0, document.scrollingElement.scrollHeight));
        await ready();
        stable(before, await state(section), ["side"]);
        assert(await install.evaluate(a => a.closest("li").classList.contains("current")), "锚点跳转收起了其他目录");

        await prepare(install);
        const beforePage = await state(install);
        await install.click();
        await page.waitForURL(base + "/installation.html");
        await ready();
        const current = page.locator(`${nav} a[href="#"]`);
        stable(beforePage, await state(current), ["top", "left", "side"]);
        await page.reload();
        await ready();
        stable(beforePage, await state(current), ["top", "left", "side"]);

        const hash = "#hscredit.core.models.boosting.xgboost_model.XGBoost.fit";
        await page.goto(base + "/api/boosting.html" + hash);
        await ready();
        const fit = page.locator(`${nav} a[href="${hash}"]`);
        assert(await fit.isVisible(), "深层方法目录未展开");
        const net = page.locator(`${nav} a[href="net.html"]`);
        await toggle(net);
        assert(await fit.isVisible(), "展开模型目录时收起了当前类的方法");
        const method = page.locator(`${nav} a[href="#hscredit.core.models.boosting.xgboost_model.XGBoost.predict"]`);
        await prepare(method);
        const beforeMethod = await state(method);
        await method.click();
        await ready();
        stable(beforeMethod, await state(method), ["top", "left", "side"]);
        await page.setViewportSize({ width: 390, height: 844 });
        await page.locator("[data-toggle='wy-nav-top']").click();
        assert(await page.locator(".wy-nav-side").evaluate(el => el.classList.contains("shift")), "移动目录无法展开");
        await page.locator(`${nav} a[href="#"]`).click();
        assert(await page.locator(".wy-nav-side").evaluate(el => !el.classList.contains("shift")), "移动目录导航后未关闭");

        // 正文触底后才接受继续下滚；静止、侧栏和嵌套区域不翻页。
        await page.setViewportSize({ width: 1440, height: 1000 });
        await page.goto(base + "/quickstart.html");
        await ready();
        const nextUrl = await page.locator("footer a[rel='next']").evaluate(a => a.href);
        assert(await page.locator(".hs-next-page-hint").count() === 1, "缺少继续阅读提示");
        await page.mouse.move(1000, 750);
        await page.mouse.wheel(0, 100000);
        await page.waitForFunction(() => document.scrollingElement.scrollHeight - innerHeight - scrollY <= 2);
        await page.waitForTimeout(450);
        assert.equal(page.url(), base + "/quickstart.html", "首次触底或静止时自动翻页");
        await page.mouse.move(150, 700);
        await page.mouse.wheel(0, 600);
        await page.waitForTimeout(200);
        assert.equal(page.url(), base + "/quickstart.html", "侧栏滚动触发翻页");
        // 用实际滚动容器覆盖图表、代码和表格的边界情况。
        await page.evaluate(() => {
            const area = document.createElement("div");
            area.id = "nested-scroll-test";
            area.style.cssText = "position:fixed;right:40px;bottom:40px;width:240px;height:160px;overflow:auto;z-index:9999;background:white";
            area.innerHTML = '<div style="height:600px;width:800px">嵌套内容</div>';
            document.querySelector(".wy-nav-content").append(area);
            area.scrollTop = area.scrollHeight;
        });
        await page.locator("#nested-scroll-test").hover();
        await page.mouse.wheel(0, 600);
        await page.waitForTimeout(200);
        assert.equal(page.url(), base + "/quickstart.html", "嵌套区域触底导致翻页");
        await page.locator("#nested-scroll-test").evaluate(el => el.remove());
        await page.mouse.move(1000, 750);
        await page.mouse.wheel(200, 10);
        await page.waitForTimeout(200);
        assert.equal(page.url(), base + "/quickstart.html", "横向滚动触发翻页");
        await page.locator(".wy-nav-content").dispatchEvent("wheel", { deltaY: 400, ctrlKey: true });
        await page.waitForTimeout(200);
        assert.equal(page.url(), base + "/quickstart.html", "缩放手势触发翻页");
        await page.mouse.wheel(0, 120);
        await page.waitForURL(nextUrl);
        await ready();
        await currentPageInView();
        assert.equal(await page.evaluate(() => scrollY), 0, "翻页后未从页首开始");
        // 模拟历史返回仍可继续向下阅读。
        await page.goBack();
        await ready();
        await page.evaluate(() => window.scrollTo(0, document.scrollingElement.scrollHeight));
        await page.waitForTimeout(450);
        await page.mouse.wheel(0, 120);
        await page.waitForURL(nextUrl);
        await ready();
        await currentPageInView();

        // 连续跨越指南、数据接入和 API 包目录，覆盖用户截图中的层级变化。
        await page.goto(base + "/articles/model-interpretability.html");
        for (let step = 0; step < 4; step++) {
            await ready();
            const next = await page.locator("footer a[rel='next']").evaluate(a => a.href);
            await page.locator(".wy-side-scroll").evaluate(el => { el.scrollTop = 0; });
            await page.evaluate(() => window.scrollTo(0, document.scrollingElement.scrollHeight));
            await page.waitForTimeout(450);
            await page.mouse.move(1100, 950);
            await page.mouse.wheel(0, 120);
            await page.waitForURL(next);
            await ready();
            await currentPageInView();
            const inactiveHighlights = await page.locator(`${nav} li.current > a:not([aria-current='location'])`)
                .evaluateAll(links => links.filter(link => getComputedStyle(link).borderImageSource !== "none").length);
            assert.equal(inactiveHighlights, 0, "展开的旧分支仍显示当前项霓虹边框");
        }
        // 使用完整导航中的真实最后一页，而不是删掉页面节点制造末页。
        await page.goto(base + "/index.html");
        const lastUrl = await page.locator(`${nav} a[href]`).evaluateAll(links => {
            const pages = links.filter(a => !a.getAttribute("href").includes("#") && new URL(a.href).origin === location.origin);
            return pages[pages.length - 1].href;
        });
        await page.goto(lastUrl);
        await ready();
        assert.equal(await page.locator("footer a[rel='next']").count(), 0, "未找到真实末页");
        assert.equal(await page.locator(".hs-next-page-hint").count(), 0, "末页仍显示下一页提示");
        await page.evaluate(() => window.scrollTo(0, document.scrollingElement.scrollHeight));
        await page.waitForTimeout(450);
        await page.mouse.wheel(0, 400);
        await page.waitForTimeout(200);
        assert.equal(page.url(), lastUrl, "末页触发了错误跳转");

        const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
        mobile.on("pageerror", error => errors.push(error.message));
        await mobile.goto(base + "/quickstart.html");
        const mobileNext = await mobile.locator("footer a[rel='next']").evaluate(a => a.href);
        const input = await mobile.context().newCDPSession(mobile);
        async function swipe(distance) {
            await input.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x: 280, y: 650 }] });
            await input.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: 280, y: 650 - distance }] });
            await input.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
        }
        await mobile.evaluate(() => scrollTo(0, document.scrollingElement.scrollHeight - innerHeight - 50));
        await swipe(250);
        await mobile.waitForFunction(() => document.scrollingElement.scrollHeight - innerHeight - scrollY <= 2);
        await mobile.waitForTimeout(450);
        assert.equal(mobile.url(), base + "/quickstart.html", "首次滑动触底时自动翻页");
        await swipe(30);
        await mobile.waitForTimeout(200);
        assert.equal(mobile.url(), base + "/quickstart.html", "轻微触摸触发翻页");
        await swipe(180);
        await mobile.waitForURL(mobileNext);
        await mobile.evaluate(() => document.fonts.ready);
        await mobile.locator("[data-toggle='wy-nav-top']").click();
        await currentPageInView(mobile);
        await mobile.close();
        assert.deepEqual(errors, []);
        console.log("文档交互通过：目录层级与滚动、跨页与刷新、移动导航、滚轮与触摸续读、嵌套区域与末页边界");
    } finally {
        if (browser) { await browser.close(); }
        await new Promise(resolve => server.close(resolve));
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
