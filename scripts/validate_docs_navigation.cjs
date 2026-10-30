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
            const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css" };
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
        await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
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
        assert.deepEqual(errors, []);
        console.log("目录交互通过：小标题、独立展开、锚点、正文滚动、跨页、刷新、深层 API、移动导航");
    } finally {
        if (browser) { await browser.close(); }
        await new Promise(resolve => server.close(resolve));
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
