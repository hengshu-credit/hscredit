/**
 * 在隔离上下文中执行 Sphinx 搜索语言脚本，捕获加载期异常。
 */

"use strict";

const fs = require("node:fs");
const vm = require("node:vm");

const languageDataPath = process.argv[2];

try {
    if (!languageDataPath) {
        throw new Error("未提供 language_data.js 路径");
    }
    const source = fs.readFileSync(languageDataPath, "utf8");
    const context = vm.createContext({ window: {} });
    vm.runInContext(source, context, { filename: languageDataPath, timeout: 2000 });

    const Stemmer = context.Stemmer || context.window.Stemmer;
    if (typeof Stemmer !== "function") {
        throw new Error("language_data.js 未提供可调用的 Stemmer");
    }
    const stemmer = new Stemmer();
    if (typeof stemmer.stemWord !== "function" || typeof stemmer.stemWord("searching") !== "string") {
        throw new Error("Stemmer.stemWord() 不可用");
    }
} catch (error) {
    process.stderr.write(error && error.stack ? error.stack : String(error));
    process.exitCode = 1;
}
