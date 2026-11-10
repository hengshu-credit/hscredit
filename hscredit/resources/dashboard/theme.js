/* 在原生 Plotly 绘制入口设置主题，保持数据、事件、交互与图片导出。 */
(() => {
  "use strict";
  const config = window.__hscreditDashboardConfig;
  const wrapped = new WeakSet();
  const family = config.font + ', "Microsoft YaHei", sans-serif';

  function isDark() {
    const root = document.querySelector("#dashboard > div");
    const background = root ? getComputedStyle(root).backgroundColor : "";
    const channels = background.match(/\d+(?:\.\d+)?/g);
    const dark = channels && channels.length >= 3 && channels.slice(0, 3).map(Number).reduce((a, b) => a + b, 0) < 180 && !(channels.length > 3 && +channels[3] === 0);
    document.documentElement.dataset.hscreditDark = dark ? "true" : "false";
    return !!dark;
  }

  function metricColor(name) {
    const label = String(name || "").toUpperCase();
    for (const metric of ["AUC", "KS", "IV", "PSI", "LIFT"]) {
      if (new RegExp("(^|[^A-Z])" + metric + "([^A-Z]|$)").test(label)) return config.metrics[metric.toLowerCase()];
    }
    return null;
  }

  function themeFigure(data, layout) {
    const dark = isDark();
    const text = dark ? "#F6F7FE" : "#282744";
    const grid = dark ? "#37305A" : "#E6E8FC";
    const bg = dark ? "#140E35" : "#FFFFFF";
    const font = original => ({ ...(original || {}), family, color: text });
    const outputLayout = { ...(layout || {}), font: font(layout?.font), colorway: [...config.series], paper_bgcolor: bg, plot_bgcolor: bg };
    if (outputLayout.title) outputLayout.title = typeof outputLayout.title === "string" ? { text: outputLayout.title, font: font() } : { ...outputLayout.title, font: font(outputLayout.title.font) };
    outputLayout.legend = { ...(outputLayout.legend || {}), font: font(outputLayout.legend?.font) };
    outputLayout.hoverlabel = { ...(outputLayout.hoverlabel || {}), bgcolor: bg, font: font(outputLayout.hoverlabel?.font) };
    for (const key of Object.keys(outputLayout)) {
      if (/^[xy]axis\d*$/.test(key)) {
        const axis = outputLayout[key] || {};
        const title = typeof axis.title === "string" ? { text: axis.title } : (axis.title || {});
        outputLayout[key] = { ...axis, gridcolor: grid, linecolor: dark ? config.lightBlue : config.primary, tickfont: font(axis.tickfont), title: { ...title, font: font(title.font) } };
      }
    }
    if (outputLayout.annotations) outputLayout.annotations = outputLayout.annotations.map(note => ({ ...note, font: font(note.font) }));
    const traces = (data || []).map((trace, index) => {
      const colorbarTitle = trace.marker?.colorbar?.title;
      const name = String(trace.name || (typeof colorbarTitle === "string" ? colorbarTitle : colorbarTitle?.text) || "");
      const best = /best|pareto|最优|最佳|帕累托/i.test(name);
      const color = metricColor(name) || (best ? config.secondary : /trial|complete/i.test(name) ? config.primary : config.series[index % config.series.length]);
      const scale = [[0, dark ? "#37305A" : "#F6F7FE"], [1, color]];
      const result = { ...trace };
      if (trace.marker || ["scatter", "scattergl", "bar"].includes(trace.type || "scatter")) {
        const marker = { ...(trace.marker || {}) };
        if (Array.isArray(marker.color) || ArrayBuffer.isView(marker.color)) {
          if (Array.from(marker.color).every(value => typeof value === "number")) {
            marker.colorscale = scale;
            marker.reversescale = false;
          }
          // 类别数组中的选中/未选中含义仍由上游控制。
        } else marker.color = color;
        if (marker.colorbar) marker.colorbar = { ...marker.colorbar, tickfont: font(marker.colorbar.tickfont), title: typeof marker.colorbar.title === "string" ? { text: marker.colorbar.title, font: font() } : { ...(marker.colorbar.title || {}), font: font(marker.colorbar.title?.font) } };
        result.marker = marker;
      }
      if (trace.line || /lines/.test(trace.mode || "")) result.line = { ...(trace.line || {}), color };
      if (trace.colorscale || ["heatmap", "contour", "surface", "parcoords"].includes(trace.type)) {
        if (trace.type === "parcoords") result.line = { ...(trace.line || {}), colorscale: scale };
        else result.colorscale = scale;
      }
      if (trace.textfont) result.textfont = font(trace.textfont);
      if (trace.type === "bar" && trace.textposition === "outside") result.cliponaxis = false;
      return result;
    });
    return [traces, outputLayout];
  }

  function wrapPlotly(plotly) {
    if (wrapped.has(plotly)) return;
    wrapped.add(plotly);
    for (const method of ["newPlot", "react"]) {
      const original = plotly[method];
      plotly[method] = function (element, data, layout, options) {
        if (data && !Array.isArray(data) && Object.prototype.hasOwnProperty.call(data, "data")) {
          const [traces, themed] = themeFigure(data.data, data.layout);
          return original.call(this, element, { ...data, data: traces, layout: themed });
        }
        const [traces, themed] = themeFigure(data, layout);
        return original.call(this, element, traces, themed, options);
      };
    }
  }

  function formatAttributeText(raw) {
    // 只格式化完整的 JSON 对象/数组，普通字符串和上游截断的内容保持原样。
    const source = raw.trim();
    if (!source.startsWith("{") && !source.startsWith("[")) return null;
    try { JSON.parse(source); } catch { return null; }
    // 保留数字原文，避免 JSON.parse/stringify 将大整数或高精度小数舍入。
    const tokens = source.match(/"(?:\\.|[^"\\])*"|[{}\[\],:]|true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/g);
    let result = "";
    let depth = 0;
    const newline = () => { result += "\n" + "  ".repeat(depth); };
    tokens.forEach((token, index) => {
      if (token === "{" || token === "[") {
        result += token;
        if (tokens[index + 1] !== "}" && tokens[index + 1] !== "]") { depth++; newline(); }
      } else if (token === "}" || token === "]") {
        if (tokens[index - 1] !== "{" && tokens[index - 1] !== "[") { depth--; newline(); }
        result += token;
      } else if (token === ",") {
        result += token;
        newline();
      } else if (token === ":") result += ": ";
      else result += token.startsWith('"') ? JSON.stringify(JSON.parse(token)) : token;
    });
    return result;
  }

  function formatAttributeTables(root) {
    for (const heading of root.querySelectorAll("h6")) {
      if (!/^(?:(?:Study|Trial) )?User Attributes$/.test(heading.textContent.trim())) continue;
      const section = heading.closest(".MuiCardContent-root");
      if (!section) continue;
      for (const table of section.querySelectorAll("table")) {
        if (!table.classList.contains("hscredit-attributes-table")) table.classList.add("hscredit-attributes-table");
        for (const row of table.querySelectorAll("tbody tr")) {
          const cell = row.cells[1];
          // 保留 React 管理的文本节点；不替换子树、不触碰编辑控件或注入 HTML。
          if (!cell || cell.childNodes.length !== 1 || cell.firstChild.nodeType !== 3) continue;
          const formatted = formatAttributeText(cell.firstChild.data);
          if (formatted === null) continue;
          if (!cell.classList.contains("hscredit-attribute-json")) cell.classList.add("hscredit-attribute-json");
          if (cell.firstChild.data !== formatted) cell.firstChild.data = formatted;
        }
      }
    }
  }

  window.__hscreditDashboard = { wrapPlotly, themeFigure, formatAttributeText };
  document.addEventListener("DOMContentLoaded", () => {
    isDark();
    const root = document.getElementById("dashboard");
    if (!root) return;
    formatAttributeTables(root);
    let pending = false;
    new MutationObserver(records => {
      isDark();
      if (!pending && records.some(record => record.type === "childList" || record.type === "characterData")) {
        pending = true;
        requestAnimationFrame(() => { pending = false; formatAttributeTables(root); });
      }
    }).observe(root, { childList: true, characterData: true, attributes: true, attributeFilter: ["class", "style"], subtree: true });
  });
})();
