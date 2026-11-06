(() => {
  "use strict";
  const view = document.querySelector("#feature-report");
  const report = JSON.parse(document.querySelector("#feature-report-data").textContent);
  const {rows, headers, formats} = report;
  const table = view.querySelector("table");
  const count = view.querySelector(".report-status");
  const search = view.querySelector("input");
  const prev = view.querySelector(".report-prev");
  const next = view.querySelector(".report-next");
  const sizeSelect = view.querySelector(".report-size");
  const pageLinks = view.querySelector(".report-pages");
  let pageSize = Number(sizeSelect.value);
  let matches = rows.slice(), page = 0, sortCol = -1, direction = 1;

  // 比例、计数和小数的展示精度沿用工作簿，排序始终使用原始值。
  const formatters = formats.map(code => {
    const section = code.split(";")[0];
    const fraction = section.split(".")[1]?.match(/^[0#]+/)?.[0] || "";
    return new Intl.NumberFormat("zh-CN", {
      style: section.includes("%") ? "percent" : "decimal",
      useGrouping: section.includes(","),
      minimumFractionDigits: (fraction.match(/0/g) || []).length,
      maximumFractionDigits: code === "General" ? 15 : fraction.length
    });
  });
  function line(values) {
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 100 20");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "最小值、各分位点和最大值");
    const numbers = values.filter(Number.isFinite);
    if (!numbers.length) return svg;
    const low = Math.min(...numbers), span = Math.max(...numbers) - low;
    const path = document.createElementNS(svgNS, "path");
    let connected = false, d = "";
    values.forEach((v, i) => {
      if (!Number.isFinite(v)) { connected = false; return; }
      const x = 2 + 96 * i / Math.max(1, values.length - 1);
      const y = span ? 18 - 16 * (v - low) / span : 10;
      d += (connected ? "L" : "M") + x + " " + y;
      connected = true;
    });
    path.setAttribute("d", d);
    svg.append(path);
    return svg;
  }
  function draw() {
    const start = page * pageSize, visible = matches.slice(start, start + pageSize);
    const fragment = document.createDocumentFragment();
    for (const values of visible) {
      const tr = document.createElement("tr");
      values.forEach((v, i) => {
        const td = tr.insertCell();
        if (Array.isArray(v)) td.append(line(v));
        else if (typeof v === "number") td.textContent = v === 0 && formats[i].endsWith(";0") ? "0" : formatters[i].format(v);
        else td.textContent = v ?? "—";
      });
      fragment.append(tr);
    }
    if (!visible.length) {
      const tr = document.createElement("tr");
      const empty = tr.insertCell();
      empty.colSpan = headers.length;
      empty.className = "report-empty";
      empty.textContent = "没有匹配的特征";
      fragment.append(tr);
    }
    table.tBodies[0].replaceChildren(fragment);
    count.textContent = `共 ${matches.length.toLocaleString("zh-CN")} 条`;
    drawPageLinks();
    prev.disabled = !page;
    next.disabled = start + pageSize >= matches.length;
    view.querySelector(".report-grid").scrollTop = 0;
  }
  function drawPageLinks() {
    const total = Math.ceil(matches.length / pageSize);
    const first = Math.max(1, Math.min(page + 1 - 2, total - 4));
    const fragment = document.createDocumentFragment();
    let gap = false;
    for (let n = 1; n <= total; n++) {
      if (n !== 1 && n !== total && (n < first || n >= first + 5)) {
        if (!gap) {
          const dots = document.createElement("span");
          dots.textContent = "…";
          dots.setAttribute("aria-hidden", "true");
          fragment.append(dots);
        }
        gap = true;
        continue;
      }
      gap = false;
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.page = n;
      button.textContent = n;
      button.setAttribute("aria-label", `第 ${n} 页`);
      if (n === page + 1) button.setAttribute("aria-current", "page");
      fragment.append(button);
    }
    pageLinks.replaceChildren(fragment);
  }
  function filterAndSort() {
    const term = search.value.trim().toLowerCase();
    matches = rows.filter(r => String(r[0]).toLowerCase().includes(term));
    if (sortCol >= 0) matches.sort((a, b) => {
      const x = a[sortCol], y = b[sortCol];
      if (x == null || y == null) return Number(x == null) - Number(y == null);
      return direction * (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), "zh-CN", {numeric: true}));
    });
    page = 0;
    draw();
  }
  for (const note of report.notes) {
    const li = document.createElement("li");
    li.textContent = note;
    view.querySelector("details ul").append(li);
  }
  const bands = table.tHead.insertRow(), labels = table.tHead.insertRow();
  for (const [name, width] of report.groups) {
    const th = document.createElement("th");
    th.colSpan = width; th.scope = "colgroup"; th.textContent = name;
    bands.append(th);
  }
  headers.forEach((name, i) => {
    const th = document.createElement("th"); th.scope = "col"; labels.append(th);
    if (name === "数据分布") { th.textContent = name; return; }
    const button = document.createElement("button");
    button.type = "button"; button.textContent = name; button.title = "按" + name + "排序"; th.append(button);
    button.addEventListener("click", () => {
      if (sortCol >= 0) labels.cells[sortCol].removeAttribute("aria-sort");
      direction = sortCol === i ? -direction : (rows.some(r => typeof r[i] === "number") ? -1 : 1);
      sortCol = i;
      th.setAttribute("aria-sort", direction === 1 ? "ascending" : "descending");
      filterAndSort();
    });
  });
  search.addEventListener("input", filterAndSort);
  sizeSelect.addEventListener("change", () => {
    pageSize = Number(sizeSelect.value);
    page = 0;
    draw();
  });
  pageLinks.addEventListener("click", event => {
    const button = event.target.closest("button[data-page]");
    if (!button) return;
    page = Number(button.dataset.page) - 1;
    draw();
    pageLinks.querySelector('[aria-current="page"]').focus({preventScroll: true});
  });
  prev.addEventListener("click", () => { page--; draw(); });
  next.addEventListener("click", () => { page++; draw(); });
  draw();
})();
