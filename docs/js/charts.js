const liveHosts = new Set();

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function fmt(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 1 : abs >= 10 ? 2 : abs >= 1 ? 3 : 4;
  return value.toFixed(digits).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

function track(host, draw) {
  host._draw = draw;
  liveHosts.add(host);
  draw();
}

function redrawCharts() {
  liveHosts.forEach((host) => {
    if (!host.isConnected) {
      liveHosts.delete(host);
      return;
    }
    host._draw();
  });
}

let resizeTimer = 0;
window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(redrawCharts, 150);
});

function switchBar(parent, label, options, current, onPick) {
  if (!options || options.length < 2) return null;
  const bar = el("div", "switcher");
  bar.setAttribute("role", "group");
  bar.setAttribute("aria-label", label);
  options.forEach((option) => {
    const button = el("button", option.id === current ? "is-on" : "", option.label);
    button.type = "button";
    button.dataset.id = option.id;
    button.addEventListener("click", () => {
      bar.querySelectorAll("button").forEach((item) => {
        item.classList.toggle("is-on", item === button);
      });
      onPick(option.id);
    });
    bar.append(button);
  });
  parent.append(bar);
  return bar;
}

function mountReel(host, frames) {
  let index = Math.max(0, frames.length - 1);
  let timer = 0;
  const img = el("img");
  img.alt = "";
  const ruler = el("div", "ruler");
  const play = el("button", "play", "Play epochs");
  play.type = "button";
  play.setAttribute("aria-pressed", "false");
  const cap = el("p", "cap", "");
  host.append(play, img, ruler, cap);

  function show() {
    const frame = frames[index];
    img.src = frame.src;
    img.alt = frame.caption;
    cap.textContent = frame.caption;
    drawRuler();
  }

  function stop() {
    window.clearInterval(timer);
    timer = 0;
    play.classList.remove("is-on");
    play.textContent = "Play epochs";
    play.setAttribute("aria-pressed", "false");
  }

  function drawRuler() {
    const width = Math.max(280, ruler.clientWidth || host.clientWidth || 640);
    const height = 36;
    const svg = d3.select(ruler).selectAll("svg").data([0]).join("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("role", "img");
    svg.selectAll("*").remove();
    const x = d3.scalePoint()
      .domain(frames.map((frame) => frame.epoch))
      .range([16, width - 16]);
    svg.append("line")
      .attr("x1", 12)
      .attr("x2", width - 12)
      .attr("y1", 18)
      .attr("y2", 18)
      .attr("stroke", "#ddd4c4");
    svg.selectAll("rect")
      .data(frames)
      .join("rect")
      .attr("x", (frame) => x(frame.epoch) - (frame === frames[index] ? 5 : 2))
      .attr("y", (frame) => (frame === frames[index] ? 6 : 13))
      .attr("width", (frame) => (frame === frames[index] ? 10 : 4))
      .attr("height", (frame) => (frame === frames[index] ? 22 : 10))
      .attr("rx", 1)
      .attr("fill", (frame) => (frame === frames[index] ? "#1c1915" : "var(--accent)"))
      .style("cursor", "pointer")
      .on("click", (_, frame) => {
        stop();
        index = frames.indexOf(frame);
        show();
      });
    svg.append("text")
      .attr("x", 12)
      .attr("y", 34)
      .attr("class", "tick-label")
      .text("epoch " + frames[index].epoch);
  }

  play.addEventListener("click", () => {
    if (timer) {
      stop();
      return;
    }
    index = 0;
    show();
    play.classList.add("is-on");
    play.textContent = "Pause";
    play.setAttribute("aria-pressed", "true");
    timer = window.setInterval(() => {
      if (index >= frames.length - 1) {
        stop();
        return;
      }
      index += 1;
      show();
    }, 650);
  });

  track(ruler, drawRuler);
  show();
  return stop;
}

function mountRibbon(host, rows, series, boldEpoch) {
  const plot = el("div", "plot");
  host.append(plot);
  track(plot, () => drawRibbon(plot, rows, series, boldEpoch));
}

function drawRibbon(host, rows, series, boldEpoch) {
  const width = Math.max(320, host.clientWidth || 640);
  const band = 78;
  const margin = { top: 8, right: 54, bottom: 28, left: 72 };
  const height = margin.top + margin.bottom + series.length * band;
  const innerW = width - margin.left - margin.right;
  const svg = d3.select(host).selectAll("svg").data([0]).join("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img");
  svg.selectAll("*").remove();
  const epochs = rows.map((row) => row.epoch);
  const x = d3.scaleLinear().domain(d3.extent(epochs)).range([0, innerW]);
  series.forEach((item, index) => {
    const values = rows.map((row) => row[item.key]);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const pad = (hi - lo) * 0.18 || Math.abs(hi) * 0.08 || 0.05;
    const y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([band - 22, 10]);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top + index * band})`);
    g.append("rect")
      .attr("x", 0)
      .attr("y", 4)
      .attr("width", innerW)
      .attr("height", band - 18)
      .attr("fill", index % 2 ? "#f7f3ea" : "#fffdf8");
    g.append("path")
      .datum(rows)
      .attr("fill", item.color || "var(--accent)")
      .attr("opacity", 0.28)
      .attr("d", d3.area()
        .x((row) => x(row.epoch))
        .y0(band - 22)
        .y1((row) => y(row[item.key])));
    g.append("path")
      .datum(rows)
      .attr("fill", "none")
      .attr("stroke", item.color || "var(--accent)")
      .attr("stroke-width", 1.6)
      .attr("d", d3.line().x((row) => x(row.epoch)).y((row) => y(row[item.key])));
    const marked = rows.find((row) => row.epoch === boldEpoch);
    if (marked) {
      const px = x(marked.epoch);
      const nearEnd = px > innerW * 0.62;
      g.append("rect")
        .attr("x", px - 4)
        .attr("y", 6)
        .attr("width", 8)
        .attr("height", band - 26)
        .attr("fill", "#1c1915");
      g.append("text")
        .attr("x", nearEnd ? px - 10 : px + 10)
        .attr("y", 18)
        .attr("text-anchor", nearEnd ? "end" : "start")
        .attr("class", "bold-value")
        .text(fmt(marked[item.key]));
    }
    g.append("text")
      .attr("x", -8)
      .attr("y", band / 2)
      .attr("text-anchor", "end")
      .attr("class", "tick-label")
      .text(item.label);
    if (index === series.length - 1) {
      g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${band - 16})`)
        .call(d3.axisBottom(x).ticks(Math.min(8, rows.length)).tickFormat(d3.format("d")));
    }
  });
}

function mountStems(host, rows) {
  const plot = el("div", "plot");
  host.append(plot);
  track(plot, () => drawStems(plot, rows));
}

function drawStems(host, rows) {
  const width = Math.max(280, host.clientWidth || 640);
  const height = 220;
  const margin = { top: 22, right: 12, bottom: 48, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const values = rows.map((row) => row.value);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo) * 0.35 || Math.abs(hi) * 0.08 || 1;
  const y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([innerH, 0]);
  const x = d3.scaleBand().domain(rows.map((row) => row.label)).range([0, innerW]).padding(0.38);
  const svg = d3.select(host).selectAll("svg").data([0]).join("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img");
  svg.selectAll("*").remove();
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat((d) => fmt(d)));
  g.append("line")
    .attr("x1", 0)
    .attr("x2", innerW)
    .attr("y1", y(lo - pad))
    .attr("y2", y(lo - pad))
    .attr("stroke", "#ddd4c4");
  rows.forEach((row) => {
    const cx = x(row.label) + x.bandwidth() / 2;
    const y0 = y(lo - pad);
    const y1 = y(row.value);
    g.append("line")
      .attr("x1", cx)
      .attr("x2", cx)
      .attr("y1", y0)
      .attr("y2", y1)
      .attr("stroke", row.kept ? "#1c1915" : "var(--accent)")
      .attr("stroke-width", row.kept ? 4 : 2);
    g.append("rect")
      .attr("x", cx - (row.kept ? 8 : 6))
      .attr("y", y1 - (row.kept ? 8 : 6))
      .attr("width", row.kept ? 16 : 12)
      .attr("height", row.kept ? 16 : 12)
      .attr("fill", row.kept ? "#1c1915" : "#fffdf8")
      .attr("stroke", row.kept ? "#1c1915" : "var(--accent)")
      .attr("stroke-width", 2);
    g.append("text")
      .attr("x", cx)
      .attr("y", Math.max(12, y1 - 12))
      .attr("text-anchor", "middle")
      .attr("class", row.kept ? "bold-value" : "tick-label")
      .text(fmt(row.value));
    g.append("text")
      .attr("x", cx)
      .attr("y", innerH + 18)
      .attr("text-anchor", "middle")
      .attr("class", "tick-label")
      .text(row.label);
    if (row.kept) {
      g.append("text")
        .attr("x", cx)
        .attr("y", innerH + 32)
        .attr("text-anchor", "middle")
        .attr("class", "bold-value")
        .text(row.tag || "kept");
    }
  });
}

function mountPairs(host, metrics) {
  host.classList.add("mini-grid");
  metrics.forEach((metric) => {
    const cell = el("div", "mini");
    cell.append(el("p", "mini-title", metric.label));
    const plot = el("div", "mini-plot");
    cell.append(plot);
    host.append(cell);
    track(plot, () => drawPair(plot, metric.bars));
  });
}

function drawPair(host, bars) {
  const width = Math.max(120, host.clientWidth || 160);
  const height = 132;
  const margin = { top: 16, right: 8, bottom: 28, left: 8 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const values = bars.map((bar) => bar.value);
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);
  const span = hi - lo || 1;
  const y = d3.scaleLinear()
    .domain([lo - span * 0.12, hi + span * 0.22])
    .range([innerH, 0]);
  const x = d3.scaleBand().domain(bars.map((bar) => bar.name)).range([0, innerW]).padding(0.32);
  const svg = d3.select(host).selectAll("svg").data([0]).join("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img");
  svg.selectAll("*").remove();
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  g.append("line")
    .attr("x1", 0)
    .attr("x2", innerW)
    .attr("y1", y(0))
    .attr("y2", y(0))
    .attr("stroke", "#1c1915")
    .attr("stroke-width", 1);
  bars.forEach((bar) => {
    const y0 = y(0);
    const y1 = y(bar.value);
    const top = Math.min(y0, y1);
    const h = Math.max(2, Math.abs(y1 - y0));
    g.append("rect")
      .attr("x", x(bar.name))
      .attr("y", top)
      .attr("width", x.bandwidth())
      .attr("height", h)
      .attr("fill", bar.name === "Train" ? "#1c1915" : "var(--accent)");
    g.append("text")
      .attr("x", x(bar.name) + x.bandwidth() / 2)
      .attr("y", bar.value >= 0 ? top - 4 : top + h + 12)
      .attr("text-anchor", "middle")
      .attr("class", "tick-label")
      .text(fmt(bar.value));
    g.append("text")
      .attr("x", x(bar.name) + x.bandwidth() / 2)
      .attr("y", innerH + 16)
      .attr("text-anchor", "middle")
      .attr("class", "tick-label")
      .text(bar.name);
  });
}

function mountHeat(host, columns, matrix) {
  const plot = el("div", "plot");
  host.append(plot);
  track(plot, () => drawHeat(plot, columns, matrix));
}

function drawHeat(host, columns, matrix) {
  const width = Math.max(320, host.clientWidth || 640);
  const rowH = 28;
  const margin = { top: 26, right: 12, bottom: 8, left: 36 };
  const height = margin.top + margin.bottom + matrix.length * rowH;
  const innerW = width - margin.left - margin.right;
  const flat = matrix.flat();
  const color = d3.scaleLinear()
    .domain([Math.min(...flat), Math.max(...flat)])
    .range(["#f6f1e6", "#1c1915"]);
  const x = d3.scaleBand().domain(columns).range([0, innerW]).padding(0.08);
  const svg = d3.select(host).selectAll("svg").data([0]).join("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "img");
  svg.selectAll("*").remove();
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  columns.forEach((column) => {
    g.append("text")
      .attr("x", x(column) + x.bandwidth() / 2)
      .attr("y", -8)
      .attr("text-anchor", "middle")
      .attr("class", "tick-label")
      .text("β " + column);
  });
  matrix.forEach((row, rowIndex) => {
    g.append("text")
      .attr("x", -8)
      .attr("y", rowIndex * rowH + rowH / 2 + 4)
      .attr("text-anchor", "end")
      .attr("class", "tick-label")
      .text("z" + (rowIndex + 1));
    row.forEach((value, columnIndex) => {
      const column = columns[columnIndex];
      g.append("rect")
        .attr("x", x(column))
        .attr("y", rowIndex * rowH)
        .attr("width", x.bandwidth())
        .attr("height", rowH - 4)
        .attr("rx", 3)
        .attr("fill", color(value));
      g.append("text")
        .attr("x", x(column) + x.bandwidth() / 2)
        .attr("y", rowIndex * rowH + (rowH - 4) / 2 + 4)
        .attr("text-anchor", "middle")
        .attr("fill", value > d3.mean(flat) ? "#f6f1e6" : "#1c1915")
        .attr("class", "tick-label")
        .text(fmt(value));
    });
  });
}

function still(parent, src, alt, caption) {
  const fig = el("figure", "still");
  const img = el("img");
  img.src = src;
  img.alt = alt;
  fig.append(img, el("figcaption", "cap", caption));
  parent.append(fig);
}

function caption(parent, text) {
  parent.append(el("p", "cap", text));
}

function linkPara(parts) {
  const p = el("p");
  parts.forEach((part) => {
    if (typeof part === "string") {
      p.append(document.createTextNode(part));
      return;
    }
    const a = el("a", "plot-link", part.label);
    a.href = part.href;
    p.append(a);
  });
  return p;
}
