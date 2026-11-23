const QUESTIONS = [
  ["worked", "What worked well?"],
  ["unexpected", "What didn't turn out as expected?"],
  ["why", "Why do you think the results might not be great?"],
  ["improve", "How could the results be improved?"],
];

const tip = document.getElementById("tip");
const chapters = [];

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function fmt(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return value == null ? "—" : String(value);
  }
  if (Number.isInteger(value)) return String(value);
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 1 : abs >= 10 ? 2 : abs >= 1 ? 3 : 4;
  return value.toFixed(digits).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

function fill(template, row) {
  return template.replace(/\{(\w+)\}/g, (_, key) => fmt(row[key]));
}

function activate(article, stop, event) {
  if (!stop) return;
  const summary = article.querySelector("[data-summary]");
  const summaryChanged = summary._raw !== stop.summary;
  if (summaryChanged) {
    summary._raw = stop.summary;
    summary.textContent = stop.summary;
  }
  if (stop.frame) {
    const img = article.querySelector(".live img");
    const cap = article.querySelector(".live figcaption");
    if (img) {
      img.src = stop.frame;
      if (stop.frameCaption) img.alt = stop.frameCaption;
    }
    if (cap && stop.frameCaption) cap.textContent = stop.frameCaption;
    article.querySelectorAll(".live .switcher button").forEach((button) => {
      button.classList.toggle("is-on", button.dataset.src === stop.frame);
    });
  }
  article.querySelectorAll(".reflection li").forEach((item) => {
    item.classList.toggle("is-hot", item.dataset.q === stop.answers);
  });
  if (event) showTip(event, stop.hover || stop.summary);
  if (summaryChanged) typeset(article.querySelector(".finding"));
}

function showTip(event, text) {
  tip.hidden = false;
  tip.textContent = text;
  const pad = 14;
  const rect = tip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}

function hideTip() {
  tip.hidden = true;
}

let mathChain = Promise.resolve();

function typeset(node) {
  if (!node) return;
  const run = () => {
    if (!window.MathJax || !MathJax.typesetPromise) return;
    mathChain = mathChain
      .then(() => {
        MathJax.typesetClear([node]);
        return MathJax.typesetPromise([node]);
      })
      .catch((error) => console.error(error));
  };
  if (window.MathJax && MathJax.startup && MathJax.startup.promise) {
    MathJax.startup.promise.then(run);
    return;
  }
  const waits = (node._mathWait || 0) + 1;
  node._mathWait = waits;
  if (waits < 80) window.setTimeout(typeset, 60, node);
}

function size(host) {
  const width = host.clientWidth || 640;
  return {
    width,
    margin: { top: 16, right: 18, bottom: 36, left: 52 },
  };
}

function focusMarks(svg, datum) {
  svg.selectAll(".mark").classed("is-on", (d) => d === datum).classed("dim", (d) => datum && d !== datum);
}

function bindMark(selection, article, svg, toStop) {
  selection
    .attr("tabindex", 0)
    .attr("role", "button")
    .on("pointerenter pointermove click", (event, d) => {
      focusMarks(svg, d);
      activate(article, toStop(d), event);
    })
    .on("focus", (event, d) => {
      focusMarks(svg, d);
      activate(article, toStop(d));
    });
}

function drawLine(host, chart, article) {
  const { width, margin } = size(host);
  const height = 280;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const marks = chart.marks;
  const xValues = marks.map((d) => d[chart.xKey]);
  let x;
  if (chart.xScale === "log") {
    const lo = Math.min(...xValues);
    const hi = Math.max(...xValues);
    const logLo = Math.log(lo);
    const logHi = Math.log(hi);
    const pad = (logHi - logLo) * 0.14 || Math.log(1.6);
    x = d3.scaleLog().domain([Math.exp(logLo - pad), Math.exp(logHi + pad)]).range([0, innerW]);
  } else {
    x = d3.scaleLinear().domain(d3.extent(xValues)).range([0, innerW]);
  }
  const yValues = marks.map((d) => d[chart.yKey]);
  const yMax = Math.max(...yValues);
  const yMin = Math.min(...yValues);
  const span = yMax - yMin || Math.abs(yMax) || 1;
  const y = d3.scaleLinear()
    .domain(chart.yFromZero === false ? [yMin - span * 0.45, yMax + span * 0.55] : [0, yMax * 1.12])
    .range([innerH, 0]);
  const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  svg.append("title").text(chart.title);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(4).tickSize(-innerW).tickFormat(""));
  g.append("path")
    .datum(marks)
    .attr("fill", "none")
    .attr("stroke", "var(--accent)")
    .attr("stroke-width", 2)
    .attr("d", d3.line().x((d) => x(d[chart.xKey])).y((d) => y(d[chart.yKey])));
  const axis = d3.axisBottom(x);
  if (chart.xScale === "log") axis.tickValues(xValues).tickFormat((d) => fmt(d));
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(axis);
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
  g.append("text").attr("x", innerW / 2).attr("y", innerH + 32).attr("text-anchor", "middle").attr("class", "row-label").text(chart.xLabel);
  g.append("text").attr("x", 0).attr("y", -4).attr("class", "row-label").text(chart.yLabel);
  const circles = g.selectAll("circle").data(marks).join("circle")
    .attr("class", "mark")
    .classed("is-best", (d) => !!d.selected)
    .attr("cx", (d) => x(d[chart.xKey]))
    .attr("cy", (d) => y(d[chart.yKey]))
    .attr("r", (d) => (d.selected ? 10 : 7))
    .attr("fill", "var(--accent)")
    .attr("stroke", (d) => (d.selected ? "#1c1915" : "#fffdf8"))
    .attr("stroke-width", (d) => (d.selected ? 3 : 2));
  g.selectAll("text.best-tag").data(marks.filter((d) => d.selected)).join("text")
    .attr("class", "best-tag")
    .attr("x", (d) => x(d[chart.xKey]))
    .attr("y", (d) => Math.max(12, y(d[chart.yKey]) - 14))
    .attr("text-anchor", "middle")
    .text((d) => d.bestLabel || "best");
  bindMark(circles, article, svg, (d) => d);
  marks.forEach((mark) => article._stops.push(mark));
}

function drawPanels(host, chart, article) {
  host.classList.add("panels");
  const wrap = d3.select(host);
  const width = Math.max(160, host.clientWidth / chart.panels.length);
  chart.panels.forEach((panel, panelIndex) => {
    const height = 220;
    const margin = { top: 22, right: 12, bottom: 32, left: 44 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const x = d3.scalePoint().domain(chart.marks.map((d) => d[chart.xKey])).range([0, innerW]).padding(0.4);
    const values = chart.marks.map((d) => d[panel.yKey]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const pad = (max - min) * 0.25 || Math.abs(max) * 0.08 || 0.5;
    const y = d3.scaleLinear().domain([min - pad, max + pad]).range([innerH, 0]);
    const svg = wrap.append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .style("width", `${100 / chart.panels.length}%`)
      .attr("role", "img");
    svg.append("title").text(panel.label);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    g.append("text").attr("x", 0).attr("y", -8).attr("class", "row-label").text(panel.label);
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(3).tickSize(-innerW).tickFormat(""));
    g.append("path")
      .datum(chart.marks)
      .attr("fill", "none")
      .attr("stroke", "var(--accent)")
      .attr("stroke-width", 2)
      .attr("d", d3.line().x((d) => x(d[chart.xKey])).y((d) => y(d[panel.yKey])));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(3).tickFormat((d) => fmt(d)));
    const dots = g.selectAll("circle").data(chart.marks).join("circle")
      .attr("class", "mark")
      .classed("is-best", (d) => !!d.selected)
      .attr("cx", (d) => x(d[chart.xKey]))
      .attr("cy", (d) => y(d[panel.yKey]))
      .attr("r", (d) => (d.selected ? 10 : 7))
      .attr("fill", "var(--accent)")
      .attr("stroke", (d) => (d.selected ? "#1c1915" : "#fffdf8"))
      .attr("stroke-width", (d) => (d.selected ? 3 : 2));
    if (panelIndex === 0) {
      g.selectAll("text.best-tag").data(chart.marks.filter((d) => d.selected)).join("text")
        .attr("class", "best-tag")
        .attr("x", (d) => x(d[chart.xKey]))
        .attr("y", (d) => Math.max(11, y(d[panel.yKey]) - 12))
        .attr("text-anchor", "middle")
        .text((d) => d.bestLabel || "best");
    }
    bindMark(dots, article, svg, (d) => d);
  });
  chart.marks.forEach((mark) => article._stops.push(mark));
}

function drawHbar(host, chart, article) {
  const { width } = size(host);
  const rowH = 46;
  const margin = { top: 16, right: 56, bottom: chart.guide ? 28 : 12, left: 128 };
  const height = margin.top + margin.bottom + chart.marks.length * rowH;
  const innerW = width - margin.left - margin.right;
  const max = Math.max(...chart.marks.map((d) => d.value), chart.guide ? chart.guide.value : 0);
  const x = d3.scaleLinear().domain([0, max * 1.08]).range([0, innerW]);
  const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  svg.append("title").text(chart.title);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  if (chart.guide) {
    g.append("line")
      .attr("x1", x(chart.guide.value)).attr("x2", x(chart.guide.value))
      .attr("y1", -4).attr("y2", chart.marks.length * rowH - 8)
      .attr("stroke", "var(--accent)").attr("stroke-dasharray", "3 3");
    g.append("text").attr("class", "guide-label").attr("x", x(chart.guide.value) + 4).attr("y", chart.marks.length * rowH + 8).text(chart.guide.label);
  }
  const rows = g.selectAll("g").data(chart.marks).join("g").attr("transform", (_, i) => `translate(0,${i * rowH})`);
  rows.append("text").attr("class", "row-label").attr("x", -8).attr("y", 18).attr("text-anchor", "end")
    .attr("font-weight", (d) => (d.selected ? 700 : 400))
    .text((d) => d.label);
  rows.append("rect")
    .attr("class", "mark")
    .classed("is-best", (d) => !!d.selected)
    .attr("y", 4)
    .attr("height", 22)
    .attr("width", (d) => x(d.value))
    .attr("rx", 4)
    .attr("fill", "var(--accent)")
    .attr("stroke", (d) => (d.selected ? "#1c1915" : "none"))
    .attr("stroke-width", (d) => (d.selected ? 2.5 : 0));
  rows.append("text").attr("class", "row-value").attr("x", (d) => x(d.value) + 6).attr("y", 20)
    .attr("font-weight", (d) => (d.selected ? 700 : 400))
    .text((d) => fmt(d.value));
  bindMark(rows.select("rect"), article, svg, (d) => d);
  chart.marks.forEach((mark) => article._stops.push(mark));
}

function drawRows(host, chart, article) {
  const { width } = size(host);
  const rowH = 54;
  const height = 12 + chart.marks.length * rowH;
  const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  svg.append("title").text(chart.title);
  const g = svg.append("g");
  const rows = g.selectAll("g").data(chart.marks).join("g").attr("transform", (_, i) => `translate(0,${8 + i * rowH})`);
  rows.append("rect")
    .attr("class", "mark")
    .attr("x", 8).attr("y", 0)
    .attr("width", width - 16).attr("height", rowH - 8)
    .attr("rx", 8)
    .attr("fill", "#f7f3ea");
  rows.append("circle")
    .attr("cx", 26).attr("cy", (rowH - 8) / 2).attr("r", 6)
    .attr("fill", (d) => (d.state === "missing" ? "#a39b8d" : "var(--accent)"));
  rows.append("text").attr("class", "row-label").attr("x", 42).attr("y", 20).attr("fill", "#1c1915")
    .attr("font-weight", (d) => (d.selected ? 700 : 400))
    .text((d) => d.label);
  rows.append("text").attr("class", "row-value").attr("x", 42).attr("y", 36)
    .attr("font-weight", (d) => (d.selected ? 700 : 400))
    .text((d) => d.valueText);
  rows.selectAll("text, circle").attr("pointer-events", "none");
  bindMark(rows.select("rect"), article, svg, (d) => d);
  chart.marks.forEach((mark) => article._stops.push(mark));
}

function nearestFrame(chart, dataset, epoch) {
  const table = chart.frames && chart.frames[dataset];
  if (!table) return null;
  const epochs = Object.keys(table).map(Number);
  const nearest = epochs.reduce((best, item) => (Math.abs(item - epoch) < Math.abs(best - epoch) ? item : best));
  return { src: table[String(nearest)], epoch: nearest };
}

function seriesStop(chart, dataset, row) {
  const beat = (chart.beats || []).find((item) => item.dataset === dataset && item.epoch === row.epoch);
  const frame = nearestFrame(chart, dataset, row.epoch);
  const summary = beat ? fill(beat.summary, row) : fill(chart.fallback, row);
  const note = frame && frame.epoch !== row.epoch ? ` The nearest saved grid is epoch ${frame.epoch}.` : "";
  return {
    answers: beat ? beat.answers : chart.fallbackAnswers,
    summary: summary + note,
    hover: [
      `${dataset}, epoch ${row.epoch}`,
      row.eval_g_bce == null
        ? `Held-out discriminator ${fmt(row.eval_d)}, generator ${fmt(row.eval_g)}`
        : `Held-out discriminator ${fmt(row.eval_d)}, generator ${fmt(row.eval_g)}, generator cross-entropy ${fmt(row.eval_g_bce)}`,
      `Training discriminator ${fmt(row.train_d)}, generator ${fmt(row.train_g)}`,
      `Logged sample standard deviation ${fmt(row.sample_std)}`,
    ].join("\n"),
    frame: frame && frame.src,
    frameCaption: frame
      ? `${dataset}, epoch ${row.epoch}. Held-out discriminator loss ${fmt(row.eval_d)}, generator loss ${fmt(row.eval_g)}.${frame.epoch === row.epoch ? "" : ` The nearest saved grid is epoch ${frame.epoch}.`}`
      : "",
  };
}

function drawSeries(host, chart, history, article) {
  const head = d3.select(host).append("div").attr("class", "chart-head");
  head.append("h3").text(chart.title);
  const switcher = head.append("div").attr("class", "switcher");
  const plot = d3.select(host).append("div").attr("class", "plot");
  const state = { dataset: host.dataset.current || chart.datasets[0].id };

  function render() {
    plot.selectAll("*").remove();
    const rows = history[state.dataset];
    const { width } = size(host);
    const height = 280;
    const margin = { top: 12, right: 16, bottom: 32, left: 44 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const x = d3.scaleLinear().domain([1, rows.length]).range([0, innerW]);
    const yMax = d3.max(rows, (d) => Math.max(d.eval_d, d.eval_g, d.eval_g_bce == null ? 0 : d.eval_g_bce));
    const y = d3.scaleLinear().domain([0, yMax * 1.08]).range([innerH, 0]);
    const svg = plot.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(4).tickSize(-innerW).tickFormat(""));
    const keys = [
      { key: "eval_d", color: "#1c1915", label: "D loss" },
      { key: "eval_g", color: "var(--accent)", label: "G loss" },
    ];
    if (rows.some((row) => row.eval_g_bce != null)) {
      keys.push({ key: "eval_g_bce", color: "#8a5a2a", label: "G BCE" });
    }
    keys.forEach((series) => {
      g.append("path")
        .datum(rows)
        .attr("fill", "none")
        .attr("stroke", series.color)
        .attr("stroke-width", 2)
        .attr("d", d3.line().x((d) => x(d.epoch)).y((d) => y(d[series.key])));
    });
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(8).tickFormat(d3.format("d")));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    const legend = g.append("g").attr("transform", `translate(${Math.max(0, innerW - keys.length * 88)},0)`);
    keys.forEach((series, index) => {
      legend.append("rect").attr("x", index * 88).attr("y", 0).attr("width", 12).attr("height", 4).attr("fill", series.color);
      legend.append("text").attr("class", "row-label").attr("x", index * 88 + 16).attr("y", 6).text(series.label);
    });
    const hits = g.selectAll("circle").data(rows).join("circle")
      .attr("class", "mark")
      .attr("cx", (d) => x(d.epoch))
      .attr("cy", (d) => y(d.eval_g))
      .attr("r", 4.5)
      .attr("fill", "var(--accent)")
      .attr("stroke", "#fffdf8");
    bindMark(hits, article, svg, (row) => seriesStop(chart, state.dataset, row));
    host.__focus = (epoch) => {
      const row = rows.find((item) => item.epoch === epoch);
      if (!row) return;
      focusMarks(svg, row);
      hits.filter((item) => item.epoch === epoch).attr("fill-opacity", 1).attr("r", 6);
    };
  }

  if (chart.datasets.length < 2) switcher.remove();
  switcher.selectAll("button").data(chart.datasets).join("button")
    .attr("type", "button")
    .classed("is-on", (d) => d.id === state.dataset)
    .text((d) => d.label)
    .on("click", (event, d) => {
      state.dataset = d.id;
      host.dataset.current = d.id;
      switcher.selectAll("button").classed("is-on", (item) => item.id === d.id);
      render();
      const rows = history[d.id];
      activate(article, seriesStop(chart, d.id, rows[rows.length - 1]), event);
      host.__focus(rows.length);
    });

  host.dataset.current = state.dataset;
  render();
  host.__showEpoch = (dataset, epoch) => {
    if (state.dataset !== dataset) {
      state.dataset = dataset;
      host.dataset.current = dataset;
      switcher.selectAll("button").classed("is-on", (item) => item.id === dataset);
      render();
    }
    if (host.__focus) host.__focus(epoch);
  };
  (chart.beats || []).forEach((beat) => {
    const row = history[beat.dataset].find((item) => item.epoch === beat.epoch);
    if (!row) return;
    article._stops.push({
      ...seriesStop(chart, beat.dataset, row),
      epoch: beat.epoch,
      apply() {
        state.dataset = beat.dataset;
        host.dataset.current = beat.dataset;
        switcher.selectAll("button").classed("is-on", (item) => item.id === beat.dataset);
        render();
        host.__focus(beat.epoch);
      },
    });
  });
}

function drawChart(host, chart, history, article) {
  if (chart.type === "line") drawLine(host, chart, article);
  else if (chart.type === "panels") drawPanels(host, chart, article);
  else if (chart.type === "hbar") drawHbar(host, chart, article);
  else if (chart.type === "rows") drawRows(host, chart, article);
  else if (chart.type === "series") drawSeries(host, chart, history, article);
}

function setPlaying(button, playing) {
  if (!button) return;
  const playText = button.dataset.play || "Play";
  const pauseText = button.dataset.pause || "Pause";
  button.textContent = playing ? pauseText : playText;
  button.classList.toggle("is-on", playing);
  button.setAttribute("aria-pressed", playing ? "true" : "false");
  button.setAttribute("aria-label", playing
    ? (button.dataset.pauseName || "Pause this chart")
    : (button.dataset.playName || "Play this chart"));
}

function playSequence(button, frames, show) {
  if (!frames.length) return;
  if (button._timer) {
    window.clearInterval(button._timer);
    button._timer = 0;
    setPlaying(button, false);
    return;
  }
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    show(frames[0]);
    return;
  }
  let index = button._index || 0;
  if (index >= frames.length) index = 0;
  const step = () => {
    show(frames[index]);
    index += 1;
    button._index = index;
    if (index >= frames.length) {
      window.clearInterval(button._timer);
      button._timer = 0;
      setPlaying(button, false);
    }
  };
  step();
  if (index < frames.length) {
    setPlaying(button, true);
    button._timer = window.setInterval(step, 700);
  }
}

function addEpochPlay(parent, label, frames, show) {
  const button = el("button", "play", label.play);
  button.type = "button";
  button.dataset.play = label.play;
  button.dataset.pause = "Pause";
  button.dataset.playName = label.playName;
  button.dataset.pauseName = label.pauseName;
  button.setAttribute("aria-pressed", "false");
  button.setAttribute("aria-label", label.playName);
  parent.append(button);
  button.addEventListener("click", () => playSequence(button, frames, show));
}

function epochFrames(chart, dataset, history) {
  const table = chart.frames[dataset];
  const rows = history[dataset] || [];
  return Object.keys(table).map(Number).sort((a, b) => a - b).map((epoch) => ({
    dataset,
    epoch,
    src: table[String(epoch)],
    row: rows.find((item) => item.epoch === epoch),
  }));
}

function showStop(host, article, stop) {
  if (stop.apply) stop.apply();
  activate(article, stop);
  host.querySelectorAll(".mark").forEach((node) => {
    const data = node.__data__;
    const on = data === stop || (data && stop.epoch != null && data.epoch === stop.epoch);
    node.classList.toggle("is-on", on);
    node.classList.toggle("dim", !on);
  });
}

function playChart(host, article) {
  const stops = host._stops || [];
  const button = host.querySelector(":scope > .play");
  if (!stops.length) return;
  if (host._timer) {
    window.clearInterval(host._timer);
    host._timer = 0;
    setPlaying(button, false);
    return;
  }
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    showStop(host, article, stops[0]);
    return;
  }
  let index = host._index || 0;
  if (index >= stops.length) index = 0;
  const step = () => {
    showStop(host, article, stops[index]);
    index += 1;
    host._index = index;
    if (index >= stops.length) {
      window.clearInterval(host._timer);
      host._timer = 0;
      setPlaying(button, false);
    }
  };
  step();
  if (index < stops.length) {
    setPlaying(button, true);
    host._timer = window.setInterval(step, 1400);
  }
}

function drawCharts(article, bundle, history) {
  article.querySelectorAll(".chart").forEach((host, index) => {
    window.clearInterval(host._timer);
    host._timer = 0;
    host._index = 0;
    host._stops = [];
    // Drawers append to article._stops. Point that list at this chart.
    article._stops = host._stops;
    const chart = bundle.charts[index];
    const play = host.querySelector(":scope > .play");
    if (play) {
      play.remove();
      setPlaying(play, false);
    }
    if (chart.type === "series") {
      const current = host.dataset.current;
      host.replaceChildren();
      if (current) host.dataset.current = current;
      drawChart(host, chart, history, article);
    } else {
      const plot = host.querySelector(":scope > .plot");
      plot.replaceChildren();
      drawChart(plot, chart, history, article);
    }
    if (play) {
      play.hidden = !host._stops.length;
      host.append(play);
    }
  });
}

function renderChapter(bundle, history) {
  const article = el("article", "chapter");
  article.id = bundle.id;
  article.style.setProperty("--accent", bundle.accent);
  const head = el("div", "chapter-head");
  const titles = el("div");
  titles.append(el("p", "points", bundle.points), el("h2", null, bundle.title), el("p", "question", bundle.question));
  head.append(titles);
  article.append(head);
  const prose = Array.isArray(bundle.process) ? bundle.process : [bundle.process];
  prose.forEach((block) => article.append(el("p", "process", block)));

  const stage = el("div", "stage");
  const charts = el("div", "charts");
  bundle.charts.forEach((chart) => {
    const host = el("div", "chart");
    if (chart.type !== "series") {
      host.append(el("h3", null, chart.title));
      host.append(el("div", "plot"));
    }
    if (bundle.play !== false) {
      const play = el("button", "play", "Play");
      play.type = "button";
      play.setAttribute("aria-pressed", "false");
      play.setAttribute("aria-label", "Play this chart");
      host.append(play);
      play.addEventListener("click", () => playChart(host, article));
    }
    charts.append(host);
  });
  const finding = el("aside", "finding");
  finding.append(el("p", "summary"));
  finding.querySelector(".summary").dataset.summary = "";
  stage.append(charts, finding);
  article.append(stage);

  const reflection = el("ul", "reflection");
  QUESTIONS.forEach(([key, label]) => {
    const item = el("li");
    item.dataset.q = label;
    item.append(el("h3", null, label), el("p", null, bundle.reflection[key]));
    reflection.append(item);
  });
  article.append(reflection);

  const figures = el("div", "figures");
  if (bundle.live) {
    const fig = el("figure", "live");
    const img = el("img");
    img.src = bundle.live.src;
    img.alt = bundle.live.alt;
    const cap = el("figcaption", null, bundle.live.caption);
    fig.append(img);
    const views = bundle.live.views || [];
    if (views.length > 1) {
      const switcher = el("div", "switcher");
      switcher.setAttribute("role", "group");
      switcher.setAttribute("aria-label", "Sample grid");
      views.forEach((view) => {
        const button = el("button", null, view.label);
        button.type = "button";
        button.dataset.src = view.src;
        button.classList.toggle("is-on", view.src === bundle.live.src);
        button.addEventListener("click", () => {
          const mark = (bundle.charts || []).flatMap((chart) => chart.marks || []).find((item) => item.frame === view.src);
          if (mark) {
            activate(article, mark);
            return;
          }
          img.src = view.src;
          img.alt = view.alt;
          cap.textContent = view.caption;
          switcher.querySelectorAll("button").forEach((item) => {
            item.classList.toggle("is-on", item === button);
          });
        });
        switcher.append(button);
      });
      fig.append(switcher);
    }
    fig.append(cap);
    const plays = el("div", "epoch-plays");
    const sequences = [];
    if (bundle.live.sequence && bundle.live.sequence.length) {
      sequences.push({
        play: "Play epochs",
        playName: "Play epochs",
        pauseName: "Pause epochs",
        frames: bundle.live.sequence,
      });
    }
    (bundle.live.sequences || []).forEach((item) => sequences.push(item));
    sequences.forEach((item) => {
      addEpochPlay(plays, {
        play: item.play,
        playName: item.playName,
        pauseName: item.pauseName,
      }, item.frames, (frame) => {
        img.src = frame.src;
        img.alt = frame.caption;
        cap.textContent = frame.caption;
      });
    });
    const series = bundle.charts.find((chart) => chart.frames);
    if (series) {
      series.datasets.forEach((dataset) => {
        const frames = epochFrames(series, dataset.id, history);
        addEpochPlay(plays, {
          play: `Play ${dataset.label}`,
          playName: `Play ${dataset.label} epochs`,
          pauseName: `Pause ${dataset.label} epochs`,
        }, frames, (frame) => {
          img.src = frame.src;
          const text = frame.row
            ? `${dataset.label}, epoch ${frame.epoch}. Held-out discriminator loss ${fmt(frame.row.eval_d)}, generator loss ${fmt(frame.row.eval_g)}.`
            : `${dataset.label}, epoch ${frame.epoch}`;
          img.alt = text;
          cap.textContent = text;
          const host = article.querySelector(".chart");
          if (host && host.__showEpoch) host.__showEpoch(frame.dataset, frame.epoch);
          if (frame.row) activate(article, seriesStop(series, frame.dataset, frame.row));
        });
      });
    }
    if (plays.childElementCount) fig.append(plays);
    figures.append(fig);
  }
  bundle.figures.forEach((figure) => {
    const fig = el("figure");
    const img = el("img");
    img.src = figure.src;
    img.alt = figure.alt;
    img.loading = "lazy";
    fig.append(img, el("figcaption", null, figure.caption));
    figures.append(fig);
  });
  article.append(figures);

  const sources = el("p", "sources");
  sources.append(document.createTextNode("Sources: "));
  bundle.sources.forEach((source, index) => {
    if (index) sources.append(document.createTextNode(" · "));
    sources.append(el("code", null, source));
  });
  article.append(sources);

  document.getElementById("chapters").append(article);
  activate(article, bundle.opening);
  drawCharts(article, bundle, history);
  typeset(article);
  chapters.push({ article, bundle, history });
}

function renderNav(bundles) {
  const toc = document.getElementById("toc");
  const jump = document.getElementById("jump");
  bundles.forEach((bundle) => {
    const item = el("li");
    const link = el("a");
    link.href = `#${bundle.id}`;
    const swatch = el("span", "swatch");
    swatch.style.setProperty("--swatch", bundle.accent);
    link.append(swatch, document.createTextNode(bundle.nav));
    item.append(link);
    toc.append(item);
    const jumpItem = el("li");
    const jumpLink = el("a", null, `${bundle.nav}. ${bundle.lede}`);
    jumpLink.href = `#${bundle.id}`;
    jumpItem.append(jumpLink);
    jump.append(jumpItem);
  });
}

async function boot() {
  if (!window.d3) {
    document.getElementById("chapters").append(el("p", null, "The charts could not be drawn."));
    return;
  }
  try {
    const manifest = await fetch("bundles/manifest.json").then((response) => response.json());
    const history = await fetch("bundles/gan-history.json").then((response) => response.json());
    const bundles = await Promise.all(manifest.map((item) => fetch(item.file).then((response) => response.json())));
    renderNav(bundles);
    bundles.forEach((bundle) => renderChapter(bundle, history));
  } catch (error) {
    document.getElementById("chapters").append(el(
      "p",
      null,
      "The chapter files did not load.",
    ));
    console.error(error);
  }
}

let resizeTimer = 0;
window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    chapters.forEach(({ article, bundle, history }) => drawCharts(article, bundle, history));
  }, 150);
});

window.addEventListener("scroll", hideTip, true);

document.addEventListener("pointermove", (event) => {
  if (tip.hidden) return;
  if (!event.target.closest || !event.target.closest(".mark")) hideTip();
});

boot();
