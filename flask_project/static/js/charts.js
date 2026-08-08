// Chart builders for the EDA stages. Reads payloads from window.EDA
// (injected by the template) and renders into <canvas data-chart="...">.
// Requires Chart.js 4 (loaded by the template before this file).
//
// The model pages (train/evaluate) pass their payload via window.MODEL_PAGE
// instead; the builders are exposed as window.PPCharts so script.js can
// re-render them after an interactive benchmark run.
//
// Visual language: flat fills, hairline grids, mono ticks, fast settles.
// One accent color; red is reserved for "missing / not placed".

(function () {
  if (typeof Chart === "undefined") return;

  const EDA = window.EDA || {};

  const PALETTE = {
    text: "#EBECE8",
    text2: "#9BA29A",
    text3: "#8A9189",
    accent: "#D9A63F",
    accentFill: "rgba(217, 166, 63, 0.55)",
    slate: "#6E8FA0",
    slateFill: "rgba(110, 143, 160, 0.45)",
    neutral: "#8A9189",
    neutralFill: "rgba(138, 145, 137, 0.4)",
    danger: "#C65D55",
    dangerFill: "rgba(198, 93, 85, 0.55)",
    grid: "rgba(235, 236, 232, 0.06)",
    panel: "#1B1F1C",
    panelBorder: "#363D37",
  };

  // one fixed color per candidate, consistent across every chart and page
  const MODEL_COLORS = {
    logistic_regression: { stroke: PALETTE.neutral, fill: PALETTE.neutralFill },
    random_forest: { stroke: PALETTE.slate, fill: PALETTE.slateFill },
    gradient_boosting: { stroke: PALETTE.accent, fill: PALETTE.accentFill },
  };
  const FALLBACK_COLORS = [
    { stroke: PALETTE.accent, fill: PALETTE.accentFill },
    { stroke: PALETTE.slate, fill: PALETTE.slateFill },
    { stroke: PALETTE.text2, fill: PALETTE.neutralFill },
  ];

  function modelColor(model, index) {
    return MODEL_COLORS[model.key] || FALLBACK_COLORS[index % FALLBACK_COLORS.length];
  }

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  Chart.defaults.font.family = "'IBM Plex Mono', ui-monospace, monospace";
  Chart.defaults.font.size = 10;
  Chart.defaults.color = PALETTE.text3;
  Chart.defaults.borderColor = PALETTE.grid;
  Chart.defaults.animation.duration = prefersReducedMotion ? 0 : 450;
  Chart.defaults.animation.easing = "easeOutQuart";

  const tooltip = {
    backgroundColor: PALETTE.panel,
    borderColor: PALETTE.panelBorder,
    borderWidth: 1,
    titleColor: PALETTE.text,
    bodyColor: PALETTE.text2,
    padding: 10,
    displayColors: false,
    cornerRadius: 6,
  };

  const baseScales = {
    x: {
      grid: { display: false },
      border: { display: false },
      ticks: { maxTicksLimit: 7, maxRotation: 0, autoSkip: true },
    },
    y: {
      grid: { color: PALETTE.grid },
      border: { display: false },
      ticks: { maxTicksLimit: 6 },
    },
  };

  // histogram bars + smoothed overlay line (mirrors the notebook's KDE)
  function buildHistogram(canvas, payload, fill, stroke) {
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      data: {
        labels: payload.labels,
        datasets: [
          {
            type: "line",
            data: payload.smooth,
            borderColor: PALETTE.text,
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.45,
            fill: false,
          },
          {
            type: "bar",
            data: payload.counts,
            backgroundColor: fill,
            borderColor: stroke,
            borderWidth: 1,
            borderRadius: 1.5,
            barPercentage: 1.0,
            categoryPercentage: 0.92,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip },
        scales: baseScales,
      },
    });
  }

  function buildMissing(canvas, payload) {
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: payload.chart_labels,
        datasets: [
          {
            data: payload.chart_values,
            backgroundColor: PALETTE.dangerFill,
            borderColor: PALETTE.danger,
            borderWidth: 1,
            borderRadius: 3,
            maxBarThickness: 48,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip },
        scales: {
          ...baseScales,
          x: { ...baseScales.x, ticks: { maxRotation: 30, autoSkip: false } },
          y: {
            ...baseScales.y,
            title: { display: true, text: "missing cells", color: PALETTE.text3 },
          },
        },
      },
    });
  }

  function buildInfluence(canvas, payload, axisTitle) {
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: payload.labels,
        datasets: [
          {
            data: payload.values,
            backgroundColor: PALETTE.accentFill,
            borderColor: PALETTE.accent,
            borderWidth: 1,
            borderRadius: 2,
            maxBarThickness: 18,
          },
        ],
      },
      options: {
        indexAxis: "y",
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip },
        scales: {
          x: {
            grid: { color: PALETTE.grid },
            border: { display: false },
            beginAtZero: true,
            title: { display: true, text: axisTitle || "correlation with PlacementStatus", color: PALETTE.text3 },
          },
          y: { grid: { display: false }, border: { display: false }, ticks: { autoSkip: false } },
        },
      },
    });
  }

  function buildCategory(canvas, rows) {
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: rows.map((r) => r.label),
        datasets: [
          {
            data: rows.map((r) => r.rate),
            backgroundColor: PALETTE.accentFill,
            borderColor: PALETTE.accent,
            borderWidth: 1,
            borderRadius: 3,
            maxBarThickness: 40,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltip,
            callbacks: {
              label: (c) => ` ${c.parsed.y}% placed · n=${rows[c.dataIndex].count.toLocaleString()}`,
            },
          },
        },
        scales: {
          x: { grid: { display: false }, border: { display: false }, ticks: { maxRotation: 25, autoSkip: false } },
          y: {
            min: 0,
            max: 100,
            grid: { color: PALETTE.grid },
            border: { display: false },
            ticks: { callback: (v) => v + "%" },
          },
        },
      },
    });
  }

  function buildGender(canvas, payload) {
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: payload.labels,
        datasets: [
          {
            label: "Not placed",
            data: payload.not_placed,
            backgroundColor: PALETTE.dangerFill,
            borderColor: PALETTE.danger,
            borderWidth: 1,
            borderRadius: 3,
            maxBarThickness: 40,
          },
          {
            label: "Placed",
            data: payload.placed,
            backgroundColor: PALETTE.accentFill,
            borderColor: PALETTE.accent,
            borderWidth: 1,
            borderRadius: 3,
            maxBarThickness: 40,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { boxWidth: 9, boxHeight: 9, usePointStyle: true, pointStyle: "rectRounded" },
          },
          tooltip,
        },
        scales: baseScales,
      },
    });
  }

  // ROC curves, one line per trained model + a dashed chance diagonal
  function buildRoc(canvas, models) {
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    const ctx = canvas.getContext("2d");
    const datasets = models.map((m, i) => ({
      label: `${m.name} · ${m.metrics.roc_auc}`,
      data: m.roc.fpr.map((x, k) => ({ x, y: m.roc.tpr[k] })),
      borderColor: modelColor(m, i).stroke,
      borderWidth: 1.8,
      pointRadius: 0,
      tension: 0.1,
      fill: false,
    }));
    datasets.push({
      label: "chance",
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: PALETTE.text3,
      borderWidth: 1,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: false,
    });
    new Chart(ctx, {
      type: "line",
      data: { datasets },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { boxWidth: 18, boxHeight: 2 } },
          tooltip: { ...tooltip, callbacks: { title: (items) => `FPR ${items[0].parsed.x.toFixed(3)}` } },
        },
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "false-positive rate", color: PALETTE.text3 }, grid: { color: PALETTE.grid }, border: { display: false } },
          y: { min: 0, max: 1, title: { display: true, text: "true-positive rate", color: PALETTE.text3 }, grid: { color: PALETTE.grid }, border: { display: false } },
        },
      },
    });
  }

  // reliability curves: observed placement rate vs predicted probability,
  // one line per model + a dashed perfectly-calibrated diagonal
  function buildCalibration(canvas, models) {
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    const ctx = canvas.getContext("2d");
    const datasets = models.map((m, i) => ({
      label: `${m.name} · Brier ${m.metrics.brier}`,
      data: m.reliability.bin_mid.map((x, k) => ({ x, y: m.reliability.frac_pos[k] })),
      borderColor: modelColor(m, i).stroke,
      backgroundColor: modelColor(m, i).stroke,
      borderWidth: 1.8,
      pointRadius: 2.5,
      tension: 0.15,
      fill: false,
    }));
    datasets.push({
      label: "perfectly calibrated",
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: PALETTE.text3,
      borderWidth: 1,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: false,
    });
    new Chart(ctx, {
      type: "line",
      data: { datasets },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { boxWidth: 18, boxHeight: 2 } },
          tooltip: { ...tooltip, callbacks: { title: (items) => `predicted ${items[0].parsed.x.toFixed(2)}` } },
        },
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "mean predicted probability", color: PALETTE.text3 }, grid: { color: PALETTE.grid }, border: { display: false } },
          y: { min: 0, max: 1, title: { display: true, text: "observed placement rate", color: PALETTE.text3 }, grid: { color: PALETTE.grid }, border: { display: false } },
        },
      },
    });
  }

  // benchmark comparison: one bar per model, grouped by sealed-test metric
  function buildBenchmark(canvas, models) {    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    const metrics = [
      ["accuracy", "Accuracy"],
      ["precision", "Precision"],
      ["recall", "Recall"],
      ["f1", "F1"],
      ["roc_auc", "ROC-AUC"],
    ];
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: metrics.map(([, label]) => label),
        datasets: models.map((m, i) => ({
          label: m.name,
          data: metrics.map(([key]) => m.metrics[key]),
          backgroundColor: modelColor(m, i).fill,
          borderColor: modelColor(m, i).stroke,
          borderWidth: 1,
          borderRadius: 2,
          maxBarThickness: 26,
        })),
      },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { boxWidth: 9, boxHeight: 9, usePointStyle: true, pointStyle: "rectRounded" },
          },
          tooltip: {
            ...tooltip,
            displayColors: true,
            callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}` },
          },
        },
        scales: {
          x: { grid: { display: false }, border: { display: false } },
          y: {
            min: 0,
            max: 1,
            grid: { color: PALETTE.grid },
            border: { display: false },
            ticks: { maxTicksLimit: 6 },
          },
        },
      },
    });
  }

  // dynamic re-renders (the benchmark console in script.js) go through these
  window.PPCharts = { buildRoc, buildBenchmark, buildCalibration };

  function buildAll() {
    document.querySelectorAll("canvas[data-chart]").forEach((canvas) => {
      const kind = canvas.dataset.chart;
      const key = canvas.dataset.key;
      try {
        if (kind === "hist" && EDA.histograms) buildHistogram(canvas, EDA.histograms[key], PALETTE.accentFill, PALETTE.accent);
        else if (kind === "std" && EDA.standardized) buildHistogram(canvas, EDA.standardized[key], PALETTE.slateFill, PALETTE.slate);
        else if (kind === "missing" && EDA.missing) buildMissing(canvas, EDA.missing);
        else if (kind === "influence" && EDA.influence) buildInfluence(canvas, EDA.influence);
        else if (kind === "cat" && EDA.categories) buildCategory(canvas, EDA.categories[key]);
        else if (kind === "gender" && EDA.gender_split) buildGender(canvas, EDA.gender_split);
        else if (kind === "roc" && (EDA.models || EDA.roc)) buildRoc(canvas, EDA.models || EDA.roc);
        else if (kind === "calibration" && (EDA.models || EDA.roc)) buildCalibration(canvas, EDA.models || EDA.roc);
        else if (kind === "importance" && EDA.importance) buildInfluence(canvas, EDA.importance, "mean decrease in impurity");
        else if (kind === "rocsel" || kind === "benchmark") {
          // model pages carry their payload in window.MODEL_PAGE
          const mp = window.MODEL_PAGE;
          if (mp && Array.isArray(mp.models) && mp.models.length) {
            if (kind === "benchmark") buildBenchmark(canvas, mp.models);
            else {
              const sel = mp.models.find((m) => m.key === mp.selectedKey);
              if (sel) buildRoc(canvas, [sel]);
            }
          }
        }
      } catch (err) {
        // a broken chart should never take the page down with it
        console.error("chart failed:", kind, key || "", err);
      }
    });
  }

  // Wait for webfonts before measuring — otherwise axis labels are sized
  // with a fallback font and long labels (e.g. MockInterviewScore) clip.
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(buildAll);
  } else {
    buildAll();
  }
})();
