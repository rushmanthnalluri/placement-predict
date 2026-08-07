// Chart builders for the EDA stages. Reads payloads from window.EDA
// (injected by the template) and renders into <canvas data-chart="...">.
// Requires Chart.js 4 (loaded by the template before this file).
//
// Visual language: flat fills, hairline grids, mono ticks, fast settles.
// One accent color; red is reserved for "missing / not placed".

(function () {
  if (typeof Chart === "undefined" || typeof window.EDA === "undefined") return;

  const EDA = window.EDA;

  const PALETTE = {
    text: "#EBECE8",
    text2: "#9BA29A",
    text3: "#8A9189",
    accent: "#D9A63F",
    accentFill: "rgba(217, 166, 63, 0.55)",
    slate: "#6E8FA0",
    slateFill: "rgba(110, 143, 160, 0.45)",
    danger: "#C65D55",
    dangerFill: "rgba(198, 93, 85, 0.55)",
    grid: "rgba(235, 236, 232, 0.06)",
    panel: "#1B1F1C",
    panelBorder: "#363D37",
  };

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

  function buildInfluence(canvas, payload) {
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
            suggestedMax: 0.8,
            title: { display: true, text: "correlation with PlacementStatus", color: PALETTE.text3 },
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
