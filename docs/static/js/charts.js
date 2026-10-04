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
  if (typeof Chart === "undefined") {
    // Offline/local fallback. The public app used to depend on the jsDelivr
    // Chart.js CDN; when that CDN was unavailable the canvases stayed blank.
    // Keep the visual pipeline functional without an external runtime asset.
    const EDA = window.EDA || {};
    const C = {
      text: "#EBECE8", text2: "#9BA29A", text3: "#8A9189",
      accent: "#D9A63F", danger: "#C65D55", slate: "#6E8FA0",
      grid: "rgba(235,236,232,.10)", panel: "#1B1F1C"
    };

    function setup(canvas) {
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(320, Math.round(rect.width || 640));
      const h = Math.max(180, Math.round(rect.height || 320));
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.font = "10px 'IBM Plex Mono', ui-monospace, monospace";
      return { ctx, w, h };
    }

    function grid(ctx, x, y, w, h, max, horizontal = true) {
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      const steps = 5;
      for (let i = 0; i <= steps; i++) {
        const p = i / steps;
        ctx.beginPath();
        if (horizontal) {
          ctx.moveTo(x, y + h - p * h);
          ctx.lineTo(x + w, y + h - p * h);
        } else {
          ctx.moveTo(x + p * w, y);
          ctx.lineTo(x + p * w, y + h);
        }
        ctx.stroke();
        ctx.fillStyle = C.text3;
        ctx.textAlign = "right";
        ctx.fillText(String(Math.round(max * p)), x - 8, y + h - p * h + 3);
      }
    }

    function xLabels(ctx, labels, x, y, w, max = 8) {
      const step = Math.max(1, Math.ceil(labels.length / max));
      ctx.fillStyle = C.text3;
      ctx.textAlign = "center";
      labels.forEach((label, i) => {
        if (i % step !== 0 && i !== labels.length - 1) return;
        const px = x + (i + .5) * (w / labels.length);
        const text = String(label);
        ctx.fillText(text.length > 18 ? text.slice(0, 17) + "…" : text, px, y);
      });
    }

    function bar(canvas, labels, values, opts = {}) {
      const {ctx,w,h} = setup(canvas);
      const left = opts.left || 62, right = 18, top = 18, bottom = opts.bottom || 52;
      const cw = w - left - right, ch = h - top - bottom;
      const max = opts.max ?? Math.max(...values, 1);
      grid(ctx, left, top, cw, ch, max);
      const bw = Math.min(opts.maxBar || 48, cw / Math.max(values.length * 1.7, 1));
      values.forEach((v, i) => {
        const bh = (Number(v) / max) * ch;
        const bx = left + (i + .5) * cw / values.length - bw / 2;
        const by = top + ch - bh;
        ctx.fillStyle = opts.fill || C.accent;
        ctx.globalAlpha = .72;
        ctx.fillRect(bx, by, bw, bh);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = opts.stroke || C.accent;
        ctx.strokeRect(bx + .5, by + .5, Math.max(1,bw-1), Math.max(1,bh-1));
      });
      xLabels(ctx, labels, left, top + ch + 28, cw, opts.maxLabels || 7);
      if (opts.title) {
        ctx.fillStyle = C.text3; ctx.textAlign = "left";
        ctx.fillText(opts.title, 8, top + 4);
      }
    }

    function horizontal(canvas, labels, values, opts = {}) {
      const {ctx,w,h} = setup(canvas);
      const left = 150, right = 28, top = 16, bottom = 18;
      const cw = w-left-right, ch = h-top-bottom;
      const maxAbs = opts.max ?? Math.max(...values.map(v => Math.abs(Number(v))), 1);
      const row = ch / Math.max(values.length,1);
      values.forEach((v,i) => {
        const n = Number(v);
        const bw = Math.abs(n) / maxAbs * (cw/2);
        const cy = top + (i+.5)*row;
        const base = left + (n < 0 ? cw/2 : 0);
        ctx.fillStyle = n < 0 ? C.danger : (opts.fill || C.accent);
        ctx.globalAlpha=.72; ctx.fillRect(n<0 ? base-bw : base, cy-row*.3, bw, row*.6); ctx.globalAlpha=1;
        ctx.fillStyle=C.text3; ctx.textAlign="right";
        const label=String(labels[i]); ctx.fillText(label.length>20?label.slice(0,19)+"…":label,left-10,cy+3);
      });
      ctx.strokeStyle=C.grid; ctx.beginPath(); ctx.moveTo(left+cw/2,top); ctx.lineTo(left+cw/2,top+ch); ctx.stroke();
    }

    function line(canvas, datasets, opts = {}) {
      const {ctx,w,h} = setup(canvas);
      const left=54,right=18,top=18,bottom=34,cw=w-left-right,ch=h-top-bottom;
      ctx.strokeStyle=C.grid; ctx.lineWidth=1;
      for(let i=0;i<=5;i++){const p=i/5;ctx.beginPath();ctx.moveTo(left,top+ch-p*ch);ctx.lineTo(left+cw,top+ch-p*ch);ctx.stroke();}
      datasets.forEach((ds,di)=>{
        ctx.strokeStyle=ds.color || [C.accent,C.slate,C.text2][di%3];
        ctx.lineWidth=1.8; ctx.beginPath();
        ds.points.forEach((p,i)=>{const px=left+p.x*cw,py=top+ch-p.y*ch;i?ctx.lineTo(px,py):ctx.moveTo(px,py);});
        ctx.stroke();
      });
      ctx.fillStyle=C.text3;ctx.textAlign="center";
      ctx.fillText(opts.xTitle||"0",left,top+ch+20);ctx.fillText("1",left+cw,top+ch+20);
      ctx.textAlign="right";ctx.fillText("1",left-8,top+4);ctx.fillText("0",left-8,top+ch+3);
    }

    function donut(canvas, overview) {
      const {ctx,w,h}=setup(canvas);
      const cx=w/2,cy=h/2-8,r=Math.min(w,h)*.30;
      const total=Math.max(1,overview.placed+overview.not_placed);
      let a=-Math.PI/2;
      [[overview.placed,C.accent],[overview.not_placed,"#3A403B"]].forEach(([v,color])=>{
        const end=a+v/total*Math.PI*2;ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,a,end);ctx.closePath();ctx.fillStyle=color;ctx.fill();a=end;
      });
      ctx.fillStyle=C.panel;ctx.beginPath();ctx.arc(cx,cy,r*.62,0,Math.PI*2);ctx.fill();
      ctx.fillStyle=C.text;ctx.textAlign="center";ctx.font="700 18px 'IBM Plex Mono',monospace";ctx.fillText(((overview.placed/total)*100).toFixed(1)+"%",cx,cy+5);
    }

    function buildMissing(canvas,payload){ bar(canvas,payload.chart_labels,payload.chart_values,{fill:C.danger,stroke:C.danger,maxLabels:5,maxBar:60,title:"missing cells"}); }
    function buildRateBars(canvas,payload){ bar(canvas,payload.labels,payload.rates,{max:100,fill:C.accent,stroke:C.accent,maxLabels:9,maxBar:42}); }
    function buildInfluence(canvas,payload){ horizontal(canvas,payload.labels,payload.values,{fill:C.accent}); }
    function buildCategory(canvas,rows){ bar(canvas,rows.map(r=>r.label),rows.map(r=>r.rate),{max:100,fill:C.accent,stroke:C.accent,maxLabels:8}); }
    function buildGender(canvas,payload){ bar(canvas,payload.labels,payload.placed,{fill:C.accent,stroke:C.accent,maxLabels:8}); }
    function buildDonut(canvas,overview){ donut(canvas,overview); }

    function buildHistogram(canvas,payload,fill=C.accent,stroke=C.accent){
      bar(canvas,payload.labels,payload.counts,{fill,stroke,maxLabels:8,maxBar:32});
    }
    function buildRoc(canvas,models){
      line(canvas,models.map((m,i)=>({color:[C.text2,C.slate,C.accent][i%3],points:m.roc.fpr.map((x,k)=>({x,y:m.roc.tpr[k]}))})),{xTitle:"false-positive rate"});
    }
    function buildCalibration(canvas,models){
      line(canvas,models.map((m,i)=>({color:[C.text2,C.slate,C.accent][i%3],points:m.reliability.bin_mid.map((x,k)=>({x,y:m.reliability.frac_pos[k]}))})),{xTitle:"mean predicted probability"});
    }
    function buildBenchmark(canvas,models){
      const labels=["Accuracy","Precision","Recall","F1","ROC-AUC"];
      const {ctx,w,h}=setup(canvas); const left=58,right=18,top=18,bottom=38,cw=w-left-right,ch=h-top-bottom;
      grid(ctx,left,top,cw,ch,1);
      const group=cw/labels.length, bw=Math.min(18,group/(models.length+2));
      models.forEach((m,mi)=>labels.forEach((_,i)=>{
        const keys=["accuracy","precision","recall","f1","roc_auc"];
        const v=Number(m.metrics[keys[i]]); const bh=v*ch;
        const bx=left+i*group+group/2+(mi-(models.length-1)/2)*(bw+3)-bw/2;
        ctx.fillStyle=[C.text2,C.slate,C.accent][mi%3];ctx.globalAlpha=.72;ctx.fillRect(bx,top+ch-bh,bw,bh);ctx.globalAlpha=1;
      }));
      xLabels(ctx,labels,left,top+ch+24,cw,5);
    }
    function buildAllFallback(){
      document.querySelectorAll("canvas[data-chart]").forEach(canvas=>{
        const kind=canvas.dataset.chart,key=canvas.dataset.key;
        try{
          if(kind==="hist"&&EDA.histograms)buildHistogram(canvas,EDA.histograms[key]);
          else if(kind==="donut"&&EDA.overview)buildDonut(canvas,EDA.overview);
          else if(kind==="ratefeat"&&EDA.rateByFeature)buildRateBars(canvas,EDA.rateByFeature[key]);
          else if(kind==="std"&&EDA.standardized)buildHistogram(canvas,EDA.standardized[key],C.slate,C.slate);
          else if(kind==="missing"&&EDA.missing)buildMissing(canvas,EDA.missing);
          else if(kind==="influence"&&EDA.influence)buildInfluence(canvas,EDA.influence);
          else if(kind==="cat"&&EDA.categories)buildCategory(canvas,EDA.categories[key]);
          else if(kind==="gender"&&EDA.gender_split)buildGender(canvas,EDA.gender_split);
          else if(kind==="roc"&&EDA.models)buildRoc(canvas,EDA.models);
          else if(kind==="calibration"&&EDA.models)buildCalibration(canvas,EDA.models);
          else if(kind==="importance"&&EDA.importance)buildInfluence(canvas,EDA.importance);
          else if(kind==="benchmark"&&window.MODEL_PAGE?.models)buildBenchmark(canvas,window.MODEL_PAGE.models);
        }catch(err){console.error("fallback chart failed:",kind,key||"",err);}
      });
    }
    window.PPCharts={buildRoc,buildBenchmark,buildCalibration,buildHistogram,buildRateBars,buildDonut,palette:{
      accentFill:"rgba(217,166,63,.55)",accent:C.accent
    }};
    const ready=()=>buildAllFallback();
    if(document.fonts?.ready) document.fonts.ready.then(ready); else ready();
    return;
  }

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
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
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

  // dataset overview: placed vs not-placed donut
  function buildDonut(canvas, overview) {
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    const total = overview.placed + overview.not_placed;
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Placed", "Not placed"],
        datasets: [
          {
            data: [overview.placed, overview.not_placed],
            backgroundColor: [PALETTE.accent, "#3A403B"],
            borderColor: PALETTE.panel,
            borderWidth: 2,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        cutout: "62%",
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 9, boxHeight: 9, usePointStyle: true, pointStyle: "rectRounded" },
          },
          tooltip: {
            ...tooltip,
            callbacks: {
              label: (c) => ` ${c.label}: ${c.parsed.toLocaleString()} (${(c.parsed / total * 100).toFixed(1)}%)`,
            },
          },
        },
      },
    });
  }

  // dataset overview: placement rate per band of one numeric feature
  function buildRateBars(canvas, payload) {
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: payload.labels,
        datasets: [
          {
            data: payload.rates,
            backgroundColor: PALETTE.accentFill,
            borderColor: PALETTE.accent,
            borderWidth: 1,
            borderRadius: 2,
            maxBarThickness: 34,
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
              label: (c) => ` ${c.parsed.y}% placed · n=${payload.counts[c.dataIndex].toLocaleString()}`,
            },
          },
        },
        scales: {
          x: { grid: { display: false }, border: { display: false }, ticks: { maxRotation: 35, autoSkip: true, maxTicksLimit: 9 } },
          y: {
            min: 0,
            max: 100,
            grid: { color: PALETTE.grid },
            border: { display: false },
            ticks: { callback: (v) => v + "%" },
            title: { display: true, text: "placement rate", color: PALETTE.text3 },
          },
        },
      },
    });
  }

  // dynamic re-renders (the benchmark console in script.js) go through these
  window.PPCharts = {
    buildRoc, buildBenchmark, buildCalibration, buildHistogram, buildRateBars,
    buildDonut, palette: PALETTE,
  };

  function buildAll() {
    document.querySelectorAll("canvas[data-chart]").forEach((canvas) => {
      const kind = canvas.dataset.chart;
      const key = canvas.dataset.key;
      try {
        if (kind === "hist" && EDA.histograms) buildHistogram(canvas, EDA.histograms[key], PALETTE.accentFill, PALETTE.accent);
        else if (kind === "donut" && EDA.overview) buildDonut(canvas, EDA.overview);
        else if (kind === "ratefeat" && EDA.rateByFeature && EDA.rateByFeature[key]) buildRateBars(canvas, EDA.rateByFeature[key]);
        else if (kind === "std" && EDA.standardized) buildHistogram(canvas, EDA.standardized[key], PALETTE.slateFill, PALETTE.slate);
        else if (kind === "missing" && EDA.missing) buildMissing(canvas, EDA.missing);
        else if (kind === "influence" && EDA.influence) buildInfluence(canvas, EDA.influence);
        else if (kind === "cat" && EDA.categories) buildCategory(canvas, EDA.categories[key]);
        else if (kind === "gender" && EDA.gender_split) buildGender(canvas, EDA.gender_split);
        else if (kind === "roc" && EDA.models) buildRoc(canvas, EDA.models);
        else if (kind === "calibration" && EDA.models) buildCalibration(canvas, EDA.models);
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
