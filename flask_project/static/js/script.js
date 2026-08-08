// Upload dropzone: show the chosen filename and highlight on drag-over.
// Train page: model picker + interactive benchmark console (API-driven, with
// an embedded-data fallback for the static Pages build).
// All other behaviour is server-rendered; nothing here is required for the
// page to function.

// ------------------------------------------------------------------
// shared helpers
// ------------------------------------------------------------------

function setBusy(btn, label) {
  btn.dataset.origLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = "";
  const spin = document.createElement("span");
  spin.className = "spinner";
  btn.appendChild(spin);
  btn.appendChild(document.createTextNode(label));
}

function restoreBusyButtons() {
  document.querySelectorAll("button[data-orig-label]").forEach((btn) => {
    btn.disabled = false;
    btn.textContent = btn.dataset.origLabel;
    delete btn.dataset.origLabel;
  });
}

const fmt4 = (x) => Number(x).toFixed(4);
const pct1 = (x) => (Number(x) * 100).toFixed(1) + "%";
const intFmt = (x) => Number(x).toLocaleString("en-US");
// for values interpolated into innerHTML that crossed the network
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// single-model detail panel — mirrors the server-rendered markup in
// train.html, used by the static build (no server to navigate to)
function modelDetailHtml(m, mp) {
  const cards = [
    ["Accuracy", m.metrics.accuracy],
    ["Precision", m.metrics.precision],
    ["Recall", m.metrics.recall],
    ["F1", m.metrics.f1],
    ["ROC-AUC", m.metrics.roc_auc, true],
  ];
  return `
  <div class="panel model-detail mt-28">
    <div class="panel-head">
      <span class="panel-title">${m.name}</span>
      ${m.name === mp.best ? '<span class="badge badge-accent">Champion</span>' : ""}
    </div>
    <div class="panel-body">
      <p class="model-role">${m.note} · input ${m.needs_scaling ? "z-scored" : "raw imputed"} · ${m.settings} · ${m.calibration || "Platt sigmoid"}</p>
      <div class="metrics metrics-auto">
        ${cards.map(([label, v, accent]) => `
        <div class="metric">
          <span class="metric-value${accent ? " text-accent" : ""}">${pct1(v)}</span>
          <span class="metric-label">${label}</span>
        </div>`).join("")}
      </div>
      <p class="page-meta">Sealed test set · threshold 0.5 · CV ROC-AUC ${m.cv_auc_mean} ± ${m.cv_auc_std}
        (${mp.cvFolds}-fold on ${intFmt(mp.cvRows)} training rows) · fitted in ${m.train_time}s</p>
      <div class="eval-duo">
        <div>
          <div class="section-head mt-0"><h2 class="section-title">Confusion matrix</h2></div>
          <div class="cm-grid" role="img"
               aria-label="Confusion matrix: ${m.confusion.tp} true positives, ${m.confusion.fp} false positives, ${m.confusion.fn} false negatives, ${m.confusion.tn} true negatives">
            <span></span>
            <span class="cm-axis">Predicted placed</span>
            <span class="cm-axis">Predicted not</span>
            <span class="cm-axis cm-axis-row">Actually placed</span>
            <span class="cm-cell cm-tp"><strong>${intFmt(m.confusion.tp)}</strong><span>true positive</span></span>
            <span class="cm-cell cm-fn"><strong>${intFmt(m.confusion.fn)}</strong><span>false negative</span></span>
            <span class="cm-axis cm-axis-row">Actually not</span>
            <span class="cm-cell cm-fp"><strong>${intFmt(m.confusion.fp)}</strong><span>false positive</span></span>
            <span class="cm-cell cm-tn"><strong>${intFmt(m.confusion.tn)}</strong><span>true negative</span></span>
          </div>
        </div>
        <div>
          <div class="section-head mt-0"><h2 class="section-title">ROC curve</h2></div>
          <div class="chart-card">
            <div class="chart-wrap h-340"><canvas id="rocSelDynamic" role="img"
                 aria-label="ROC curve for ${m.name}"></canvas></div>
          </div>
        </div>
      </div>
    </div>
  </div>`;
}

// benchmark banner + table — mirrors the server-rendered markup in train.html
function benchmarkHtml(data) {
  const n = data.models.length;
  const best = data.models.find((m) => m.key === data.best.key) || data.models[0];
  const subsetNote = n < 3 ? ` · of the ${n} selected` : "";
  const freshNote = data.source === "fresh_run" ? " · fresh re-train" : "";
  const rows = data.models.map((m) => {
    const isBest = m.key === data.best.key;
    return `<tr class="${isBest ? "champion-row" : ""}">
      <td class="strong">${m.name}${isBest ? '<span class="badge badge-accent badge-offset">Best</span>' : ""}</td>
      <td class="num">${fmt4(m.metrics.accuracy)}</td>
      <td class="num">${fmt4(m.metrics.precision)}</td>
      <td class="num">${fmt4(m.metrics.recall)}</td>
      <td class="num">${fmt4(m.metrics.f1)}</td>
      <td class="num accent">${fmt4(m.metrics.roc_auc)}</td>
      <td class="num">${fmt4(m.metrics.brier)}</td>
      <td class="num">${fmt4(m.metrics.log_loss)}</td>
      <td class="num dim">${m.cv_auc_mean} ± ${m.cv_auc_std}</td>
      <td class="num dim">${m.train_time}s</td>
    </tr>`;
  }).join("");
  return `
    <div class="bench-banner">
      <span class="bench-banner-kicker">Best performing model${subsetNote}${freshNote}</span>
      <span class="bench-banner-name">${best.name}</span>
      <span class="bench-banner-sub">highest cross-validated ROC-AUC — ${best.cv_auc_mean} ± ${best.cv_auc_std}
        over ${data.cv_folds} folds of the training split · sealed-test ROC-AUC ${fmt4(best.metrics.roc_auc)} · Brier ${fmt4(best.metrics.brier)}</span>
    </div>
    <div class="table-scroll">
      <table class="table" id="benchTable">
        <thead>
          <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Brier ↓</th><th>Log-loss ↓</th><th>CV ROC-AUC · ${data.cv_folds}-fold</th><th>Train time</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function bestOfSubset(models) {
  const best = models.reduce((a, b) => (b.cv_auc_mean > a.cv_auc_mean ? b : a));
  return { key: best.key, name: best.name };
}

document.addEventListener("DOMContentLoaded", () => {
  // controls that only make sense with JavaScript (the page works without)
  document.querySelectorAll(".needs-js").forEach((el) => el.classList.remove("needs-js"));

  // bfcache restore: never leave a button stuck on its busy state
  window.addEventListener("pageshow", (e) => {
    if (e.persisted) restoreBusyButtons();
  });

  // Keep the active pipeline stage visible in the sidebar's horizontal
  // scroller on narrow screens.
  const activeStep = document.querySelector(".pipeline-stepper li.is-active a");
  if (activeStep) {
    activeStep.scrollIntoView({ block: "nearest", inline: "center", behavior: "instant" });
  }

  // ------------------------------------------------------------------
  // Static-build behaviour (GitHub Pages): the export neutralizes forms
  // (onsubmit="return false", buttons become type="button"), so there is
  // no server. We detect that and do what we can in the browser.
  // ------------------------------------------------------------------

  // Upload: nothing to compute client-side — say so instead of dead-ending.
  const uploadBtn = document.querySelector(".upload-form button[type='button']");
  if (uploadBtn) {
    uploadBtn.disabled = true;
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = "Static showcase — the bundled dataset's output is shown below. Uploading runs in the full app (see the README).";
    uploadBtn.insertAdjacentElement("afterend", note);
  }

  // Predict (static build): try the hosted API with the selected model;
  // fall back to the exported, calibrated logistic baseline in-browser.
  const predictBtn = document.querySelector(".predict-form button[type='button']");
  if (predictBtn && window.LR_MODEL) {
    // fork-friendly: a host can point the demo at its own deployment by
    // setting window.PP_LIVE_API before this script runs
    const LIVE_API = (window.PP_LIVE_API || "https://placement-predict-p2z1.onrender.com").replace(/\/$/, "");
    const modelSelect = document.getElementById("predictModel");
    const hint = document.getElementById("modelStaticHint");
    if (hint) {
      hint.textContent = "Static demo — the call goes to the hosted API with your selected model; if it's unreachable (the free tier sleeps), the calibrated logistic baseline runs in your browser instead.";
    }

    function readProfile() {
      const M = window.LR_MODEL;
      const meta = Object.fromEntries((window.FORM_META || []).map((m) => [m.name, m]));
      const payload = {};
      let z = M.intercept;
      M.features.forEach((name, i) => {
        const input = document.querySelector(`[name="${name}"]`);
        let v = parseFloat(input && input.value !== "" ? input.value : NaN);
        if (Number.isNaN(v)) v = meta[name] ? meta[name].default : M.mean[name];
        // clamp into the observed range — the live API rejects out-of-range
        // input; the static demo has no server round-trip to push back with
        if (meta[name]) v = Math.min(meta[name].max, Math.max(meta[name].min, v));
        payload[name] = v;
        z += M.coef[i] * (v - M.mean[name]) / M.std[name];
      });
      return { payload, z };
    }

    function resultHtml(placed, pct, kicker, note) {
      return `
        <div class="result-panel ${placed ? "result-placed" : "result-not"}">
          <span class="result-kicker">${kicker}</span>
          <span class="result-verdict">${placed ? "Placed" : "Not placed"}</span>
          <div class="result-track" role="img" aria-label="Placement probability ${pct} percent">
            <div class="result-fill" style="width: ${pct}%"></div>
          </div>
          <div class="result-meta">
            <span class="mono">${pct}% probability</span>
            <span class="mono dim">threshold 50%</span>
          </div>
        </div>
        <p class="note">${note}</p>`;
    }

    predictBtn.addEventListener("click", async () => {
      const aside = document.querySelector(".predict-result");
      if (!aside) return;
      const M = window.LR_MODEL;
      const { payload, z } = readProfile();
      const selected = modelSelect ? modelSelect.value : "best";

      // 1) the hosted API runs whichever model the user picked, server-side
      try {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 9000);
        const resp = await fetch(`${LIVE_API}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: selected, ...payload }),
          signal: ctrl.signal,
        });
        clearTimeout(timer);
        const data = await resp.json();
        const prob = Number(data && data.probability);
        if (!resp.ok || !Number.isFinite(prob)) {
          throw new Error((data && data.error) || `HTTP ${resp.status}`);
        }
        aside.innerHTML = resultHtml(
          Boolean(data.placed), prob,
          `${esc(data.model)} · live API`,
          `Served by the hosted app — the selected model ran server-side (calibrated probability, sealed-test ROC-AUC ${Number(data.roc_auc)}).`,
        );
        return;
      } catch (err) {
        // unreachable host (free tier sleeps) — fall through to the baseline
      }

      // 2) in-browser fallback: logistic baseline + its Platt calibrator
      //    (p = expit(-(a·z + b)) on the decision score — same map the
      //    server applies)
      let p = 1 / (1 + Math.exp(-z));
      if (M.cal) p = 1 / (1 + Math.exp(M.cal.a * z + M.cal.b));
      const pct = Math.round(p * 1000) / 10;
      const C = window.CHAMPION || {};
      aside.innerHTML = resultHtml(
        p >= 0.5, pct,
        "Logistic Regression · in-browser demo",
        `The hosted API was unreachable, so this calibrated call comes from the logistic baseline (ROC-AUC ${M.auc}) running in your browser. The full app predicts server-side with any selected model — recommended: ${C.name || "the champion"} (ROC-AUC ${C.auc}).`,
      );
    });
  }

  // Live predict form: busy state on submit (prevents duplicate requests
  // while the server trains/loads the selected model).
  const predictForm = document.querySelector(".predict-form");
  if (predictForm && !predictForm.hasAttribute("onsubmit")) {
    predictForm.addEventListener("submit", () => {
      const btn = predictForm.querySelector("button[type='submit']");
      if (btn) setBusy(btn, "Predicting…");
    });
  }

  // ------------------------------------------------------------------
  // Train page: model picker + benchmark console
  // ------------------------------------------------------------------
  const MP = window.MODEL_PAGE || null;
  const pickForm = document.getElementById("modelPickForm");
  const pickSelect = document.getElementById("modelPick");
  // the static export neutralizes every form (onsubmit="return false")
  const staticBuild = !!(pickForm && pickForm.hasAttribute("onsubmit"));

  function renderModelDetail(key) {
    const host = document.getElementById("modelDetail");
    const m = MP.models.find((entry) => entry.key === key);
    if (!host || !m) return;
    host.innerHTML = modelDetailHtml(m, MP);
    const canvas = host.querySelector("#rocSelDynamic");
    if (canvas && window.PPCharts) window.PPCharts.buildRoc(canvas, [m]);
    host.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  if (MP && pickForm && pickSelect) {
    if (staticBuild) {
      // no server round-trips on Pages — render from the embedded payload
      const render = () => { if (pickSelect.value) renderModelDetail(pickSelect.value); };
      pickSelect.addEventListener("change", render);
      const btn = pickForm.querySelector("button");
      if (btn) btn.addEventListener("click", render);
    } else {
      pickForm.addEventListener("submit", () => {
        const btn = pickForm.querySelector("button[type='submit']");
        if (btn) setBusy(btn, "Evaluating…");
      });
    }
  }

  const benchRun = document.getElementById("benchRun");
  if (MP && benchRun) {
    const status = document.getElementById("benchStatus");
    const errorBox = document.getElementById("benchError");
    const results = document.getElementById("benchResults");
    const chartCanvas = document.getElementById("benchChart");
    const freshBox = document.getElementById("benchFresh");
    const checks = Array.from(document.querySelectorAll(".bench-check"));

    function renderBenchmark(data) {
      results.innerHTML = benchmarkHtml(data);
      if (chartCanvas && window.PPCharts) window.PPCharts.buildBenchmark(chartCanvas, data.models);
    }

    function showBenchError(message) {
      // textContent, never innerHTML — the API echoes request values in its
      // error messages, so injecting markup here would be an XSS hole
      errorBox.textContent = "";
      const strong = document.createElement("strong");
      strong.textContent = "Benchmark failed";
      errorBox.appendChild(strong);
      errorBox.appendChild(document.createTextNode(message));
      errorBox.hidden = false;
      status.textContent = "";
    }

    function renderEmbeddedSubset(keys, fresh) {
      // static showcase: no API — filter the recorded run embedded in the page
      const subset = MP.models.filter((m) => keys.includes(m.key));
      renderBenchmark({
        models: subset,
        best: bestOfSubset(subset),
        cv_folds: MP.cvFolds,
        split: MP.split,
      });
      status.textContent = fresh
        ? "Static showcase — re-training needs the full app; showing the recorded run embedded in the page."
        : "Static showcase — no API on this host, so this filters the recorded run embedded in the page.";
    }

    benchRun.addEventListener("click", async () => {
      const keys = checks.filter((c) => c.checked).map((c) => c.value);
      const fresh = !!(freshBox && freshBox.checked);
      errorBox.hidden = true;
      if (!keys.length) {
        status.textContent = "Select at least one model to benchmark.";
        return;
      }
      setBusy(benchRun, "Benchmarking…");
      status.textContent = fresh
        ? "Re-training from scratch — re-fitting and re-evaluating the selected models; tens of seconds on the free host."
        : "Training & evaluating — a first run on a fresh dataset fits every candidate (~15–30 s on the free host); cached runs answer instantly.";
      try {
        const resp = await fetch("/api/benchmark", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ models: keys, fresh }),
        });
        const isJson = (resp.headers.get("content-type") || "").includes("application/json");
        const data = isJson ? await resp.json() : null;
        if (isJson && resp.ok && data && data.ok) {
          renderBenchmark(data);
          status.textContent = `Done — ${data.models.length} model(s) compared on ${intFmt(data.split.test)} sealed test rows · ${data.dataset}.`
            + (data.source === "fresh_run" ? " Freshly re-trained — the recipe is deterministic, so the numbers match the cached evaluation." : "");
        } else if (staticBuild) {
          renderEmbeddedSubset(keys, fresh);
        } else {
          showBenchError((data && data.error) || `The server answered with HTTP ${resp.status}.`);
        }
      } catch (err) {
        if (staticBuild) {
          renderEmbeddedSubset(keys, fresh);
        } else {
          showBenchError(err.message || "Network error.");
        }
      } finally {
        benchRun.disabled = false;
        benchRun.textContent = benchRun.dataset.origLabel || "Run benchmark";
        delete benchRun.dataset.origLabel;
      }
    });
  }

  // Predict form: show the native validation popup as soon as a field is
  // left with invalid data — don't wait for the submit click.
  document.querySelectorAll(".predict-form .field-input").forEach((input) => {
    input.addEventListener("blur", () => {
      if (input.value !== "" && !input.checkValidity()) {
        input.reportValidity();
      }
    });
  });

  const dropZone = document.getElementById("uploadDrop");
  const fileInput = document.getElementById("datasetInput");
  const filenameLabel = document.getElementById("uploadFilename");

  if (!dropZone || !fileInput || !filenameLabel) return;

  const showFilename = () => {
    filenameLabel.textContent = fileInput.files.length ? fileInput.files[0].name : "";
  };

  fileInput.addEventListener("change", showFilename);

  ["dragenter", "dragover"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("is-dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const dropped = e.dataTransfer.files;
    if (dropped.length) {
      fileInput.files = dropped;
      showFilename();
    }
  });
});
