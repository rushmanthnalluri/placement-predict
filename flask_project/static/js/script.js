// Upload dropzone: show the chosen filename and highlight on drag-over.
// All other behaviour is server-rendered; nothing here is required for the
// page to function.

document.addEventListener("DOMContentLoaded", () => {
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

  // Predict: run the exported logistic model in the browser.
  const predictBtn = document.querySelector(".predict-form button[type='button']");
  if (predictBtn && window.LR_MODEL) {
    predictBtn.addEventListener("click", () => {
      const M = window.LR_MODEL;
      const meta = Object.fromEntries((window.FORM_META || []).map((m) => [m.name, m]));

      let z = M.intercept;
      M.features.forEach((name, i) => {
        const input = document.querySelector(`[name="${name}"]`);
        let v = parseFloat(input && input.value !== "" ? input.value : NaN);
        if (Number.isNaN(v)) v = meta[name] ? meta[name].default : M.mean[name];
        // clamp into the observed range, same guard as the server
        if (meta[name]) v = Math.min(meta[name].max, Math.max(meta[name].min, v));
        z += M.coef[i] * (v - M.mean[name]) / M.std[name];
      });

      const p = 1 / (1 + Math.exp(-z));
      const pct = Math.round(p * 1000) / 10;
      const placed = p >= 0.5;

      const aside = document.querySelector(".predict-result");
      if (!aside) return;
      aside.innerHTML = `
        <div class="result-panel ${placed ? "result-placed" : "result-not"}">
          <span class="result-kicker">Logistic Regression · in-browser demo</span>
          <span class="result-verdict">${placed ? "Placed" : "Not placed"}</span>
          <div class="result-track" role="img" aria-label="Placement probability ${pct} percent">
            <div class="result-fill" style="width: ${pct}%"></div>
          </div>
          <div class="result-meta">
            <span class="mono">${pct}% probability</span>
            <span class="mono dim">threshold 50%</span>
          </div>
        </div>
        <p class="note">Static demo: this call comes from the logistic baseline
          (ROC-AUC ${M.auc}) running in your browser. The full app predicts
          server-side with the gradient-boosting champion (ROC-AUC 0.9733).</p>`;
    });
  }

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
