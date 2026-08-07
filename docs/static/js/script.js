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
