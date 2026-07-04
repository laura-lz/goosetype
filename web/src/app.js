const FONT_OPTIONS = {
  arial: {
    label: "Arial",
    family: "Arial, Helvetica, sans-serif",
    feature: "Arial shape masks",
  },
  times: {
    label: "Times New Roman",
    family: "\"Times New Roman\", Times, serif",
    feature: "Times New Roman serifs",
  },
  rubik: {
    label: "Rubik",
    family: "Rubik, Arial, sans-serif",
    feature: "Rubik rounded geometry",
  },
};

const STYLE_OPTIONS = {
  standard: { label: "Standard", weight: 500, style: "normal", familySuffix: "" },
  italic: { label: "Italic", weight: 500, style: "italic", familySuffix: "" },
  bold: { label: "Bold", weight: 800, style: "normal", familySuffix: "" },
  cursive: { label: "Cursive", weight: 520, style: "italic", familySuffix: ", cursive" },
};

const state = {
  text: "GOOSETYPE",
  font: "arial",
  style: "standard",
  readability: 68,
  mode: "photo",
  allCaps: true,
};

const els = {
  textInput: document.querySelector("#textInput"),
  fontSelect: document.querySelector("#fontSelect"),
  styleGroup: document.querySelector("#styleGroup"),
  modeGroup: document.querySelector("#modeGroup"),
  readabilitySlider: document.querySelector("#readabilitySlider"),
  allCapsToggle: document.querySelector("#allCapsToggle"),
  downloadButton: document.querySelector("#downloadButton"),
  requestButton: document.querySelector("#requestButton"),
  fontPreview: document.querySelector("#fontPreview"),
  previewTitle: document.querySelector("#previewTitle"),
  readabilityLabel: document.querySelector("#readabilityLabel"),
  readabilityModeLabel: document.querySelector("#readabilityModeLabel"),
  fontFeatureSummary: document.querySelector("#fontFeatureSummary"),
  renderModeSummary: document.querySelector("#renderModeSummary"),
  styleSummary: document.querySelector("#styleSummary"),
  exportStatus: document.querySelector("#exportStatus"),
};

bindControls();
render();

function bindControls() {
  els.textInput.addEventListener("input", () => {
    state.text = els.textInput.value;
    render();
  });

  els.fontSelect.addEventListener("change", () => {
    state.font = els.fontSelect.value;
    render();
  });

  els.styleGroup.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-style]");
    if (!button) return;
    state.style = button.dataset.style;
    setActiveStyle();
    render();
  });

  els.modeGroup.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;
    state.mode = button.dataset.mode;
    setActiveMode();
    render();
  });

  els.readabilitySlider.addEventListener("input", () => {
    state.readability = Number(els.readabilitySlider.value);
    render();
  });

  els.allCapsToggle.addEventListener("input", () => {
    state.allCaps = els.allCapsToggle.checked;
    render();
  });

  els.requestButton.addEventListener("click", () => {
    const payload = buildGenerationRequest();
    downloadRequestPayload(JSON.stringify(payload, null, 2));
  });

  els.downloadButton.addEventListener("click", () => {
    els.exportStatus.textContent = "TTF export will download from the backend font generator once connected.";
  });
}

function render() {
  const font = FONT_OPTIONS[state.font];
  const style = STYLE_OPTIONS[state.style];
  const text = state.allCaps ? state.text.toUpperCase() : state.text;
  const abstractness = 100 - state.readability;

  els.fontPreview.textContent = text || "GOOSETYPE";
  els.fontPreview.style.fontFamily = `${font.family}${style.familySuffix}`;
  els.fontPreview.style.fontWeight = style.weight;
  els.fontPreview.style.fontStyle = style.style;
  els.fontPreview.style.letterSpacing = `${Math.round(abstractness / 16)}px`;
  els.fontPreview.classList.toggle("abstract", state.readability < 42);
  els.fontPreview.classList.toggle("precise", state.readability >= 70);
  els.fontPreview.classList.toggle("silhouette", state.mode === "silhouette");

  els.previewTitle.textContent = `${font.label} ${style.label}`;
  els.readabilityLabel.textContent = `${state.readability}%`;
  els.readabilityModeLabel.textContent = readabilityLabel();
  els.fontFeatureSummary.textContent = font.feature;
  els.renderModeSummary.textContent = `${state.mode === "photo" ? "Photo" : "Silhouette"} mode`;
  els.styleSummary.textContent = styleSummary();

  els.exportStatus.textContent = "Frontend request builder ready. Backend font generation endpoint pending.";
}

function setActiveStyle() {
  for (const button of els.styleGroup.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.style === state.style);
  }
}

function setActiveMode() {
  for (const button of els.modeGroup.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.mode === state.mode);
  }
}

function styleSummary() {
  return `${readabilityLabel()}, ${STYLE_OPTIONS[state.style].label.toLowerCase()}`;
}

function readabilityLabel() {
  if (state.readability >= 70) return "readable";
  if (state.readability <= 35) return "abstract";
  return "balanced";
}

function buildGenerationRequest() {
  return {
    text: state.allCaps ? state.text.toUpperCase() : state.text,
    reference_font: {
      family: FONT_OPTIONS[state.font].label,
      style: STYLE_OPTIONS[state.style].label,
    },
    non_font_dependent_controls: {
      readability: state.readability / 100,
      abstraction: (100 - state.readability) / 100,
      render_mode: state.mode,
      all_caps: state.allCaps,
    },
    expected_backend_pipeline: [
      "extract feature similarity from selected font",
      "read precomputed goose image parameters",
      "score goose fit against font and non-font controls",
      "generate and return ttf font",
    ],
  };
}

function downloadRequestPayload(payload) {
  const blob = new Blob([payload], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "goosetype-font-request.json";
  link.click();
  URL.revokeObjectURL(url);
  els.exportStatus.textContent = "Downloaded backend request JSON. TTF download will replace this once generation is connected.";
}
