const FONT_OPTIONS = {
  arial: {
    label: "Arial",
    family: "Arial, Helvetica, sans-serif",
    candidatesUrl: "/data/font_candidates/arial_candidates.json",
  },
  times: {
    label: "Times New Roman",
    family: "\"Times New Roman\", Times, serif",
    candidatesUrl: "/data/font_candidates/times_new_roman_candidates.json",
  },
};

const state = {
  text: "GOOSETYPE",
  font: "arial",
  italic: false,
  bold: false,
  readability: 68,
  blur: 0,
  mode: "photo",
  allCaps: true,
};

const candidateCache = {};

const els = {
  textInput: document.querySelector("#textInput"),
  fontSelect: document.querySelector("#fontSelect"),
  styleGroup: document.querySelector("#styleGroup"),
  modeGroup: document.querySelector("#modeGroup"),
  readabilitySlider: document.querySelector("#readabilitySlider"),
  blurSlider: document.querySelector("#blurSlider"),
  allCapsToggle: document.querySelector("#allCapsToggle"),
  downloadButton: document.querySelector("#downloadButton"),
  requestButton: document.querySelector("#requestButton"),
  fontPreview: document.querySelector("#fontPreview"),
  previewTitle: document.querySelector("#previewTitle"),
  readabilityLabel: document.querySelector("#readabilityLabel"),
  readabilityModeLabel: document.querySelector("#readabilityModeLabel"),
  exportStatus: document.querySelector("#exportStatus"),
  candidateStatus: document.querySelector("#candidateStatus"),
  candidateStrip: document.querySelector("#candidateStrip"),
};

bindControls();
setActiveStyle();
setActiveMode();
loadCandidates(state.font);
render();

function bindControls() {
  els.textInput.addEventListener("input", () => {
    state.text = els.textInput.value;
    render();
  });

  els.fontSelect.addEventListener("change", () => {
    state.font = els.fontSelect.value;
    loadCandidates(state.font);
    render();
  });

  els.styleGroup.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-style]");
    if (!button) return;
    state[button.dataset.style] = !state[button.dataset.style];
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

  els.blurSlider.addEventListener("input", () => {
    state.blur = Number(els.blurSlider.value);
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

async function loadCandidates(fontKey) {
  if (candidateCache[fontKey]) {
    render();
    return;
  }

  const font = FONT_OPTIONS[fontKey];
  els.candidateStatus.textContent = `Loading ${font.label} goose candidates...`;
  try {
    const response = await fetch(font.candidatesUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    candidateCache[fontKey] = await response.json();
    const letters = Object.keys(candidateCache[fontKey]).length;
    els.candidateStatus.textContent = `${font.label}: ${letters} letters loaded, 12 ranked candidates per letter.`;
  } catch (error) {
    els.candidateStatus.textContent = `Could not load candidate JSON from ${font.candidatesUrl}. Run the page from the repo root server.`;
    console.error(error);
  }
  render();
}

function render() {
  const font = FONT_OPTIONS[state.font];
  const style = currentStyle();
  const text = state.allCaps ? state.text.toUpperCase() : state.text;

  els.previewTitle.textContent = `${font.label} ${style.label}`;
  els.readabilityLabel.textContent = `${state.readability}%`;
  els.readabilityModeLabel.textContent = readabilityLabel();
  els.exportStatus.textContent = "Frontend request builder ready. Backend font generation endpoint pending.";

  renderGooseText(text || "GOOSETYPE");
}

function renderGooseText(text) {
  const candidates = candidateCache[state.font];
  els.fontPreview.innerHTML = "";
  els.fontPreview.classList.toggle("silhouette", state.mode === "silhouette");
  els.fontPreview.style.setProperty("--goose-blur", `${state.blur}px`);
  els.candidateStrip.style.setProperty("--goose-blur", `${state.blur}px`);

  if (!candidates) {
    const loading = document.createElement("p");
    loading.className = "fontPreviewFallback";
    loading.textContent = text;
    loading.style.fontFamily = FONT_OPTIONS[state.font].family;
    els.fontPreview.append(loading);
    els.candidateStrip.innerHTML = "";
    return;
  }

  const fragment = document.createDocumentFragment();
  const chosen = [];
  for (const char of text) {
    if (char === "\n") {
      fragment.append(document.createElement("br"));
      continue;
    }
    if (char === " ") {
      const spacer = document.createElement("span");
      spacer.className = "gooseGlyph spacer";
      fragment.append(spacer);
      continue;
    }
    const letter = state.allCaps ? char.toUpperCase() : char;
    const ranked = candidates[letter];
    if (!ranked || !ranked.length) {
      const fallback = document.createElement("span");
      fallback.className = "gooseGlyph missing";
      fallback.textContent = char;
      fragment.append(fallback);
      continue;
    }
    const candidate = pickCandidate(ranked);
    chosen.push({ letter, candidate });
    fragment.append(renderGlyph(letter, candidate));
  }
  els.fontPreview.append(fragment);
  renderCandidateStrip(chosen);
}

function pickCandidate(ranked) {
  const readableIndex = state.readability >= 72 ? 0 : state.readability >= 45 ? 1 : 2;
  return ranked[Math.min(readableIndex, ranked.length - 1)];
}

function renderGlyph(letter, candidate) {
  const glyph = document.createElement("span");
  glyph.className = "gooseGlyph";
  glyph.title = `${letter}: ${candidate.goose_id}, score ${candidate.score}`;

  const image = document.createElement("img");
  image.alt = `${letter} goose candidate`;
  image.src = toWebPath(candidateImagePath(candidate));
  image.loading = "lazy";
  image.decoding = "async";
  image.style.transform = candidate.flip_x ? "scaleX(-1)" : "";

  const label = document.createElement("small");
  label.textContent = letter;
  glyph.append(image, label);
  return glyph;
}

function renderCandidateStrip(chosen) {
  els.candidateStrip.innerHTML = "";
  const unique = [];
  const seen = new Set();
  for (const item of chosen) {
    const key = `${item.letter}-${item.candidate.mask_path}-${item.candidate.flip_x}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(item);
  }

  for (const item of unique.slice(0, 10)) {
    const card = document.createElement("article");
    card.className = "candidateCard";
    card.classList.toggle("silhouette", state.mode === "silhouette");
    const image = document.createElement("img");
    image.alt = `${item.letter} selected candidate`;
    image.src = toWebPath(candidateImagePath(item.candidate));
    image.style.transform = item.candidate.flip_x ? "scaleX(-1)" : "";
    const text = document.createElement("span");
    text.textContent = `${item.letter} ${item.candidate.score.toFixed(2)}`;
    card.append(image, text);
    els.candidateStrip.append(card);
  }
}

function candidateImagePath(candidate) {
  if (state.mode === "silhouette") {
    return candidate.silhouette_path || candidate.mask_path || candidate.cutout_path;
  }
  return candidate.cutout_path || candidate.silhouette_path || candidate.mask_path;
}

function toWebPath(path) {
  return `/${path}`.replace(/\/+/g, "/");
}

function setActiveStyle() {
  for (const button of els.styleGroup.querySelectorAll("button")) {
    const isActive = Boolean(state[button.dataset.style]);
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  }
}

function setActiveMode() {
  for (const button of els.modeGroup.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.mode === state.mode);
  }
}

function readabilityLabel() {
  if (state.readability >= 70) return "readable";
  if (state.readability <= 35) return "abstract";
  return "balanced";
}

function buildGenerationRequest() {
  const style = currentStyle();
  return {
    text: state.allCaps ? state.text.toUpperCase() : state.text,
    reference_font: {
      family: FONT_OPTIONS[state.font].label,
      style: style.label,
      bold: state.bold,
      italic: state.italic,
    },
    candidate_source: FONT_OPTIONS[state.font].candidatesUrl,
    non_font_dependent_controls: {
      readability: state.readability / 100,
      abstraction: (100 - state.readability) / 100,
      gaussian_blur_px: state.blur,
      render_mode: state.mode,
      all_caps: state.allCaps,
    },
    expected_backend_pipeline: [
      "load selected font candidate rankings",
      "choose goose cutouts or silhouettes for requested text",
      "compose glyph preview",
      "generate and return ttf font",
    ],
  };
}

function currentStyle() {
  if (state.bold && state.italic) return { label: "Bold Italic", weight: 800 };
  if (state.bold) return { label: "Bold", weight: 800 };
  if (state.italic) return { label: "Italic", weight: 500 };
  return { label: "Standard", weight: 500 };
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
