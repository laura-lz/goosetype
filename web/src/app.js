const CANDIDATE_LIBRARY = {
  label: "GooseType",
  family: "Arial, Helvetica, sans-serif",
  candidatesUrl: "/data/font_candidates/goosetype_candidates.json",
};

const state = {
  text: "goosetype",
  italic: false,
  bold: false,
  readability: 68,
  smoothing: 0,
  mode: "photo",
};

const candidateCache = {};

const els = {
  textInput: document.querySelector("#textInput"),
  styleGroup: document.querySelector("#styleGroup"),
  modeGroup: document.querySelector("#modeGroup"),
  readabilitySlider: document.querySelector("#readabilitySlider"),
  blurSlider: document.querySelector("#blurSlider"),
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
loadCandidates();
render();

function bindControls() {
  els.textInput.addEventListener("input", () => {
    state.text = els.textInput.value;
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
    state.smoothing = Number(els.blurSlider.value);
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

async function loadCandidates() {
  if (candidateCache.library) {
    render();
    return;
  }

  els.candidateStatus.textContent = "Loading GooseType candidate library...";
  try {
    const response = await fetch(CANDIDATE_LIBRARY.candidatesUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    candidateCache.library = normalizeCandidateData(await response.json());
    const letters = Object.keys(candidateCache.library).length;
    const firstRanked = Object.values(candidateCache.library)[0] || [];
    els.candidateStatus.textContent = `${letters} letters loaded, ${firstRanked.length} current-mask candidates per letter.`;
  } catch (error) {
    els.candidateStatus.textContent = `Could not load candidate JSON from ${CANDIDATE_LIBRARY.candidatesUrl}. Run the page from the repo root server.`;
    console.error(error);
  }
  render();
}

function normalizeCandidateData(data) {
  if (!data || data.schema !== "goosetype-candidates-v2") {
    return data;
  }

  const normalized = {};
  const geese = data.geese || {};
  for (const [letter, ranked] of Object.entries(data.letters || {})) {
    normalized[letter] = ranked.map((candidate) => ({
      ...(geese[candidate.goose_id] || {}),
      ...candidate,
      letter,
    }));
  }
  return normalized;
}

function render() {
  const style = currentStyle();
  const text = state.text;

  els.previewTitle.textContent = `GooseType ${style.label}`;
  els.readabilityLabel.textContent = `${state.readability}%`;
  els.readabilityModeLabel.textContent = readabilityLabel();
  els.exportStatus.textContent = "Frontend request builder ready. Backend font generation endpoint pending.";

  renderGooseText(text || "goosetype");
}

function renderGooseText(text) {
  const candidates = candidateCache.library;
  els.fontPreview.innerHTML = "";
  els.fontPreview.classList.toggle("silhouette", state.mode === "silhouette");

  if (!candidates) {
    const loading = document.createElement("p");
    loading.className = "fontPreviewFallback";
    loading.textContent = text;
    loading.style.fontFamily = CANDIDATE_LIBRARY.family;
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
    const letter = char;
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

  const image = renderCandidateImage(candidate, `${letter} goose candidate`);

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
    const image = renderCandidateImage(item.candidate, `${item.letter} selected candidate`);
    const text = document.createElement("span");
    text.textContent = `${item.letter} ${item.candidate.score.toFixed(2)}`;
    card.append(image, text);
    els.candidateStrip.append(card);
  }
}

function renderCandidateImage(candidate, altText) {
  const src = toWebPath(candidateImagePath(candidate));
  if (state.mode === "silhouette" && state.smoothing > 0) {
    const canvas = document.createElement("canvas");
    canvas.className = "smoothedSilhouette";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", altText);
    canvas.style.transform = candidateTransform(candidate);
    drawSmoothedSilhouette(canvas, src, false);
    return canvas;
  }

  const image = document.createElement("img");
  image.alt = altText;
  image.src = src;
  image.loading = "lazy";
  image.decoding = "async";
  image.style.transform = candidateTransform(candidate);
  return image;
}

function candidateTransform(candidate) {
  return candidate.flip_x ? "scaleX(-1)" : "";
}

function drawSmoothedSilhouette(canvas, src, flipX) {
  const source = new Image();
  source.onload = () => {
    const width = source.naturalWidth || source.width || 1;
    const height = source.naturalHeight || source.height || 1;
    canvas.width = width;
    canvas.height = height;

    const filtered = document.createElement("canvas");
    filtered.width = width;
    filtered.height = height;
    const filteredContext = filtered.getContext("2d", { willReadFrequently: true });
    filteredContext.clearRect(0, 0, width, height);
    filteredContext.filter = `blur(${state.smoothing}px)`;
    if (flipX) {
      filteredContext.translate(width, 0);
      filteredContext.scale(-1, 1);
    }
    filteredContext.drawImage(source, 0, 0, width, height);

    const imageData = filteredContext.getImageData(0, 0, width, height);
    const data = imageData.data;
    const threshold = Math.max(14, 96 - state.smoothing * 7);
    for (let index = 0; index < data.length; index += 4) {
      if (data[index + 3] >= threshold) {
        data[index] = 0;
        data[index + 1] = 0;
        data[index + 2] = 0;
        data[index + 3] = 255;
      } else {
        data[index] = 0;
        data[index + 1] = 0;
        data[index + 2] = 0;
        data[index + 3] = 0;
      }
    }

    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    context.putImageData(imageData, 0, 0);
  };
  source.src = src;
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
    text: state.text,
    style: {
      preset: style.label,
      bold: state.bold,
      italic: state.italic,
    },
    candidate_source: CANDIDATE_LIBRARY.candidatesUrl,
    controls: {
      readability: state.readability / 100,
      abstraction: (100 - state.readability) / 100,
      silhouette_smoothing_px: state.smoothing,
      render_mode: state.mode,
      all_caps: false,
    },
    expected_backend_pipeline: [
      "load image-derived goose candidate rankings",
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
