const IMAGE_NAMES = [
  "2024-07-04.png",
  "2024-07-09.png",
  "2024-07-16.png",
  "2024-07-29.png",
  "2024-07-31.png",
  "2025-04-27.png",
  "2025-05-10.png",
  "2025-06-14.png",
  "2025-08-21.png",
  "2025-08-31.png",
  "2025-11-08-1.png",
  "2025-11-08-2.png",
  "2025-12-06.png",
  "2025-12-07-1.png",
  "2025-12-07-2.png",
  "2025-12-18-2.png",
  "2025-12-18.png",
  "2025-12-19.png",
  "2025-12-23.png",
  "2026-01-08.png",
  "2026-02-16-1.png",
  "2026-02-16-2.png",
  "2026-02-22-1.png",
  "2026-02-22-2.png",
  "2026-02-23.png",
  "2026-03-04.png",
  "2026-03-05-1.png",
  "2026-03-05-2.png",
  "2026-03-05-3.png",
  "2026-03-08.png",
  "2026-03-15.png",
  "2026-03-19-1.png",
  "2026-03-19-2.png",
  "2026-03-20.png",
  "2026-04-11.png",
  "2026-04-19.png",
  "2026-04-20.png",
  "2026-04-21.png",
  "2026-04-23.png",
  "2026-04-28.png",
  "2026-04-30.png",
  "2026-05-02.png",
];

const TARGETS = {
  A: [0.78, 0.56, 0.05, 0.2, 2],
  B: [0.68, 0.7, 0, 0.72, 2],
  C: [0.76, 0.55, -0.05, 0.95, 1],
  D: [0.74, 0.68, 0, 0.78, 2],
  E: [0.62, 0.62, 0, 0.18, 2],
  F: [0.58, 0.5, 0, 0.18, 1],
  G: [0.8, 0.62, -0.04, 0.88, 1],
  H: [0.76, 0.66, 0, 0.2, 2],
  I: [0.28, 0.45, 0, 0.08, 1],
  J: [0.42, 0.48, 0.08, 0.58, 1],
  K: [0.74, 0.56, 0.12, 0.24, 2],
  L: [0.58, 0.5, 0, 0.18, 1],
  M: [0.95, 0.7, 0, 0.26, 2],
  N: [0.78, 0.62, 0.18, 0.22, 2],
  O: [0.82, 0.66, 0, 1, 1],
  P: [0.63, 0.6, 0, 0.56, 2],
  Q: [0.84, 0.66, 0.1, 0.92, 2],
  R: [0.72, 0.62, 0.12, 0.55, 2],
  S: [0.66, 0.58, -0.08, 0.9, 1],
  T: [0.72, 0.54, 0, 0.12, 1],
  U: [0.78, 0.6, 0, 0.72, 1],
  V: [0.78, 0.48, -0.05, 0.24, 1],
  W: [1.05, 0.64, 0, 0.3, 2],
  X: [0.78, 0.58, 0, 0.2, 2],
  Y: [0.78, 0.5, 0.08, 0.28, 1],
  Z: [0.72, 0.54, -0.08, 0.16, 1],
};

const PRESETS = {
  regular: { readability: 0.82, abstraction: 0.18, boldness: 0.48, slant: 0, width: 0.52, cursive: 0.12, mode: "photo", twoGeese: true },
  abstract: { readability: 0.32, abstraction: 0.86, boldness: 0.62, slant: -0.12, width: 0.62, cursive: 0.22, mode: "contour", twoGeese: true },
  italic: { readability: 0.74, abstraction: 0.26, boldness: 0.48, slant: 0.72, width: 0.5, cursive: 0.28, mode: "photo", twoGeese: true },
  bold: { readability: 0.76, abstraction: 0.2, boldness: 0.92, slant: 0.04, width: 0.7, cursive: 0.08, mode: "silhouette", twoGeese: true },
  cursive: { readability: 0.58, abstraction: 0.44, boldness: 0.44, slant: 0.38, width: 0.56, cursive: 0.88, mode: "photo", twoGeese: false },
};

const state = {
  library: [],
  text: "GOOSE TYPE",
  style: { ...PRESETS.regular },
};

const els = {
  row: document.querySelector("#glyphRow"),
  status: document.querySelector("#status"),
  metrics: document.querySelector("#metrics"),
  textInput: document.querySelector("#textInput"),
  presetGroup: document.querySelector("#presetGroup"),
  modeGroup: document.querySelector("#modeGroup"),
  twoGeese: document.querySelector("#twoGeese"),
};

for (const key of ["readability", "abstraction", "boldness", "slant", "width", "cursive"]) {
  els[key] = document.querySelector(`#${key}`);
  els[`${key}Value`] = document.querySelector(`#${key}Value`);
}

init();

async function init() {
  bindControls();
  state.library = await Promise.all(IMAGE_NAMES.map((name, index) => loadGoose(name, index)));
  state.library = state.library.filter(Boolean);
  els.status.textContent = `${state.library.length} geese analyzed`;
  render();
}

function bindControls() {
  els.textInput.addEventListener("input", () => {
    state.text = els.textInput.value;
    render();
  });

  for (const key of ["readability", "abstraction", "boldness", "slant", "width", "cursive"]) {
    els[key].addEventListener("input", () => {
      state.style[key] = Number(els[key].value);
      syncValues();
      render();
    });
  }

  els.twoGeese.addEventListener("input", () => {
    state.style.twoGeese = els.twoGeese.checked;
    render();
  });

  els.presetGroup.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-preset]");
    if (!button) return;
    applyPreset(button.dataset.preset);
  });

  els.modeGroup.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;
    state.style.mode = button.dataset.mode;
    setActive(els.modeGroup, "mode", state.style.mode);
    render();
  });
}

function applyPreset(name) {
  state.style = { ...PRESETS[name] };
  for (const key of ["readability", "abstraction", "boldness", "slant", "width", "cursive"]) {
    els[key].value = state.style[key];
  }
  els.twoGeese.checked = state.style.twoGeese;
  setActive(els.presetGroup, "preset", name);
  setActive(els.modeGroup, "mode", state.style.mode);
  syncValues();
  render();
}

function setActive(group, datasetKey, value) {
  for (const button of group.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset[datasetKey] === value);
  }
}

function syncValues() {
  for (const key of ["readability", "abstraction", "boldness", "slant", "width", "cursive"]) {
    els[`${key}Value`].textContent = Number(state.style[key]).toFixed(2);
  }
}

async function loadGoose(name, index) {
  const img = new Image();
  img.decoding = "async";
  img.src = `../goose_photos_square/${name}`;
  await img.decode().catch(() => null);
  if (!img.naturalWidth) return null;

  const derived = segmentImage(img);
  return {
    id: `goose_${String(index + 1).padStart(4, "0")}`,
    name,
    img,
    ...derived,
  };
}

function segmentImage(img) {
  const size = 220;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  canvas.width = size;
  canvas.height = size;
  ctx.drawImage(img, 0, 0, size, size);

  const pixels = ctx.getImageData(0, 0, size, size);
  const data = pixels.data;
  const bg = estimateBackground(data, size);
  let minX = size, minY = size, maxX = 0, maxY = 0;
  let area = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, perimeter = 0;

  const mask = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const distance = colorDistance([r, g, b], bg);
      const saturation = max3(r, g, b) - min3(r, g, b);
      const cx = (x / size - 0.5) * 2;
      const cy = (y / size - 0.5) * 2;
      const centerBias = 1 - Math.min(1, Math.hypot(cx, cy));
      const active = distance + saturation * 0.5 + centerBias * 26 > 54;
      if (!active) continue;
      mask[y * size + x] = 1;
      area++;
      sx += x;
      sy += y;
      sxx += x * x;
      syy += y * y;
      sxy += x * y;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  if (area < 20) {
    minX = minY = 0;
    maxX = maxY = size - 1;
    area = size * size;
  }

  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      if (!mask[y * size + x]) continue;
      if (!mask[y * size + x - 1] || !mask[y * size + x + 1] || !mask[(y - 1) * size + x] || !mask[(y + 1) * size + x]) {
        perimeter++;
      }
    }
  }

  const cx = sx / area;
  const cy = sy / area;
  const covXX = sxx / area - cx * cx;
  const covYY = syy / area - cy * cy;
  const covXY = sxy / area - cx * cy;
  const angle = 0.5 * Math.atan2(2 * covXY, covXX - covYY) * 180 / Math.PI;
  const width = Math.max(1, maxX - minX + 1);
  const height = Math.max(1, maxY - minY + 1);
  const bboxArea = width * height;
  const aspectRatio = width / height;
  const boldnessScore = clamp(area / bboxArea);
  const thinnessScore = clamp(perimeter / Math.max(area, 1) * 4.8);
  const curvatureScore = clamp((perimeter * perimeter) / (Math.max(area, 1) * 42));
  const slantScore = clamp(angle / 45, -1, 1);

  const photoUrl = makeDerivedUrl(canvas, pixels, mask, "photo");
  const silhouetteUrl = makeDerivedUrl(canvas, pixels, mask, "silhouette");

  return {
    photoUrl,
    silhouetteUrl,
    bbox: [minX / size, minY / size, width / size, height / size],
    aspectRatio,
    area,
    perimeter,
    centerOfMass: [cx / size, cy / size],
    orientationAngle: angle,
    boldnessScore,
    thinnessScore,
    curvatureScore,
    slantScore,
  };
}

function estimateBackground(data, size) {
  const samples = [];
  const points = [[4, 4], [size - 5, 4], [4, size - 5], [size - 5, size - 5], [size / 2, 4], [size / 2, size - 5]];
  for (const [x, y] of points) {
    const i = (Math.floor(y) * size + Math.floor(x)) * 4;
    samples.push([data[i], data[i + 1], data[i + 2]]);
  }
  return samples.reduce((acc, rgb) => acc.map((value, i) => value + rgb[i] / samples.length), [0, 0, 0]);
}

function makeDerivedUrl(canvas, pixels, mask, mode) {
  const out = new ImageData(new Uint8ClampedArray(pixels.data), pixels.width, pixels.height);
  for (let index = 0; index < mask.length; index++) {
    const i = index * 4;
    const active = mask[index];
    out.data[i + 3] = active ? 255 : 0;
    if (mode === "silhouette" && active) {
      out.data[i] = 22;
      out.data[i + 1] = 28;
      out.data[i + 2] = 29;
    }
  }
  const ctx = canvas.getContext("2d");
  ctx.putImageData(out, 0, 0);
  return canvas.toDataURL("image/png");
}

function render() {
  if (!state.library.length) return;
  els.row.innerHTML = "";
  const compositions = [];
  const chars = [...state.text.toUpperCase()].slice(0, 42);
  chars.forEach((char, index) => {
    if (char === " ") {
      const space = document.createElement("div");
      space.className = "glyph space";
      els.row.append(space);
      return;
    }
    const composition = generateGlyph(char, index);
    compositions.push(composition);
    els.row.append(renderGlyph(composition, index));
  });
  renderMetrics(compositions);
}

function generateGlyph(letter, index) {
  const target = getTarget(letter);
  const ranked = state.library
    .map((goose) => ({ goose, score: scoreGoose(goose, target, index) }))
    .sort((a, b) => b.score - a.score);

  const parts = state.style.twoGeese && (target.parts === 2 || state.style.abstraction > 0.72) ? 2 : 1;
  const offset = (index * 5 + Math.round(state.style.abstraction * 17)) % Math.min(9, ranked.length);
  const selected = ranked.slice(offset, offset + parts);
  if (selected.length < parts) selected.push(...ranked.slice(0, parts - selected.length));

  return {
    letter,
    target,
    score: selected.reduce((sum, item) => sum + item.score, 0) / selected.length,
    geese: selected.map((item, partIndex) => placeGoose(item.goose, target, partIndex, parts)),
  };
}

function getTarget(letter) {
  const [aspect, boldness, slant, curvature, parts] = TARGETS[letter] || [0.7, 0.5, 0, 0.5, 1];
  return { aspect, boldness, slant, curvature, parts };
}

function scoreGoose(goose, target, index) {
  const aspect = 1 - Math.min(Math.abs(goose.aspectRatio - target.aspect) / 2, 1);
  const boldness = 1 - Math.abs(goose.boldnessScore - (target.boldness * 0.55 + state.style.boldness * 0.45));
  const curvature = 1 - Math.abs(goose.curvatureScore - target.curvature);
  const slant = 1 - Math.min(Math.abs(goose.slantScore - state.style.slant) / 2, 1);
  const diversity = ((hash(goose.id + index) % 100) / 100) * 0.06;
  return state.style.readability * (aspect * 0.44 + curvature * 0.3 + boldness * 0.26)
    + state.style.abstraction * (0.28 + Math.abs(goose.slantScore) * 0.28 + goose.thinnessScore * 0.18 + diversity)
    + slant * 0.2;
}

function placeGoose(goose, target, partIndex, parts) {
  const pairOffset = parts === 2 ? (partIndex === 0 ? -18 : 18) : 0;
  const flip = parts === 2 && partIndex === 1;
  return {
    goose,
    x: 50 + pairOffset + state.style.cursive * partIndex * 8,
    y: 50 + state.style.cursive * partIndex * 5,
    scale: 0.88 + state.style.boldness * 0.16 + state.style.width * 0.1 - Math.abs(goose.aspectRatio - target.aspect) * 0.06,
    rotation: goose.orientationAngle * 0.16 + state.style.slant * 18 + target.slant * 12 + (parts === 2 ? pairOffset * 0.45 : 0),
    flip,
  };
}

function renderGlyph(composition, index) {
  const glyph = document.createElement("div");
  glyph.className = "glyph";
  const width = 94 + state.style.width * 76 + (composition.target.aspect - 0.7) * 36;
  glyph.style.setProperty("--glyph-width", `${Math.max(70, width)}px`);
  glyph.style.setProperty("--glyph-slant", `${state.style.slant * -7}deg`);

  composition.geese.forEach((placement) => {
    const img = document.createElement("img");
    img.className = `goosePart ${state.style.mode === "contour" ? "contour" : ""}`;
    img.alt = "";
    img.src = state.style.mode === "silhouette" ? placement.goose.silhouetteUrl : placement.goose.photoUrl;
    const flip = placement.flip ? -1 : 1;
    const cursiveShift = state.style.cursive * index * 1.6;
    img.style.transform = `translate(-50%, -50%) translate(${placement.x - 50 + cursiveShift}%, ${placement.y - 50}%) rotate(${placement.rotation}deg) scale(${placement.scale * flip}, ${placement.scale})`;
    glyph.append(img);
  });

  const label = document.createElement("span");
  label.className = "glyphLetter";
  label.textContent = composition.letter;
  glyph.append(label);
  return glyph;
}

function renderMetrics(compositions) {
  const averageScore = compositions.length
    ? compositions.reduce((sum, item) => sum + item.score, 0) / compositions.length
    : 0;
  const used = new Set(compositions.flatMap((item) => item.geese.map((part) => part.goose.id)));
  const averageSlant = compositions.length
    ? compositions.flatMap((item) => item.geese).reduce((sum, part) => sum + part.goose.slantScore, 0) / compositions.flatMap((item) => item.geese).length
    : 0;
  els.metrics.innerHTML = `
    <div class="metric"><strong>${averageScore.toFixed(2)}</strong>match score</div>
    <div class="metric"><strong>${used.size}</strong>unique geese used</div>
    <div class="metric"><strong>${state.style.mode}</strong>render mode</div>
    <div class="metric"><strong>${averageSlant.toFixed(2)}</strong>average goose slant</div>
  `;
}

function colorDistance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function max3(a, b, c) {
  return Math.max(a, b, c);
}

function min3(a, b, c) {
  return Math.min(a, b, c);
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function hash(value) {
  let out = 0;
  for (let index = 0; index < value.length; index++) {
    out = ((out << 5) - out + value.charCodeAt(index)) | 0;
  }
  return Math.abs(out);
}

