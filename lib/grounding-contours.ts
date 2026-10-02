import { setImmediate as yieldImmediate } from "node:timers/promises";
import sharp, { type OverlayOptions } from "sharp";
import type { GroundingViewBox } from "./grounding-views";

export const GROUNDING_CONTOUR_LIMITS = {
  sourceBytes: 64_000_000, sourcePixels: 100_000_000, regionPixels: 1_000_000,
  candidates: 3, components: 4096, panelSide: 640,
} as const;

export type GroundingContourOptions = {
  region: readonly number[];
  coarseBox: readonly number[];
  /** Filters component BOUNDS only; this is not a foreground segmentation seed. */
  point?: readonly number[];
  lowThreshold?: number;
  highThreshold?: number;
};

function box(value: readonly number[], name: string): GroundingViewBox {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(v => Number.isFinite(v) && v >= 0 && v <= 1)
    || value[0] >= value[2] || value[1] >= value[3]) throw new Error(`${name} must be ordered source-normalized edges in [0, 1].`);
  return [...value] as GroundingViewBox;
}

export function validateGroundingContourOptions(input: GroundingContourOptions) {
  const region = box(input.region, "region"), coarseBox = box(input.coarseBox, "coarseBox");
  if (coarseBox[0] < region[0] || coarseBox[1] < region[1] || coarseBox[2] > region[2] || coarseBox[3] > region[3]) {
    throw new Error("region must contain coarseBox; include background context around the full visible target.");
  }
  const lowThreshold = input.lowThreshold ?? 20, highThreshold = input.highThreshold ?? 50;
  if (![lowThreshold, highThreshold].every(v => Number.isFinite(v) && v >= 1 && v <= 255) || lowThreshold >= highThreshold) {
    throw new Error("Require 1 <= lowThreshold < highThreshold <= 255.");
  }
  const point = input.point;
  if (point !== undefined && (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)
    || point[0] < coarseBox[0] || point[0] >= coarseBox[2] || point[1] < coarseBox[1] || point[1] >= coarseBox[3])) {
    throw new Error("point must contain two source-normalized coordinates inside coarseBox's half-open bounds.");
  }
  return { region, coarseBox, lowThreshold, highThreshold, ...(point ? { point: [...point] as [number, number] } : {}) };
}

/** Deterministic Canny-like edge components, NOT filled object segmentation. No I/O or models. */
export async function proposeGroundingContours(bytes: Buffer, input: GroundingContourOptions, signal?: AbortSignal) {
  const options = validateGroundingContourOptions(input);
  const checkpoint = async () => { await yieldImmediate(); signal?.throwIfAborted(); };
  signal?.throwIfAborted();
  if (bytes.length > GROUNDING_CONTOUR_LIMITS.sourceBytes) throw new Error("Contour source exceeds 64 MB.");
  const source = sharp(bytes, { limitInputPixels: GROUNDING_CONTOUR_LIMITS.sourcePixels, failOn: "error" });
  const metadata = await source.metadata();
  if (!metadata.width || !metadata.height || (metadata.pages ?? 1) !== 1) throw new Error("Contours require a single still image with known dimensions.");
  const sourceWidth = metadata.width, sourceHeight = metadata.height;
  const snap = (v: number, d: number) => Math.abs(v * d - Math.round(v * d)) < 1e-8 ? Math.round(v * d) : v * d;
  const left = Math.floor(snap(options.region[0], sourceWidth)), top = Math.floor(snap(options.region[1], sourceHeight));
  const right = Math.min(sourceWidth, Math.ceil(snap(options.region[2], sourceWidth)));
  const bottom = Math.min(sourceHeight, Math.ceil(snap(options.region[3], sourceHeight)));
  const width = right - left, height = bottom - top, n = width * height;
  if (width < 7 || height < 7) throw new Error("Contour ROI must be at least 7 pixels on each side.");
  if (n > GROUNDING_CONTOUR_LIMITS.regionPixels) throw new Error("Contour ROI exceeds 1 million pixels; choose a smaller ROI. No silent downsampling.");
  // Preserve stored source orientation. Exclude alpha-affected neighborhoods from evidence.
  const rgba = await source.extract({ left, top, width, height }).toColourspace("srgb").ensureAlpha().raw().toBuffer();
  const gray = new Float32Array(n), smooth = new Float32Array(n), magnitude = new Float32Array(n);
  const direction = new Uint8Array(n), eligible = new Uint8Array(n), edges = new Uint8Array(n);
  const queue = new Int32Array(n);
  for (let y = 0; y < height; y++) {
    if (y % 32 === 0) await checkpoint();
    for (let x = 0; x < width; x++) {
      const i = y * width + x, p = i * 4;
      gray[i] = .2126 * rgba[p] + .7152 * rgba[p + 1] + .0722 * rgba[p + 2];
    }
  }
  // Fixed 3x3 binomial blur. Do not erode/open/dilate or merge disconnected parts.
  for (let y = 1; y < height - 1; y++) {
    if (y % 32 === 0) await checkpoint();
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      smooth[i] = (gray[i - width - 1] + 2 * gray[i - width] + gray[i - width + 1]
        + 2 * gray[i - 1] + 4 * gray[i] + 2 * gray[i + 1]
        + gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1]) / 16;
    }
  }
  for (let y = 2; y < height - 2; y++) {
    if (y % 32 === 0) await checkpoint();
    for (let x = 2; x < width - 2; x++) {
      const i = y * width + x;
      let opaque = true;
      for (let dy = -2; dy <= 2 && opaque; dy++) for (let dx = -2; dx <= 2; dx++) {
        if (rgba[((y + dy) * width + x + dx) * 4 + 3] !== 255) { opaque = false; break; }
      }
      if (!opaque) continue;
      eligible[i] = 1;
      const gx = -smooth[i - width - 1] + smooth[i - width + 1] - 2 * smooth[i - 1] + 2 * smooth[i + 1] - smooth[i + width - 1] + smooth[i + width + 1];
      const gy = -smooth[i - width - 1] - 2 * smooth[i - width] - smooth[i - width + 1] + smooth[i + width - 1] + 2 * smooth[i + width] + smooth[i + width + 1];
      magnitude[i] = Math.hypot(gx, gy) / 4;
      const angle = (Math.atan2(gy, gx) * 180 / Math.PI + 180) % 180;
      direction[i] = angle < 22.5 || angle >= 157.5 ? 0 : angle < 67.5 ? 1 : angle < 112.5 ? 2 : 3;
    }
  }
  let tail = 0;
  const offsets = [1, width + 1, width, width - 1];
  for (let y = 3; y < height - 3; y++) {
    if (y % 32 === 0) await checkpoint();
    for (let x = 3; x < width - 3; x++) {
      const i = y * width + x, d = offsets[direction[i]], m = magnitude[i];
      if (!eligible[i] || !eligible[i - d] || !eligible[i + d] || m < options.lowThreshold || m < magnitude[i - d] || m < magnitude[i + d]) continue;
      edges[i] = m >= options.highThreshold ? 2 : 1;
      if (edges[i] === 2) queue[tail++] = i;
    }
  }
  const neighbors = [-width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1];
  for (let head = 0; head < tail; head++) {
    if (head % 4096 === 0) await checkpoint();
    for (const d of neighbors) { const j = queue[head] + d; if (edges[j] === 1) { edges[j] = 2; queue[tail++] = j; } }
  }
  const edgePreview = Buffer.alloc(n);
  for (let i = 0; i < n; i++) { if (i % 16384 === 0) await checkpoint(); if (edges[i] === 2) edgePreview[i] = 255; }
  type Candidate = { bbox: GroundingViewBox; edgePixels: number; rankScore: number; coarseBoxIoU: number; touchesAnalysisBorder: boolean };
  const candidates: Candidate[] = [];
  let componentCount = 0, truncated = false;
  const c = options.coarseBox, coarseArea = (c[2] - c[0]) * (c[3] - c[1]);
  for (let i = 0; i < n; i++) {
    if (i % 16384 === 0) await checkpoint();
    if (edges[i] !== 2) continue;
    if (++componentCount > GROUNDING_CONTOUR_LIMITS.components) { truncated = true; break; }
    let minX = width, minY = height, maxX = 0, maxY = 0;
    tail = 1; queue[0] = i; edges[i] = 3;
    for (let head = 0; head < tail; head++) {
      if (head % 4096 === 0) await checkpoint();
      const p = queue[head], x = p % width, y = Math.floor(p / width);
      minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
      for (const d of neighbors) { const j = p + d; if (edges[j] === 2) { edges[j] = 3; queue[tail++] = j; } }
    }
    if (tail < 8 || minX === maxX || minY === maxY) continue;
    const bbox: GroundingViewBox = [(left + minX) / sourceWidth, (top + minY) / sourceHeight, (left + maxX + 1) / sourceWidth, (top + maxY + 1) / sourceHeight];
    const point = options.point;
    if (point && (point[0] < bbox[0] || point[0] >= bbox[2] || point[1] < bbox[1] || point[1] >= bbox[3])) continue;
    const intersection = Math.max(0, Math.min(c[2], bbox[2]) - Math.max(c[0], bbox[0])) * Math.max(0, Math.min(c[3], bbox[3]) - Math.max(c[1], bbox[1]));
    if (!intersection) continue;
    const area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
    const coarseBoxIoU = intersection / (coarseArea + area - intersection);
    const touchesAnalysisBorder = minX <= 3 || minY <= 3 || maxX >= width - 4 || maxY >= height - 4;
    candidates.push({ bbox, edgePixels: tail, coarseBoxIoU, touchesAnalysisBorder, rankScore: coarseBoxIoU * (touchesAnalysisBorder ? .5 : 1) });
    candidates.sort((a, b) => b.rankScore - a.rankScore || a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0]);
    if (candidates.length > GROUNDING_CONTOUR_LIMITS.candidates) candidates.pop();
  }
  const proposals = candidates.map((candidate, index) => ({ id: `E${index + 1}`, ...candidate }));
  const original = await sharp(rgba, { raw: { width, height, channels: 4 } }).flatten({ background: "white" }).png().toBuffer();
  const edgeImage = await sharp(edgePreview, { raw: { width, height, channels: 1 } }).png().toBuffer();
  const colors = ["#ff8c00", "#00b8ff", "#d946ef"];
  const rectSvg = (b: readonly number[], color: string, dashed = false) => `<rect x="${b[0] * sourceWidth - left}" y="${b[1] * sourceHeight - top}" width="${(b[2] - b[0]) * sourceWidth}" height="${(b[3] - b[1]) * sourceHeight}" fill="none" stroke="${color}" stroke-width="1" ${dashed ? 'stroke-dasharray="4 3"' : ""}/>`;
  const overlay = await sharp(original).composite([{ input: Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${rectSvg(c, "#ffffff", true)}${proposals.map((p, i) => rectSvg(p.bbox, colors[i])).join("")}</svg>`) }]).png().toBuffer();
  const scale = Math.min(1, GROUNDING_CONTOUR_LIMITS.panelSide / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale)), h = Math.max(1, Math.round(height * scale));
  const cell = Math.max(300, w), gap = 12, title = 30, canvasWidth = cell * 3 + gap * 4, canvasHeight = h + title + gap * 2;
  const composites: OverlayOptions[] = [], panels: Array<{ label: string; rect: GroundingViewBox; derived: boolean; overlay: boolean }> = [];
  const labels = ["ORIGINAL ROI", "DERIVED EDGES (not mask)", "CANDIDATES E1 orange E2 blue E3 purple"];
  for (const [index, image] of [original, edgeImage, overlay].entries()) {
    await checkpoint();
    const x = gap + index * (cell + gap), imageLeft = x + Math.floor((cell - w) / 2), imageTop = gap + title;
    composites.push({ input: await sharp(image).resize(w, h).png().toBuffer(), left: imageLeft, top: imageTop });
    composites.push({ input: Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${cell}" height="${title}"><text x="2" y="20" font-family="sans-serif" font-size="11" fill="#172033">${labels[index]}</text></svg>`), left: x, top: gap });
    panels.push({ label: labels[index], rect: [imageLeft, imageTop, imageLeft + w, imageTop + h], derived: index !== 0, overlay: index === 2 });
  }
  const image = await sharp({ create: { width: canvasWidth, height: canvasHeight, channels: 3, background: "#e8edf3" } }).composite(composites).png().toBuffer();
  await checkpoint();
  return { image, width: canvasWidth, height: canvasHeight, sourceWidth, sourceHeight,
    region: [left / sourceWidth, top / sourceHeight, right / sourceWidth, bottom / sourceHeight] as GroundingViewBox,
    regionPixels: [left, top, right, bottom] as GroundingViewBox, options, panels, candidates: proposals,
    status: proposals.length ? "candidates_require_review" : "unresolved", componentsTruncated: truncated,
    warnings: ["Candidates bound connected edge pixels, not filled objects or verified closed contours. No candidate is automatically selected or saved.",
      "rankScore is coarse-box IoU with a border penalty, not identity confidence. Point filtering tests component bounds only, not foreground membership.",
      "Blur, weak contrast, texture, shadows, alpha and open/disconnected boundaries can omit thin tails or include neighboring objects. Detached parts are not merged. No hidden geometry is recovered.",
      "Three ROI-border pixels are excluded. Border-touching candidates may be clipped; inspect a wider ROI. Alpha-affected neighborhoods are excluded; original transparency is displayed on white.",
      "Compare every candidate with original pixels, including protrusions beyond its box. Candidate colors are E1 orange, E2 blue, E3 purple; dashed white is the unchanged coarse box.",
      ...(truncated ? ["Component budget reached; returned candidates are incomplete and scanning order may bias the retained set."] : [])],
  };
}
