import sharp, { type OverlayOptions } from "sharp";
import type { GroundingViewBox } from "./grounding-views";

export type GroundingImageOperation =
  | { kind: "edges" }
  | { kind: "blur"; sigma?: number }
  | { kind: "median"; size?: number }
  | { kind: "sharpen"; sigma?: number }
  | { kind: "contrast"; gain?: number }
  | { kind: "threshold"; level?: number };

export const GROUNDING_PROCESSING_LIMITS = {
  sourcePixels: 100_000_000, sourceBytes: 64_000_000, regionPixels: 4_000_000,
  operations: 3, panelSide: 720,
} as const;

/** Validate independently of the tool schema; no silent clamping or implicit operation chains. */
export function validateGroundingImageOperations(input: readonly GroundingImageOperation[]): GroundingImageOperation[] {
  if (!Array.isArray(input) || input.length < 1 || input.length > GROUNDING_PROCESSING_LIMITS.operations) {
    throw new Error("Choose 1–3 independent image operations, only those needed for the boundary question.");
  }
  const number = (value: unknown, fallback: number, min: number, max: number, name: string) => {
    const resolved = value === undefined ? fallback : value;
    if (typeof resolved !== "number" || !Number.isFinite(resolved) || resolved < min || resolved > max) {
      throw new Error(`${name} must be finite and in [${min}, ${max}].`);
    }
    return resolved;
  };
  return input.map<GroundingImageOperation>((op) => {
    if (!op || typeof op !== "object") throw new Error("Each image operation needs a kind.");
    const fields: Record<string, string[]> = { edges: [], blur: ["sigma"], median: ["size"], sharpen: ["sigma"], contrast: ["gain"], threshold: ["level"] };
    const allowed = Object.prototype.hasOwnProperty.call(fields, op.kind) ? fields[op.kind] : undefined;
    if (!allowed) throw new Error("Unknown image operation; choose edges, blur, median, sharpen, contrast or threshold.");
    if (Object.keys(op).some((key) => key !== "kind" && !allowed.includes(key))) throw new Error(`Unexpected parameter for ${op.kind}.`);
    switch (op.kind) {
      case "edges": return { kind: op.kind };
      case "blur": case "sharpen": return { kind: op.kind, sigma: number(op.sigma, 1, .3, 3, "sigma") };
      case "contrast": return { kind: op.kind, gain: number(op.gain, 1.5, .5, 3, "gain") };
      case "threshold": {
        const level = number(op.level, 128, 0, 255, "level");
        if (!Number.isInteger(level)) throw new Error("threshold level must be an integer.");
        return { kind: op.kind, level };
      }
      case "median": {
        const size = number(op.size, 3, 3, 7, "size");
        if (!Number.isInteger(size) || size % 2 !== 1) throw new Error("median size must be 3, 5 or 7.");
        return { kind: op.kind, size };
      }
      default: throw new Error("Unsupported image operation.");
    }
  });
}

/** Sobel magnitude, clamped border sampling, no inferred segments or object boundaries. */
async function sobel(bytes: Buffer, width: number, height: number, signal?: AbortSignal): Promise<Buffer> {
  const gray = await sharp(bytes).greyscale().raw().toBuffer();
  const out = Buffer.alloc(width * height);
  const sample = (x: number, y: number) => gray[Math.max(0, Math.min(height - 1, y)) * width + Math.max(0, Math.min(width - 1, x))];
  for (let y = 0; y < height; y++) {
    if (y % 64 === 0) signal?.throwIfAborted();
    for (let x = 0; x < width; x++) {
      const a = sample(x - 1, y - 1), b = sample(x, y - 1), c = sample(x + 1, y - 1);
      const d = sample(x - 1, y), f = sample(x + 1, y);
      const g = sample(x - 1, y + 1), h = sample(x, y + 1), i = sample(x + 1, y + 1);
      const gx = -a + c - 2 * d + 2 * f - g + i;
      const gy = -a - 2 * b - c + g + 2 * h + i;
      out[y * width + x] = Math.min(255, Math.round(Math.hypot(gx, gy) / 4));
    }
  }
  return sharp(out, { raw: { width, height, channels: 1 } }).png().toBuffer();
}

/** Original and independent derived panels share an exact, unrotated source ROI. No I/O or models. */
export async function processGroundingImage(
  bytes: Buffer,
  options: { region: readonly number[]; operations: readonly GroundingImageOperation[] },
  signal?: AbortSignal,
) {
  signal?.throwIfAborted();
  const operations = validateGroundingImageOperations(options.operations);
  const region = options.region;
  if (!Array.isArray(region) || region.length !== 4 || !region.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)
    || region[0] >= region[2] || region[1] >= region[3]) throw new Error("region must be ordered source-normalized edges in [0, 1].");
  if (bytes.length > GROUNDING_PROCESSING_LIMITS.sourceBytes) throw new Error("Source exceeds the 64 MB image-processing limit.");
  const source = sharp(bytes, { limitInputPixels: GROUNDING_PROCESSING_LIMITS.sourcePixels, failOn: "error" });
  const metadata = await source.metadata();
  if (!metadata.width || !metadata.height) throw new Error("Cannot determine source dimensions.");
  if ((metadata.pages ?? 1) !== 1) throw new Error("Image processing requires one still image.");
  const sourceWidth = metadata.width, sourceHeight = metadata.height;
  const pixel = (v: number, dimension: number) => {
    const p = v * dimension;
    return Math.abs(p - Math.round(p)) < 1e-8 ? Math.round(p) : p;
  };
  const left = Math.floor(pixel(region[0], sourceWidth)), top = Math.floor(pixel(region[1], sourceHeight));
  const right = Math.min(sourceWidth, Math.ceil(pixel(region[2], sourceWidth)));
  const bottom = Math.min(sourceHeight, Math.ceil(pixel(region[3], sourceHeight)));
  const width = right - left, height = bottom - top;
  if (width < 1 || height < 1) throw new Error("Image-processing region has no pixels after edge snapping; choose a larger ROI.");
  if (width * height > GROUNDING_PROCESSING_LIMITS.regionPixels) throw new Error("Image-processing ROI exceeds 4 million pixels; choose a smaller target region.");
  // Match the grounding source's stored orientation (never auto-rotate). Alpha
  // is composited on white for every panel; transparent pixels are not evidence.
  const raw = await source.extract({ left, top, width, height }).toColourspace("srgb").flatten({ background: "white" }).png().toBuffer();
  const rendered: Array<{ image: Buffer; label: string; operation?: GroundingImageOperation }> = [{ image: raw, label: "ORIGINAL ROI (display only)" }];
  for (const op of operations) {
    signal?.throwIfAborted();
    let image: Buffer;
    let pipeline = sharp(raw);
    if (op.kind === "edges") image = await sobel(raw, width, height, signal);
    else {
      if (op.kind === "blur") pipeline = pipeline.blur(op.sigma!);
      if (op.kind === "median") pipeline = pipeline.median(op.size!);
      if (op.kind === "sharpen") pipeline = pipeline.sharpen({ sigma: op.sigma! });
      if (op.kind === "contrast") pipeline = pipeline.linear(op.gain!, 128 * (1 - op.gain!));
      if (op.kind === "threshold") pipeline = pipeline.greyscale().threshold(op.level!);
      image = await pipeline.png().toBuffer();
    }
    const parameter = Object.entries(op).filter(([key]) => key !== "kind").map(([key, value]) => `${key}=${value}`).join(" ");
    rendered.push({ image, label: `DERIVED ${op.kind}${parameter ? ` ${parameter}` : ""}`, operation: op });
  }
  const scale = Math.min(1, GROUNDING_PROCESSING_LIMITS.panelSide / Math.max(width, height));
  const panelWidth = Math.max(360, Math.round(width * scale)), panelHeight = Math.max(1, Math.round(height * scale));
  const gap = 12, title = 30;
  const canvasWidth = panelWidth * 2 + gap * 3;
  const canvasHeight = Math.ceil(rendered.length / 2) * (panelHeight + title + gap) + gap;
  const composites: OverlayOptions[] = [];
  const panels: Array<{ label: string; rect: GroundingViewBox; operation?: GroundingImageOperation }> = [];
  for (const [index, panel] of rendered.entries()) {
    signal?.throwIfAborted();
    const x = gap + (index % 2) * (panelWidth + gap), y = gap + Math.floor(index / 2) * (panelHeight + title + gap);
    const resized = await sharp(panel.image).resize({ width: panelWidth, height: panelHeight, fit: "inside", withoutEnlargement: true }).png().toBuffer({ resolveWithObject: true });
    const imageLeft = x + Math.floor((panelWidth - resized.info.width) / 2), imageTop = y + title + Math.floor((panelHeight - resized.info.height) / 2);
    // Labels contain only validated enum names and finite bounded numbers.
    const label = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${panelWidth}" height="${title}"><text x="4" y="22" font-family="sans-serif" font-size="18" fill="#172033">${panel.label}</text></svg>`);
    composites.push({ input: label, left: x, top: y }, { input: resized.data, left: imageLeft, top: imageTop });
    panels.push({ label: panel.label, rect: [imageLeft, imageTop, imageLeft + resized.info.width, imageTop + resized.info.height], ...(panel.operation ? { operation: panel.operation } : {}) });
  }
  const image = await sharp({ create: { width: canvasWidth, height: canvasHeight, channels: 3, background: "#e8edf3" } }).composite(composites).png().toBuffer();
  signal?.throwIfAborted();
  return { image, width: canvasWidth, height: canvasHeight, sourceWidth, sourceHeight,
    region: [left / sourceWidth, top / sourceHeight, right / sourceWidth, bottom / sourceHeight] as GroundingViewBox,
    regionPixels: [left, top, right, bottom] as GroundingViewBox, operations, panels,
    warnings: ["Derived pixels may erase or invent apparent edges; compare with the original. Processing does not establish identity or recover occluded detail.",
      "Filters use ROI-local border handling. Crop edges and transparent pixels composited on white are not object boundaries. Previews may be downsampled or compressed."],
  };
}
