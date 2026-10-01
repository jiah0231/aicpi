import sharp from "sharp";

type Box = [number, number, number, number];
type Point = readonly [number, number];

export type GroundingColorEdge = "left" | "top" | "right" | "bottom";

type GroundingColorEdges = {
  /** ROI sides actually touched by selected matching pixels. */
  touchesRoiEdges: GroundingColorEdge[];
  /** Touched ROI sides that coincide with the source image boundary. */
  touchesSourceEdges: GroundingColorEdge[];
  /** Touched ROI sides with additional source pixels available beyond the ROI. */
  clippedRoiEdges: GroundingColorEdge[];
};

export type GroundingColorOptions = {
  /** Source-normalized pixel-edge coordinates, not crop-relative coordinates. */
  region: readonly [number, number, number, number];
  color: string;
  /** 0..1. Broadens the fixed HSV preset, or is normalized RGB distance for hex colors. */
  tolerance?: number;
  minAreaPixels?: number;
  selection?: "largest" | "all" | "point";
  /** Select only the component containing this source-normalized point. */
  point?: Point;
};

export type GroundingColorCandidate = GroundingColorEdges & {
  id: number;
  bbox: Box;
  pixelBbox: Box;
  areaPixels: number;
  /** Fraction of the component's bounding rectangle occupied by matching pixels. */
  matchingFraction: number;
  touchesRegionEdge: boolean;
};

export type GroundingColorResult = GroundingColorEdges & {
  status: "matched" | "no_match";
  sourceWidth: number;
  sourceHeight: number;
  region: Box;
  regionPixels: Box;
  color: string;
  tolerance: number;
  minAreaPixels: number;
  selection: "largest" | "all" | "point";
  method: string;
  matchingPixels: number;
  /** All matching pixels divided by ROI pixels, before component filtering. */
  matchingFraction: number;
  candidateCount: number;
  candidates: GroundingColorCandidate[];
  candidatesTruncated: boolean;
  /** IDs of the returned selected candidates; may be truncated for selection=all. */
  selectedIds: number[];
  selectedComponentCount: number;
  selectedAreaPixels: number;
  bbox: Box | null;
  touchesRegionEdge: boolean;
  pointSample?: {
    sourceNormalized: [number, number];
    sourcePixel: [number, number] | null;
    insideRegion: boolean;
    rgba: [number, number, number, number] | null;
    hex: string | null;
    hsv: { hue: number; saturation: number; value: number } | null;
    matchesRequestedColor: boolean;
    retainedComponentId: number | null;
  };
  selectionAssessment: {
    role: "pixel_measurement_only";
    establishesObjectIdentity: false;
    boundaryStatus: "no_selection" | "clipped_roi" | "tiny_component" | "measured";
    recommendation: string;
  };
  warnings: string[];
  rawPreview: Buffer;
  maskPreview: Buffer;
  previewWidth: number;
  previewHeight: number;
};

const MAX_SOURCE_PIXELS = 100_000_000;
const MAX_REGION_PIXELS = 8_000_000;
const MAX_CANDIDATES = 32;
const MAX_PREVIEW_SIDE = 1600;
const PRESETS: Record<string, { hue: number; halfWidth: number }> = {
  red: { hue: 0, halfWidth: 12 },
  orange: { hue: 30, halfWidth: 14 },
  yellow: { hue: 57, halfWidth: 14 },
  green: { hue: 120, halfWidth: 40 },
  cyan: { hue: 180, halfWidth: 24 },
  blue: { hue: 230, halfWidth: 32 },
  purple: { hue: 278, halfWidth: 30 },
  pink: { hue: 330, halfWidth: 22 },
  brown: { hue: 30, halfWidth: 18 },
};

function validateNormalized(values: readonly number[], length: number, name: string): void {
  if (!Array.isArray(values) || values.length !== length
    || values.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error(`${name} must contain ${length} finite source-normalized coordinates in [0, 1].`);
  }
}

function colorMatcher(color: string, tolerance: number): (r: number, g: number, b: number) => boolean {
  if (/^#[\da-f]{6}$/i.test(color)) {
    const target = [1, 3, 5].map((offset) => Number.parseInt(color.slice(offset, offset + 2), 16));
    // Root-mean-square RGB distance: zero tolerance is an exact byte match.
    const limit = 3 * (255 * tolerance) ** 2;
    return (r, g, b) => (r - target[0]) ** 2 + (g - target[1]) ** 2 + (b - target[2]) ** 2 <= limit;
  }
  const preset = PRESETS[color];
  if (!preset && !["black", "white", "gray"].includes(color)) {
    throw new Error("color must be black, white, gray, red, orange, yellow, green, cyan, blue, purple, pink, brown, or #RRGGBB.");
  }
  return (r, g, b) => {
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    const value = max / 255;
    const saturation = max === 0 ? 0 : delta / max;
    const blackCeiling = 0.18 + tolerance * 0.5;
    const whiteFloor = 0.90 - tolerance * 0.3;
    if (color === "black") return value <= blackCeiling;
    if (color === "white") return value >= whiteFloor && saturation <= 0.08 + tolerance * 0.5;
    // Gray keeps a fixed midtone band: increasing tolerance must not make an
    // already-matching gray disappear as black/white presets grow wider.
    if (color === "gray") return value > 0.25 && value < 0.85 && saturation <= 0.10 + tolerance * 0.4;
    if (saturation < Math.max(0.12, 0.35 - tolerance * 0.5) || value < Math.max(0.08, 0.22 - tolerance * 0.3)) return false;
    if (color === "brown" && value > 0.60 + tolerance * 0.2) return false;
    if (color === "pink" && value < 0.45 - tolerance * 0.2) return false;
    let hue = max === r ? (g - b) / delta : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
    hue = (hue * 60 + 360) % 360;
    const distance = Math.abs(hue - preset.hue);
    return Math.min(distance, 360 - distance) <= preset.halfWidth + tolerance * 40;
  };
}

function rgbToHsv(r: number, g: number, b: number): { hue: number; saturation: number; value: number } {
  const red = r / 255, green = g / 255, blue = b / 255;
  const max = Math.max(red, green, blue), min = Math.min(red, green, blue);
  const delta = max - min;
  let hue = 0;
  if (delta > 0) {
    if (max === red) hue = 60 * (((green - blue) / delta) % 6);
    else if (max === green) hue = 60 * ((blue - red) / delta + 2);
    else hue = 60 * ((red - green) / delta + 4);
  }
  if (hue < 0) hue += 360;
  return { hue, saturation: max === 0 ? 0 : delta / max, value: max };
}

function normalizedBox(box: Box, width: number, height: number): Box {
  return [box[0] / width, box[1] / height, box[2] / width, box[3] / height];
}

function matchingEdges(box: Box | null, roi: Box, sourceWidth: number, sourceHeight: number): GroundingColorEdges {
  const touchesRoiEdges: GroundingColorEdge[] = [];
  const touchesSourceEdges: GroundingColorEdge[] = [];
  const clippedRoiEdges: GroundingColorEdge[] = [];
  if (box) {
    const names = ["left", "top", "right", "bottom"] as const;
    const sourceEdges = [0, 0, sourceWidth, sourceHeight];
    for (let index = 0; index < names.length; index += 1) {
      // Bounding extrema come from actual matching pixels, not an inferred shape.
      if (box[index] !== roi[index]) continue;
      const edge = names[index];
      touchesRoiEdges.push(edge);
      if (roi[index] === sourceEdges[index]) touchesSourceEdges.push(edge);
      else clippedRoiEdges.push(edge);
    }
  }
  return { touchesRoiEdges, touchesSourceEdges, clippedRoiEdges };
}

/**
 * Local deterministic color measurement on the clean source image. No detector,
 * dilation, erosion, hole filling, threshold retries, file writes, or ground truth.
 * Components use 8-neighbor connectivity; pixels with alpha < 128 are ignored.
 */
export async function analyzeGroundingColor(imageBytes: Buffer, options: GroundingColorOptions): Promise<GroundingColorResult> {
  validateNormalized(options.region, 4, "region");
  if (options.region[0] >= options.region[2] || options.region[1] >= options.region[3]) {
    throw new Error("region must have x1 < x2 and y1 < y2.");
  }
  const tolerance = options.tolerance ?? 0.12;
  if (!Number.isFinite(tolerance) || tolerance < 0 || tolerance > 1) throw new Error("tolerance must be in [0, 1].");
  const minAreaPixels = options.minAreaPixels ?? 3;
  if (!Number.isSafeInteger(minAreaPixels) || minAreaPixels < 1) throw new Error("minAreaPixels must be a positive integer.");
  const selection = options.selection ?? "largest";
  if (!["largest", "all", "point"].includes(selection)) throw new Error("selection must be largest, all, or point.");
  if (options.point !== undefined) validateNormalized(options.point, 2, "point");
  if (selection === "point" && options.point === undefined) throw new Error("selection=point requires a source-normalized point.");
  const color = options.color.trim().toLowerCase();
  const matches = colorMatcher(color, tolerance);
  const image = sharp(imageBytes, { limitInputPixels: MAX_SOURCE_PIXELS });
  const metadata = await image.metadata();
  const sourceWidth = metadata.width;
  const sourceHeight = metadata.height;
  if (!sourceWidth || !sourceHeight) throw new Error("Cannot measure image dimensions.");
  if ((metadata.pages ?? 1) > 1) throw new Error("Color measurement requires a single still image.");
  // Intersecting pixels are included. Report the actual pixel-edge ROI explicitly.
  const sourcePixel = (coordinate: number, dimension: number) => {
    const pixel = coordinate * dimension;
    return Math.abs(pixel - Math.round(pixel)) < 1e-8 ? Math.round(pixel) : pixel;
  };
  const left = Math.floor(sourcePixel(options.region[0], sourceWidth));
  const top = Math.floor(sourcePixel(options.region[1], sourceHeight));
  const right = Math.min(sourceWidth, Math.ceil(sourcePixel(options.region[2], sourceWidth)));
  const bottom = Math.min(sourceHeight, Math.ceil(sourcePixel(options.region[3], sourceHeight)));
  const width = right - left;
  const height = bottom - top;
  const regionPixels: Box = [left, top, right, bottom];
  const pixels = width * height;
  if (pixels > MAX_REGION_PIXELS) throw new Error(`Color region exceeds ${MAX_REGION_PIXELS} pixels; choose a smaller region around the target.`);
  const raw = await image.extract({ left, top, width, height }).toColourspace("srgb").ensureAlpha().raw().toBuffer();
  const mask = new Uint8Array(pixels);
  const labels = new Int32Array(pixels);
  // A bounded queue avoids recursive flood-fill overflowing on a solid image.
  const queue = new Int32Array(pixels);
  let matchingPixels = 0;
  for (let index = 0; index < pixels; index += 1) {
    const offset = index * 4;
    if (raw[offset + 3] >= 128 && matches(raw[offset], raw[offset + 1], raw[offset + 2])) {
      mask[index] = 1;
      matchingPixels += 1;
    }
  }
  const candidates: GroundingColorCandidate[] = [];
  let candidateCount = 0;
  let componentId = 0;
  let largest: GroundingColorCandidate | undefined;
  let pointCandidate: GroundingColorCandidate | undefined;
  const pointX = options.point ? Math.floor(sourcePixel(options.point[0], sourceWidth)) - left : -1;
  const pointY = options.point ? Math.floor(sourcePixel(options.point[1], sourceHeight)) - top : -1;
  const pointIndex = pointX >= 0 && pointX < width && pointY >= 0 && pointY < height ? pointY * width + pointX : -1;
  const allBounds: Box = [right, bottom, left, top];
  let allArea = 0;
  for (let start = 0; start < pixels; start += 1) {
    if (!mask[start] || labels[start]) continue;
    componentId += 1;
    labels[start] = componentId;
    queue[0] = start;
    let tail = 1;
    let x1 = width;
    let y1 = height;
    let x2 = 0;
    let y2 = 0;
    for (let head = 0; head < tail; head += 1) {
      const index = queue[head];
      const y = Math.floor(index / width);
      const x = index - y * width;
      x1 = Math.min(x1, x);
      y1 = Math.min(y1, y);
      x2 = Math.max(x2, x + 1);
      y2 = Math.max(y2, y + 1);
      for (let ny = Math.max(0, y - 1); ny <= Math.min(height - 1, y + 1); ny += 1) {
        for (let nx = Math.max(0, x - 1); nx <= Math.min(width - 1, x + 1); nx += 1) {
          const neighbor = ny * width + nx;
          if (!mask[neighbor] || labels[neighbor]) continue;
          labels[neighbor] = componentId;
          queue[tail++] = neighbor;
        }
      }
    }
    if (tail < minAreaPixels) {
      for (let index = 0; index < tail; index += 1) labels[queue[index]] = -componentId;
      continue;
    }
    const pixelBbox: Box = [left + x1, top + y1, left + x2, top + y2];
    const edges = matchingEdges(pixelBbox, regionPixels, sourceWidth, sourceHeight);
    const candidate: GroundingColorCandidate = {
      id: componentId,
      bbox: normalizedBox(pixelBbox, sourceWidth, sourceHeight),
      pixelBbox,
      areaPixels: tail,
      matchingFraction: tail / ((x2 - x1) * (y2 - y1)),
      touchesRegionEdge: edges.touchesRoiEdges.length > 0,
      ...edges,
    };
    candidateCount += 1;
    allArea += tail;
    allBounds[0] = Math.min(allBounds[0], pixelBbox[0]);
    allBounds[1] = Math.min(allBounds[1], pixelBbox[1]);
    allBounds[2] = Math.max(allBounds[2], pixelBbox[2]);
    allBounds[3] = Math.max(allBounds[3], pixelBbox[3]);
    if (!largest || tail > largest.areaPixels) largest = candidate;
    if (pointIndex >= 0 && labels[pointIndex] === componentId) pointCandidate = candidate;
    // Keep diagnostic output bounded even for highly speckled source images.
    const rank = candidates.findIndex((item) => item.areaPixels < tail);
    if (rank >= 0) candidates.splice(rank, 0, candidate);
    else if (candidates.length < MAX_CANDIDATES) candidates.push(candidate);
    if (candidates.length > MAX_CANDIDATES) candidates.pop();
  }
  const selected = selection === "point" ? pointCandidate : largest;
  if (selection === "point" && selected && !candidates.some((item) => item.id === selected.id)) {
    if (candidates.length === MAX_CANDIDATES) candidates.pop();
    candidates.push(selected);
  }
  const selectedAreaPixels = selection === "all" ? allArea : selected?.areaPixels ?? 0;
  const selectedComponentCount = selection === "all" ? candidateCount : selected ? 1 : 0;
  const selectedBox = selectedAreaPixels ? selection === "all" ? allBounds : selected!.pixelBbox : null;
  const selectedEdges = matchingEdges(selectedBox, regionPixels, sourceWidth, sourceHeight);
  const touchesRegionEdge = selectedEdges.touchesRoiEdges.length > 0;
  const selectedIds = selection === "all" ? candidates.map((item) => item.id) : selected ? [selected.id] : [];
  const warnings: string[] = [];
  if (selectedAreaPixels === 0) warnings.push(selection === "point"
    ? "No retained matching component contains the point. No nearest component was substituted."
    : "No matching component meets minAreaPixels. No box was invented or threshold changed.");
  if (selectedEdges.clippedRoiEdges.length > 0) warnings.push(`Selected pixels touch the ROI boundary at ${selectedEdges.clippedRoiEdges.join(", ")}; the region may clip the target. Inspect a wider region on those sides before using this box.`);
  if (selectedEdges.touchesSourceEdges.length > 0) warnings.push(`Selected pixels touch the source image boundary at ${selectedEdges.touchesSourceEdges.join(", ")}. No further source pixels are available on those sides.`);
  if (selection === "all" && candidateCount > 1) warnings.push("The box encloses disconnected color components and the gaps between them. Verify they all belong to the requested target.");
  if (candidateCount > MAX_CANDIDATES) warnings.push(`Only ${MAX_CANDIDATES} candidate summaries are returned; selection=all still includes every retained component.`);
  const pointSample = options.point ? (() => {
    if (pointIndex < 0) return {
      sourceNormalized: [...options.point] as [number, number], sourcePixel: null, insideRegion: false,
      rgba: null, hex: null, hsv: null, matchesRequestedColor: false, retainedComponentId: null,
    };
    const offset = pointIndex * 4;
    const rgba: [number, number, number, number] = [raw[offset], raw[offset + 1], raw[offset + 2], raw[offset + 3]];
    return {
      sourceNormalized: [...options.point] as [number, number],
      sourcePixel: [left + pointX, top + pointY] as [number, number],
      insideRegion: true,
      rgba,
      hex: `#${rgba.slice(0, 3).map((value) => value.toString(16).padStart(2, "0")).join("")}`,
      hsv: rgbToHsv(rgba[0], rgba[1], rgba[2]),
      matchesRequestedColor: mask[pointIndex] === 1,
      retainedComponentId: labels[pointIndex] > 0 ? labels[pointIndex] : null,
    };
  })() : undefined;
  const boundaryStatus = selectedAreaPixels === 0
    ? "no_selection" as const
    : selectedEdges.clippedRoiEdges.length > 0
      ? "clipped_roi" as const
      : selectedAreaPixels < 9
        ? "tiny_component" as const
        : "measured" as const;
  const recommendation = boundaryStatus === "no_selection"
    ? "The requested point/component was not selected. Do not infer absence or move the target to a nearby dark patch; inspect the reported point sample and visible structure."
    : boundaryStatus === "clipped_roi"
      ? "The selected pixels cross an artificial ROI edge, so this box is incomplete for boundary use. Expand only the reported clipped sides and verify the same physical part."
      : boundaryStatus === "tiny_component"
        ? "This is a very small pixel sample. It may measure a real detail or noise; confirm that it is visibly connected to the requested part before using its bounds."
        : "The selected component has measurable pixel bounds, but color membership alone does not establish which object or anatomical part it belongs to.";
  const marked = Buffer.from(raw);
  for (let index = 0; index < pixels; index += 1) {
    if (labels[index] <= 0 || (selection !== "all" && labels[index] !== selected?.id)) continue;
    const offset = index * 4;
    marked[offset] = Math.round(raw[offset] * 0.4);
    marked[offset + 1] = Math.round(raw[offset + 1] * 0.4 + 255 * 0.6);
    marked[offset + 2] = Math.round(raw[offset + 2] * 0.4 + 255 * 0.6);
  }
  const longSide = Math.max(width, height);
  const scale = Math.min(8, Math.max(1, 768 / longSide), MAX_PREVIEW_SIDE / longSide);
  const previewWidth = Math.max(1, Math.round(width * scale));
  const previewHeight = Math.max(1, Math.round(height * scale));
  const preview = (bytes: Buffer) => sharp(bytes, { raw: { width, height, channels: 4 } })
    .resize(previewWidth, previewHeight, { kernel: "nearest" }).png();
  const rawPreview = await preview(raw).toBuffer();
  let maskImage = preview(marked);
  if (selectedBox) {
    const x = (selectedBox[0] - left) / width * previewWidth;
    const y = (selectedBox[1] - top) / height * previewHeight;
    const boxWidth = (selectedBox[2] - selectedBox[0]) / width * previewWidth;
    const boxHeight = (selectedBox[3] - selectedBox[1]) / height * previewHeight;
    const svg = `<svg width="${previewWidth}" height="${previewHeight}"><rect x="${x + 0.5}" y="${y + 0.5}" width="${Math.max(0, boxWidth - 1)}" height="${Math.max(0, boxHeight - 1)}" fill="none" stroke="#ff00ff" stroke-width="1"/></svg>`;
    maskImage = maskImage.composite([{ input: Buffer.from(svg) }]);
  }
  return {
    status: selectedAreaPixels ? "matched" : "no_match",
    sourceWidth, sourceHeight,
    region: normalizedBox([left, top, right, bottom], sourceWidth, sourceHeight),
    regionPixels,
    color, tolerance, minAreaPixels, selection,
    method: "Fixed HSV color presets; #RRGGBB uses normalized RMS RGB distance. Alpha >= 128; 8-connected components; no morphology or automatic threshold changes. Cyan = selected pixels; magenta = their exact enclosing pixel-edge bbox. Color membership is not object identity.",
    matchingPixels, matchingFraction: matchingPixels / pixels,
    candidateCount, candidates, candidatesTruncated: candidateCount > candidates.length,
    selectedIds, selectedComponentCount, selectedAreaPixels,
    bbox: selectedBox ? normalizedBox(selectedBox, sourceWidth, sourceHeight) : null,
    touchesRegionEdge, ...selectedEdges,
    ...(pointSample ? { pointSample } : {}),
    selectionAssessment: { role: "pixel_measurement_only", establishesObjectIdentity: false, boundaryStatus, recommendation },
    warnings,
    rawPreview, maskPreview: await maskImage.toBuffer(), previewWidth, previewHeight,
  };
}
