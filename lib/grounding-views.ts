import { randomUUID } from "node:crypto";
import sharp, { type OverlayOptions } from "sharp";

export type GroundingViewBox = [number, number, number, number];

export type GroundingSourceReuse = {
  relation: "exact" | "contained_rerender" | "near_duplicate";
  previousViewId: string;
  coveredFraction: number;
  newSourcePixels: number;
  iou: number;
  previousMagnification: number;
  currentMagnification: number;
  displayScaleRatio: number;
  similarViewCount: number;
};

export interface GroundingViewDescriptor {
  id: string;
  modality: "visible" | "infrared" | "depth";
  /** Exact extracted source pixel edges, normalized against the full source. */
  region: GroundingViewBox;
  sourceWidth: number;
  sourceHeight: number;
  /** Dimensions of the full returned image, including any padding or labels. */
  width: number;
  height: number;
  /** Image content edges within the returned image; excludes labels/padding. */
  displayRect?: GroundingViewBox;
  decorations?: "none" | "grid" | "hypothesis" | "all";
  label?: string;
  /** Geometry remains source-mapped; derived appearance is never identity evidence. */
  derived?: { kind: "image_processing"; operation: string; role: "measurement_only"; establishesObjectIdentity: false };
}

function positiveDimension(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer.`);
}

function validBox(box: readonly number[], name: string): void {
  if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)
    || box[0] >= box[2] || box[1] >= box[3]) {
    throw new Error(`${name} must contain four finite edges [left, top, right, bottom] with positive area.`);
  }
}

function contained(box: readonly number[], bounds: readonly number[], name: string): void {
  if (box[0] < bounds[0] || box[1] < bounds[1] || box[2] > bounds[2] || box[3] > bounds[3]) {
    throw new Error(`${name} crosses the image content boundary; labels and padding are not source pixels.`);
  }
}

function cloneDescriptor(view: GroundingViewDescriptor): GroundingViewDescriptor {
  return {
    ...view,
    region: [...view.region],
    ...(view.displayRect ? { displayRect: [...view.displayRect] as GroundingViewBox } : {}),
    ...(view.derived ? { derived: { ...view.derived } } : {}),
  };
}

function sourcePixelBox(
  view: Pick<GroundingViewDescriptor, "region" | "sourceWidth" | "sourceHeight">,
): GroundingViewBox {
  return [
    Math.round(view.region[0] * view.sourceWidth),
    Math.round(view.region[1] * view.sourceHeight),
    Math.round(view.region[2] * view.sourceWidth),
    Math.round(view.region[3] * view.sourceHeight),
  ];
}

function boxArea(box: GroundingViewBox): number {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

function intersectionArea(left: GroundingViewBox, right: GroundingViewBox): number {
  return Math.max(0, Math.min(left[2], right[2]) - Math.max(left[0], right[0]))
    * Math.max(0, Math.min(left[3], right[3]) - Math.max(left[1], right[1]));
}

function viewMagnification(view: Pick<GroundingViewDescriptor, "region" | "sourceWidth" | "sourceHeight" | "width" | "height" | "displayRect">): number {
  const pixels = sourcePixelBox(view);
  const rect = view.displayRect ?? [0, 0, view.width, view.height];
  const x = (rect[2] - rect[0]) / Math.max(1, pixels[2] - pixels[0]);
  const y = (rect[3] - rect[1]) / Math.max(1, pixels[3] - pixels[1]);
  return Math.min(x, y);
}

/** Metadata only. A registry belongs to one active record, so old IDs cannot map a new record. */
export class GroundingViewRegistry {
  private readonly prefix = randomUUID();
  private readonly views = new Map<string, GroundingViewDescriptor>();

  register(input: Omit<GroundingViewDescriptor, "id">): GroundingViewDescriptor {
    positiveDimension(input.sourceWidth, "sourceWidth");
    positiveDimension(input.sourceHeight, "sourceHeight");
    positiveDimension(input.width, "width");
    positiveDimension(input.height, "height");
    if (!["visible", "infrared", "depth"].includes(input.modality)) throw new Error("Unknown source modality.");
    validBox(input.region, "region");
    contained(input.region, [0, 0, 1, 1], "region");
    if (input.displayRect) {
      validBox(input.displayRect, "displayRect");
      contained(input.displayRect, [0, 0, input.width, input.height], "displayRect");
    }
    const id = `view-${this.prefix}-${this.views.size + 1}`;
    const view = cloneDescriptor({ ...input, id });
    this.views.set(id, view);
    return cloneDescriptor(view);
  }

  get(id: string): GroundingViewDescriptor | undefined {
    const view = this.views.get(id);
    return view ? cloneDescriptor(view) : undefined;
  }

  list(): GroundingViewDescriptor[] {
    return [...this.views.values()].map(cloneDescriptor);
  }

  private availableViewIds(modality?: GroundingViewDescriptor["modality"]): string {
    const views = [...this.views.values()].filter((view) => !modality || view.modality === modality);
    const recent = views.slice(-6).reverse();
    const scope = modality ? `${modality} ` : "";
    if (!recent.length) return `No ${scope}viewIds are registered for the current record.`;
    const limit = views.length > recent.length ? ` (most recent ${recent.length} of ${views.length})` : "";
    return `Available ${scope}viewIds${limit}: ${recent.map((view) => `${view.id} (${view.modality})`).join(", ")}.`;
  }

  private requireView(id: string): GroundingViewDescriptor {
    const view = this.views.get(id);
    if (!view) throw new Error(`Unknown or stale viewId. Use the view that supplied the measured coordinates for the current record. ${this.availableViewIds()}`);
    return view;
  }

  private requireVisibleView(id: string): void {
    const view = this.requireView(id);
    if (view.modality !== "visible") {
      throw new Error(`viewId ${view.id} is ${view.modality}, not visible. ${view.modality}-to-visible registration is unsupported; equal image dimensions do not establish alignment. Remeasure the coordinates on a visible view for this record. ${this.availableViewIds("visible")}`);
    }
  }

  findEquivalent(input: Pick<GroundingViewDescriptor, "modality" | "region" | "sourceWidth" | "sourceHeight">): GroundingViewDescriptor | undefined {
    const view = [...this.views.values()].find((candidate) => candidate.modality === input.modality
      && candidate.sourceWidth === input.sourceWidth && candidate.sourceHeight === input.sourceHeight
      && candidate.region.every((edge, index) => Math.abs(edge - input.region[index]) < 1e-12));
    return view ? cloneDescriptor(view) : undefined;
  }

  /**
   * Compare a requested render with one similarly sized prior local view. A
   * full-frame overview is intentionally ignored for a much smaller crop: the
   * crop may expose source detail that was downsampled in the overview. This is
   * advisory metadata only and never blocks another rendering.
   */
  sourceReuse(input: Omit<GroundingViewDescriptor, "id">): GroundingSourceReuse | undefined {
    const currentPixels = sourcePixelBox(input);
    const currentArea = boxArea(currentPixels);
    if (!currentArea) return;
    const candidates = [...this.views.values()].flatMap((candidate) => {
      if (candidate.modality !== input.modality
        || candidate.sourceWidth !== input.sourceWidth
        || candidate.sourceHeight !== input.sourceHeight) return [];
      const previousPixels = sourcePixelBox(candidate);
      const previousArea = boxArea(previousPixels);
      const areaRatio = previousArea / currentArea;
      // Prevent the initial overview from making every useful focus crop look
      // redundant, while retaining exact/full-frame and similarly sized views.
      if (areaRatio > 4 || areaRatio < 0.25) return [];
      const overlap = intersectionArea(currentPixels, previousPixels);
      const coveredFraction = overlap / currentArea;
      const union = currentArea + previousArea - overlap;
      const iou = union > 0 ? overlap / union : 0;
      const exact = currentPixels.every((edge, index) => edge === previousPixels[index]);
      const relation = exact
        ? "exact" as const
        : currentArea === overlap
          ? "contained_rerender" as const
          : iou >= 0.85
            ? "near_duplicate" as const
            : undefined;
      if (!relation) return [];
      return [{ candidate, relation, coveredFraction, iou, overlap, areaRatio }];
    });
    if (!candidates.length) return;
    candidates.sort((left, right) => right.coveredFraction - left.coveredFraction
      || right.iou - left.iou
      || Math.abs(Math.log(left.areaRatio)) - Math.abs(Math.log(right.areaRatio)));
    const best = candidates[0];
    const previousMagnification = viewMagnification(best.candidate);
    const currentMagnification = viewMagnification(input);
    return {
      relation: best.relation,
      previousViewId: best.candidate.id,
      coveredFraction: best.coveredFraction,
      newSourcePixels: currentArea - best.overlap,
      iou: best.iou,
      previousMagnification,
      currentMagnification,
      displayScaleRatio: previousMagnification > 0 ? currentMagnification / previousMagnification : 1,
      similarViewCount: candidates.length,
    };
  }

  /** Maps within the view's own source modality; this does not register IR/depth to visible. */
  toSource(id: string, box: readonly number[], space: "view_pixels" | "view_normalized"): GroundingViewBox {
    const view = this.requireView(id);
    if (space !== "view_pixels" && space !== "view_normalized") throw new Error("Unknown view coordinate space.");
    validBox(box, "bbox");
    const rect = view.displayRect ?? [0, 0, view.width, view.height];
    // Compare normalized edges before conversion to avoid rejecting a mathematically exact boundary
    // merely because (pixel / dimension) * dimension has floating-point roundoff.
    const bounds = space === "view_normalized"
      ? [rect[0] / view.width, rect[1] / view.height, rect[2] / view.width, rect[3] / view.height]
      : rect;
    contained(box, bounds, "bbox");
    const xSpan = bounds[2] - bounds[0];
    const ySpan = bounds[3] - bounds[1];
    const region = view.region;
    const map = (edge: number, axis: 0 | 1): number => {
      if (edge === bounds[axis]) return region[axis];
      if (edge === bounds[axis + 2]) return region[axis + 2];
      return region[axis] + ((edge - bounds[axis]) / (axis === 0 ? xSpan : ySpan)) * (region[axis + 2] - region[axis]);
    };
    return [map(box[0], 0), map(box[1], 1), map(box[2], 0), map(box[3], 1)];
  }

  /** Maps within the view's own source modality; this does not register IR/depth to visible. */
  toSourcePoint(id: string, point: readonly number[], space: "view_pixels" | "view_normalized"): [number, number] {
    const view = this.requireView(id);
    if (space !== "view_pixels" && space !== "view_normalized") throw new Error("Unknown view coordinate space.");
    if (!Array.isArray(point) || point.length !== 2 || point.some((value) => !Number.isFinite(value))) {
      throw new Error("point must contain two finite coordinates.");
    }
    const rect = view.displayRect ?? [0, 0, view.width, view.height];
    const bounds = space === "view_normalized"
      ? [rect[0] / view.width, rect[1] / view.height, rect[2] / view.width, rect[3] / view.height]
      : rect;
    if (point[0] < bounds[0] || point[1] < bounds[1] || point[0] > bounds[2] || point[1] > bounds[3]) {
      throw new Error("point crosses the image content boundary; labels and padding are not source pixels.");
    }
    const region = view.region;
    const map = (value: number, axis: 0 | 1) => region[axis]
      + ((value - bounds[axis]) / (bounds[axis + 2] - bounds[axis])) * (region[axis + 2] - region[axis]);
    return [map(point[0], 0), map(point[1], 1)];
  }

  /** Visible-source boxes for saving or color analysis require visible measurement provenance. */
  toVisibleSource(id: string, box: readonly number[], space: "view_pixels" | "view_normalized"): GroundingViewBox {
    this.requireVisibleView(id);
    return this.toSource(id, box, space);
  }

  /** Visible-source points for color sampling require visible measurement provenance. */
  toVisibleSourcePoint(id: string, point: readonly number[], space: "view_pixels" | "view_normalized"): [number, number] {
    this.requireVisibleView(id);
    return this.toSourcePoint(id, point, space);
  }
}

function escapeXml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" })[char]!);
}

export interface GroundingComparison {
  image: Buffer;
  width: number;
  height: number;
  overviewRect: GroundingViewBox;
  panels: Array<{ label: string; rect: GroundingViewBox }>;
}

export type GroundingOverviewAnnotation = {
  label: string;
  region: GroundingViewBox;
  kind: "roi" | "object";
};

/** Geometry of declared object boxes only; an inspection ROI is not an object. */
export function groundingCandidateGeometry(candidates: Array<{ region: GroundingViewBox; bbox?: GroundingViewBox }>, sourceWidth: number, sourceHeight: number) {
  positiveDimension(sourceWidth, "sourceWidth");
  positiveDimension(sourceHeight, "sourceHeight");
  const items = candidates.map((candidate, index) => {
    validBox(candidate.region, "region");
    contained(candidate.region, [0, 0, 1, 1], "region");
    const id = String.fromCharCode(65 + index);
    if (!candidate.bbox) return { id, region: candidate.region };
    validBox(candidate.bbox, "bbox");
    contained(candidate.bbox, [0, 0, 1, 1], "bbox");
    const [left, top, right, bottom] = candidate.bbox;
    return { id, region: candidate.region, objectBbox: candidate.bbox,
      objectCenter: [(left + right) / 2, (top + bottom) / 2] as [number, number],
      objectSizePixels: [(right - left) * sourceWidth, (bottom - top) * sourceHeight] as [number, number] };
  });
  const declared = items.filter((item) => item.objectCenter);
  const ordered = (axis: number) => [...declared].sort((a, b) => a.objectCenter![axis] - b.objectCenter![axis]).map((item) => item.id);
  const tied = (axis: number) => declared.flatMap((item, index) => declared.slice(index + 1)
    .filter((other) => Math.abs(item.objectCenter![axis] - other.objectCenter![axis]) < 1e-12)
    .map((other) => [item.id, other.id]));
  return { candidates: items, leftToRight: ordered(0), topToBottom: ordered(1),
    tiedX: tied(0), tiedY: tied(1), missingObjectBoxes: items.filter((item) => !item.objectCenter).map((item) => item.id),
    note: "Only model-declared object boxes are sorted. ROI centers, panel/discovery order and magnification are not object rank. Ties have no unique rank. This does not verify identity, completeness or query interpretation." };
}

/** Four contextual ROIs straddling the proposal; clipping never changes its edges. */
export function groundingBoundaryRegions(bbox: GroundingViewBox, sourceWidth: number, sourceHeight: number) {
  validBox(bbox, "bbox");
  contained(bbox, [0, 0, 1, 1], "bbox");
  positiveDimension(sourceWidth, "sourceWidth");
  positiveDimension(sourceHeight, "sourceHeight");
  const [left, top, right, bottom] = bbox;
  const dx = Math.max(4 / sourceWidth, (right - left) * 0.12);
  const dy = Math.max(4 / sourceHeight, (bottom - top) * 0.12);
  return ([
    { edge: "top", axis: "y", position: top, requested: [left - dx, top - dy, right + dx, top + dy] },
    { edge: "bottom", axis: "y", position: bottom, requested: [left - dx, bottom - dy, right + dx, bottom + dy] },
    { edge: "left", axis: "x", position: left, requested: [left - dx, top - dy, left + dx, bottom + dy] },
    { edge: "right", axis: "x", position: right, requested: [right - dx, top - dy, right + dx, bottom + dy] },
  ] as const).map(({ requested, ...item }) => ({ ...item,
    region: requested.map((value) => Math.max(0, Math.min(1, value))) as GroundingViewBox,
    contextClipped: requested.some((value) => value < 0 || value > 1),
    outsideSourcePixels: item.axis === "x"
      ? (item.edge === "left" ? left : 1 - right) * sourceWidth
      : (item.edge === "top" ? top : 1 - bottom) * sourceHeight,
  }));
}

/** A bounded contact sheet; the per-call panel limit does not limit subsequent views or crops. */
export async function buildGroundingComparison(
  overview: Buffer,
  candidates: Array<{ label: string; image: Buffer; edgeMarker?: { axis: "x" | "y"; fraction: number } }>,
  annotations: GroundingOverviewAnnotation[] = [],
): Promise<GroundingComparison> {
  if (candidates.length < 1 || candidates.length > 4) throw new Error("A comparison accepts 1 to 4 candidates per call; request another comparison for more candidates.");
  const width = 1600;
  const padding = 16;
  const gap = 16;
  const labelHeight = 32;
  const overviewHeight = 340;
  const columns = candidates.length === 1 ? 1 : 2;
  const rows = Math.ceil(candidates.length / columns);
  const panelHeight = rows === 1 ? 540 : 500;
  const cellWidth = Math.floor((width - padding * 2 - gap * (columns - 1)) / columns);
  const height = padding * 2 + labelHeight + overviewHeight + gap + rows * (labelHeight + panelHeight) + (rows - 1) * gap;
  const composites: OverlayOptions[] = [];

  async function addPanel(bytes: Buffer, label: string, left: number, top: number, cellWidth: number, cellHeight: number, enlarge: boolean): Promise<GroundingViewBox> {
    // Candidate magnification makes small parts comparable; their source coordinates remain unchanged.
    const resized = await sharp(bytes, { limitInputPixels: 100_000_000, animated: false })
      .resize({ width: cellWidth, height: cellHeight, fit: "inside", withoutEnlargement: !enlarge })
      .png().toBuffer({ resolveWithObject: true });
    const imageLeft = left + Math.floor((cellWidth - resized.info.width) / 2);
    const imageTop = top + labelHeight + Math.floor((cellHeight - resized.info.height) / 2);
    const displayedLabel = [...label].slice(0, 90).join("");
    const title = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${cellWidth}" height="${labelHeight}"><text x="4" y="23" font-family="sans-serif" font-size="20" fill="#172033">${escapeXml(displayedLabel)}</text></svg>`);
    composites.push({ input: title, left, top }, { input: resized.data, left: imageLeft, top: imageTop });
    return [imageLeft, imageTop, imageLeft + resized.info.width, imageTop + resized.info.height];
  }

  const overviewRect = await addPanel(overview, "Overview", padding, padding, width - padding * 2, overviewHeight, false);
  if (annotations.length > 8) throw new Error("At most 8 overview annotations are supported.");
  const annotationSvg = annotations.map((annotation) => {
    validBox(annotation.region, "overview annotation");
    contained(annotation.region, [0, 0, 1, 1], "overview annotation");
    const [l, t, r, b] = annotation.region;
    const w = overviewRect[2] - overviewRect[0], h = overviewRect[3] - overviewRect[1];
    const x = overviewRect[0] + l * w, y = overviewRect[1] + t * h;
    const color = annotation.kind === "object" ? "#ff9d00" : "#00bde8";
    return `<g fill="none" stroke="${color}" stroke-width="2"><rect x="${x}" y="${y}" width="${(r - l) * w}" height="${(b - t) * h}" ${annotation.kind === "roi" ? 'stroke-dasharray="5 4"' : ""}/><circle cx="${x + (r - l) * w / 2}" cy="${y + (b - t) * h / 2}" r="3"/></g><text x="${Math.min(x + 2, overviewRect[2] - 60)}" y="${Math.max(overviewRect[1] + 14, y + 14)}" font-size="13" font-family="sans-serif" fill="${color}" stroke="#172033" stroke-width=".5">${escapeXml(annotation.label.slice(0, 80))}</text>`;
  }).join("");
  if (annotationSvg) composites.push({ input: Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${annotationSvg}</svg>`), left: 0, top: 0 });
  const panels: GroundingComparison["panels"] = [];
  for (const [index, candidate] of candidates.entries()) {
    const label = `${String.fromCharCode(65 + index)}: ${candidate.label}`;
    const left = padding + (index % columns) * (cellWidth + gap);
    const top = padding + labelHeight + overviewHeight + gap + Math.floor(index / columns) * (labelHeight + panelHeight + gap);
    const rect = await addPanel(candidate.image, label, left, top, cellWidth, panelHeight, true);
    if (candidate.edgeMarker) {
      const { axis, fraction } = candidate.edgeMarker;
      if (!Number.isFinite(fraction) || fraction < 0 || fraction > 1) throw new Error("Edge marker must lie within its panel's source ROI.");
      const p = axis === "x" ? rect[0] + fraction * (rect[2] - rect[0]) : rect[1] + fraction * (rect[3] - rect[1]);
      // Triangles sit outside the pixels; no line hides a thin boundary/part.
      const points = axis === "x" ? `${p - 4},${rect[1] - 6} ${p + 4},${rect[1] - 6} ${p},${rect[1] - 1}`
        : `${rect[0] - 6},${p - 4} ${rect[0] - 6},${p + 4} ${rect[0] - 1},${p}`;
      composites.push({ input: Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><polygon points="${points}" fill="#172033"/></svg>`), left: 0, top: 0 });
    }
    panels.push({ label, rect });
  }
  const image = await sharp({ create: { width, height, channels: 3, background: "#e8edf3" } })
    .composite(composites).png().toBuffer();
  return { image, width, height, overviewRect, panels };
}
