import type { GroundingViewDescriptor, GroundingViewBox } from "./grounding-views";

export type GroundingInspectionRegion = { modality: GroundingViewDescriptor["modality"]; region: GroundingViewBox; displayBounds?: [number, number] };
export type GroundingInspectionTest = { condition: string; observable: string };

/** Match the crop renderer's outward-rounded extraction, not JSON/prose equality. */
export function groundingInspectionPixels(region: GroundingViewBox, width: number, height: number): GroundingViewBox {
  const edge = (value: number, size: number) => {
    const pixel = value * size;
    return Math.abs(pixel - Math.round(pixel)) < 1e-8 ? Math.round(pixel) : pixel;
  };
  const left = Math.min(width - 1, Math.max(0, Math.floor(edge(region[0], width))));
  const top = Math.min(height - 1, Math.max(0, Math.floor(edge(region[1], height))));
  return [left, top, Math.min(width, Math.max(left + 1, Math.ceil(edge(region[2], width)))),
    Math.min(height, Math.max(top + 1, Math.ceil(edge(region[3], height))))];
}

/** Conservative: only a previously adequately resolved, non-derived exposure can
 * suppress rendering. Downsampled evidence never blocks a higher-detail crop. */
export function redundantGroundingRegions(requests: GroundingInspectionRegion[], previous: GroundingViewDescriptor[]) {
  return requests.map((request) => previous.find((view) => {
    if (view.modality !== request.modality || view.derived) return false;
    const old = view.region.map((edge, i) => Math.round(edge * (i % 2 ? view.sourceHeight : view.sourceWidth)));
    const rect = view.displayRect ?? [0, 0, view.width, view.height];
    const current = groundingInspectionPixels(request.region, view.sourceWidth, view.sourceHeight);
    // The renderer caps its long side at 1600. A smaller crop can reveal detail
    // lost in a large overview, whereas rerendering the same capped span cannot.
    const bounds = request.displayBounds ?? [1600, 1600];
    const requestedResolution = Math.min(1, bounds[0] / (current[2] - current[0]), bounds[1] / (current[3] - current[1]));
    const previousX = (rect[2] - rect[0]) / (old[2] - old[0]);
    const previousY = (rect[3] - rect[1]) / (old[3] - old[1]);
    // At most one display-pixel rounding difference per axis. A thin strip
    // must not turn that allowance into permission to discard long-axis detail.
    if ((requestedResolution - previousX) * (current[2] - current[0]) > 1 + 1e-9
      || (requestedResolution - previousY) * (current[3] - current[1]) > 1 + 1e-9) return false;
    const area = (current[2] - current[0]) * (current[3] - current[1]);
    const overlap = Math.max(0, Math.min(old[2], current[2]) - Math.max(old[0], current[0]))
      * Math.max(0, Math.min(old[3], current[3]) - Math.max(old[1], current[1]));
    return overlap / area >= .98;
  }));
}
