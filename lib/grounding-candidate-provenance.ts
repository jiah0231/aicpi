import type { GroundingConstraintCandidate, GroundingConstraintBox } from "./grounding-constraints";

/** A lookup owned by the current record, never descriptors supplied by a model. */
export type GroundingCandidateProvenanceContext = {
  get(id: string): {
    id: string;
    modality: "visible" | "infrared" | "depth";
    region: GroundingConstraintBox;
    sourceWidth: number;
    sourceHeight: number;
  } | undefined;
};

export type GroundingCandidateGeometryIssue = {
  candidateId: string;
  viewId?: string;
  code: "missing_measurement_view" | "unavailable_measurement_view" | "non_visible_measurement_view"
    | "inconsistent_visible_source" | "measurement_view_coverage";
  message: string;
};

/**
 * Checks measurement provenance, not object recognition. Every returned view
 * must belong to one current-record registry whose visible views share the
 * original visible source. Equal dimensions NEVER register a different sensor.
 * Legacy contracts remain parseable but cannot establish order without a view.
 */
export function assessGroundingCandidateProvenance(
  candidates: readonly GroundingConstraintCandidate[],
  context?: GroundingCandidateProvenanceContext,
): GroundingCandidateGeometryIssue[] {
  const issues: GroundingCandidateGeometryIssue[] = [];
  let dimensions: [number, number] | undefined;
  for (const candidate of candidates) {
    const id = candidate.measurementViewId;
    const issue = (code: GroundingCandidateGeometryIssue["code"], message: string) => {
      issues.push({ candidateId: candidate.id, ...(id ? { viewId: id } : {}), code, message });
    };
    if (!id) {
      issue("missing_measurement_view", `Candidate ${candidate.id} has no measurementViewId. Reference an already returned current-record visible overview or crop covering its full bbox; do not invent provenance or request another image merely to fill this field.`);
      continue;
    }
    const view = context?.get(id);
    if (!view || view.id !== id) {
      issue("unavailable_measurement_view", `Candidate ${candidate.id} references unavailable or stale measurementViewId ${id}. Use an existing current-record visible view only if its pixels support this bbox; otherwise keep the order unresolved.`);
      continue;
    }
    if (view.modality !== "visible") {
      issue("non_visible_measurement_view", `Candidate ${candidate.id} uses ${view.modality} geometry from ${id}. Infrared/depth can supplement identity but cannot establish visible-frame order without registration; equal image dimensions are not registration.`);
      continue;
    }
    if (!Number.isSafeInteger(view.sourceWidth) || !Number.isSafeInteger(view.sourceHeight)
      || view.sourceWidth <= 0 || view.sourceHeight <= 0
      || (dimensions && (dimensions[0] !== view.sourceWidth || dimensions[1] !== view.sourceHeight))) {
      issue("inconsistent_visible_source", `Candidate ${candidate.id} does not have consistent current-record visible source geometry.`);
      continue;
    }
    dimensions = [view.sourceWidth, view.sourceHeight];
    const region = view.region;
    const bbox = candidate.bbox;
    if (!Array.isArray(region) || region.length !== 4 || region.some((edge) => !Number.isFinite(edge) || edge < 0 || edge > 1)
      || region[0] >= region[2] || region[1] >= region[3]
      || bbox[0] < region[0] - 1e-9 || bbox[1] < region[1] - 1e-9
      || bbox[2] > region[2] + 1e-9 || bbox[3] > region[3] + 1e-9) {
      issue("measurement_view_coverage", `Candidate ${candidate.id}'s full visible-source bbox is not covered by measurementViewId ${id}. Use a covering visible view already inspected or keep the geometry unresolved; crop/display coordinates must first be mapped to the original visible frame.`);
    }
  }
  return issues;
}
