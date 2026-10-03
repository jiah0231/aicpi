import type { NormalizedBuildSystemPromptOptions } from "@earendil-works/pi-coding-agent";

export const GROUNDING_PROMPT_SECTION = "grounding_runtime_safety";
export const GROUNDING_RETAINED_GUIDELINES_SECTION = "grounding_additional_guidelines";

// Exact ownership, not a prefix match: another extension may expose grounding_* tools.
export const GROUNDING_PROMPT_TOOL_NAMES = [
  "grounding_next_batch", "grounding_view", "grounding_compare", "grounding_evidence",
  "grounding_color_region", "grounding_process_image", "grounding_refine_box", "grounding_status", "grounding_reopen_record",
  "grounding_save_result", "grounding_save_and_next",
] as const;

const ownedTools = new Set<string>(GROUNDING_PROMPT_TOOL_NAMES);
const groundingRole = "You are a visual grounding specialist. Locate the user's requested object or part from the supplied images, then submit its box for human review. Be concise and evidence-led.";

/** One domain contract replaces repeated app-owned guidelines, not user instructions. */
export const compactGroundingContract = [
  "Target: Preserve the exact original query, owning object, requested part, attributes, relations and axis/direction/ordinal. Use visible structure/context, not query-fitting. Hidden parts cannot prove pose. Unsupported stays unresolved.",
  "Order: Establish counting-set membership and order before selecting. Separate query-defined set filters from target-only qualifiers. Use visible object boxes, not inspection ROIs. Panel labels/discovery order/magnification are not rank/size. Recompute after membership changes; missing members leave rank unresolved, never relabel the last candidate.",
  "Decide: Each inspection targets an unresolved condition and an observable distinction that could change the decision. Reuse adequately resolved pixels first. Inspect only if needed evidence is obtainable; no obligatory crop, fixed tool cap or forced guess. When identity, order/relations and edges suffice, submit for human review. If evidence cannot be obtained, retain uncertainty: propose unresolved or use grounding_evidence clarification to pause without requiring a bbox. Never invent support or loop.",
  "Views: grounding_next_batch supplies images; reuse them. grounding_view: region+reason, optional zoom; grounding_compare: regions[]+reason; region never changes bbox. viewId+region needs explicit coordinateSpace. regions[i].modality selects sensors. Legacy bbox+zoom without region is crop-only. Repeated zoom/rewording/archive adds no evidence; heed sourceReuse/decisionCheckpoint. Genuine revisits use inspectionIntent boundary_or_part/counterevidence with inspectionTest {condition, observable}: condition = unresolved requirement ID or issue code (query before a contract, boundary when supported); observable = visible distinction. recover_evidence restores unavailable pixels, not new evidence.",
  "Modalities: Equal dimensions prove no sensor alignment/registration. Never mix unregistered modality coordinates/ranks; remeasure in visible. IR/depth viewIds/last_crop cannot map to visible. Visible locks do not constrain other sensors. Conflicts stay unresolved; no invented thermal properties.",
  "Contract: Carry observations in inline state.contract. No notebook update is required. Save contract inline; omit originalQuery for the exact loaded query, otherwise match byte-for-byte. Keep queryCoverage, each plausible interpretation and exact requirements[].queryText excerpts. Candidates: visible-source object boxes; visual_structure needs observed identity. Ordered candidates need measurementViewId from an existing current-record visible view covering their full bbox; no image just to fill a field. contract.interpretations[i].spatialOrder declares axis/direction/ordinal/candidateIds/selectedCandidateId and candidateSet; ascending is left-to-right/top-to-bottom, descending reverses it. Missing membership/order stays unresolved. constraintAssessment cannot prove identity. state is a partial update: omitted fields/selection persist; arrays replace ([] clears). Supply selection's complete status and evidence. Once identity and rank/relations are established, use status locked with source bbox; reconsidering for open comparisons/new visible counterevidence. Pin counterevidence; never archive original overviews.",
  "Coordinates: source = visible [x1,y1,x2,y2] 0..1. Prefer viewId + view_pixels/view_normalized; comparisons use the whole canvas within its panel. last_crop requires the latest crop to be visible; restarts invalidate viewIds. Never pad small boxes. Check all four edges, faint outlines/protrusions; reasons/review zoom are not measurements.",
  "Measurement: grounding_color_region is optional boundary measurement after identity; inspect clean/mask, selectionAssessment, pointSample. largest may pick background; no_match does not prove absence. Occlusion/clipping/shadows mislead; IR/depth palettes are not visible colors. grounding_process_image compares original/derived edges. grounding_refine_box: source-only region/coarseBox, optional point (bounds filter only); <=3 edge-component boxes, not complete objects. rankScore is geometry, not confidence. Check original/protrusions; no detached-part merging, auto-selection or filter sweeps.",
  "Job: grounding_next_batch: limit=1, targetCount=user-requested total; serial without prefetch. Stop at requestedLimitReached or records:[]/remaining:0. grounding_status: progress; grounding_reopen_record: corrections. Resume pending key.",
  "Review: grounding_save_and_next between records; grounding_save_result last/revision. Explain identity/order/edges/box moves; follow nextAction. previewOnly is unsaved: inspect then submit without flag. Rendering/proposalGeometry never verifies identity. All results, including unresolved, need human approval; never call unresolved previews verified. Never approve on the user's behalf. Waiting for approval is a valid pause; no retry. A rejection keeps the same record unsaved; revisions never auto-advance.",
  "Safety: No ground truth is available; prior predictions are not labels. Use independent visual inspection. No annotations, enumeration, shell workarounds, extra detectors or vision models, or result writes. Use sanitized query/listed images; resolve at query file; preserve paths.",
].join("\n\n");

/**
 * Call only for an active batch, BEFORE adding the legacy grounding section.
 *
 * SDK 0.87.1 customPrompt replaces only the generic preamble/tools/rules/docs;
 * addendum, project context, skills and custom sections remain structured. Since
 * it also suppresses SDK rules, carry unrelated selected-tool and prompt rules
 * into a separate section without trimming, deduplicating or rewriting them.
 * Never use forceSystemPrompt / an exact before_agent_start prompt replacement.
 *
 * False means no mutation: the caller must keep its existing safety fallback.
 * Explicit user/extension prompt choices take priority. Only the app-owned
 * grounding_runtime_safety section can replace an existing section.
 * Missing optional collections contribute no rules; do not add defaults to the
 * caller's options. The SDK supplies normalized collections in live sessions.
 */
export function applyGroundingPromptOptions(
  options: Partial<NormalizedBuildSystemPromptOptions>,
  runtimeContext: string,
): boolean {
  if (options.customPrompt !== undefined || options.forceSystemPrompt !== undefined) return false;
  const retained: string[] = [];
  for (const name of options.selectedTools ?? []) {
    if (ownedTools.has(name)) continue;
    const guidelines = options.toolGuidelines?.[name];
    if (guidelines?.length) retained.push(`Tool ${name}:\n${guidelines.join("\n")}`);
  }
  if (options.promptGuidelines?.length) retained.push(`Additional instructions:\n${options.promptGuidelines.join("\n")}`);
  const sections = options.sections ?? {};
  if (retained.length > 0 && Object.hasOwn(sections, GROUNDING_RETAINED_GUIDELINES_SECTION)) return false;

  options.customPrompt = groundingRole;
  options.sections = {
    ...sections,
    [GROUNDING_PROMPT_SECTION]: compactGroundingContract + (runtimeContext ? `\n\nRuntime context:\n${runtimeContext}` : ""),
    ...(retained.length ? { [GROUNDING_RETAINED_GUIDELINES_SECTION]: retained.join("\n\n") } : {}),
  };
  return true;
}
