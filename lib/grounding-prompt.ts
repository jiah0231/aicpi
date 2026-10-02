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
  "Target: Preserve the original query: identity, owning object, requested part, attributes/color, relations and order. Establish identity from structure/context before boundaries; never fit queries to candidates. Unsupported conditions stay unresolved/low confidence. Hidden parts do not prove pose.",
  "Order: Rank the query-defined counting set by axis/direction in the overview; keep explicit set filters, not target-only qualifiers. Panel labels/discovery order/magnification are not rank/size. Recompute after membership changes. Too few candidates leaves rank unresolved; never relabel the last or search empty areas for hypothetical members.",
  "Views: grounding_next_batch supplies modalities; do not reread. grounding_view/grounding_compare resolve questions; no obligatory crop or fixed count. Crops: region+reason, optional zoom. viewId+region needs explicit coordinateSpace; never guess. regions[i].modality selects sensors; separate geometry. region never changes bbox. Legacy bbox+zoom without region is crop-only. Rescaling adds no detail; heed sourceReuse/decisionCheckpoint.",
  "Modalities: IR/depth for query needs/ambiguity. Equal dimensions prove no sensor alignment/registration. IR/depth viewIds/last_crop cannot map to visible; remeasure. Visible locks do not constrain unregistered sensors. Conflicts stay unresolved; no invented thermal properties.",
  "Evidence: Views/compare accept inline grounding_evidence state; no notes call. Keep boxes/conditions in state.contract; color components are not the counting set. Repeated source actions reuse retained pixels; rewording/zoom adds no evidence. Use inspectionIntent for boundary/recovery/new visible counterevidence. At decisionCheckpoint name the unresolved condition, observable distinction and next action, or propose uncertainty; avoid narrative loops. Save contract inline; omit originalQuery for loaded query; mismatches fail. Keep queryCoverage, each plausible interpretation with requirements[].queryText, source-box candidates with visual_structure evidence, and selectedCandidateId. contract.interpretations[i].spatialOrder beside requirements declares axis/direction/ordinal/candidateIds/selectedCandidateId and candidateSet. Missing order stays unresolved; field repairs need no images. constraintAssessment cannot prove identity; never fabricate support. state is a partial update: omitted fields/selection persist; arrays replace ([] clears). Supply selection's complete status and evidence. Once identity and rank/relations are established, use status locked with source bbox; inspect only boundaries/parts. Use reconsidering for open comparisons/new visible counterevidence. No notebook update is required per crop; pin counterevidence. Runtime manages capacity. Original overviews cannot be unpinned/archived.",
  "Coordinates: source = visible [x1,y1,x2,y2] normalized to 0..1. Prefer viewId + view_pixels/view_normalized; comparisons use the whole canvas within its panel. last_crop requires the latest crop to be visible. Restarts invalidate viewIds. Never pad small boxes. Check all four edges, faint outlines/protrusions. Reasons are not measurements; review zoom changes no edge.",
  "Color: grounding_color_region: optional boundary measurement after identity when useful. Inspect clean/mask, selectionAssessment, pointSample. largest may pick background; no_match does not prove absence. Shadows/occlusion/clipping mislead; never force matches or assume complete parts; IR/depth palettes are not visible colors.",
  "Processing: grounding_process_image compares original/derived edges. grounding_refine_box takes source-only region/coarseBox, optional point (bounds filter only): <=3 edge-component boxes, not complete objects. rankScore is geometry, not confidence. Check original/protrusions; no detached-part merging or auto-selection. Filters invent/erase edges; skip clear boxes/sweeps.",
  "Job: grounding_next_batch: limit=1, targetCount=user-requested total, empty output outside dataset. Serially without prefetch. Stop at requestedLimitReached or records:[]/remaining:0. grounding_status: progress; grounding_reopen_record: corrections. Resume pending key after restart; past errors request no annotations.",
  "Review: grounding_save_and_next between records; grounding_save_result last or revision. Explain identity/part, order/count, edges and box moves. Follow nextAction. previewOnly: unsaved overlay/clean pixels; inspect then submit without flag; rendering never verifies identity or resolves missing conditions. proposalGeometry: box overlap, not identity; occlusion allowed. All results, including unresolved, need human approval; unresolved previews must never be called verified. Never approve on the user's behalf. Waiting for approval is a valid pause; do not retry. A rejection keeps the same record unsaved; revisions never auto-advance. Without useful checks, grounding_evidence clarification pauses without requiring a bbox.",
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
