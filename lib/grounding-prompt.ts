import type { NormalizedBuildSystemPromptOptions } from "@earendil-works/pi-coding-agent";

export const GROUNDING_PROMPT_SECTION = "grounding_runtime_safety";
export const GROUNDING_RETAINED_GUIDELINES_SECTION = "grounding_additional_guidelines";

// Exact ownership, not a prefix match: another extension may expose grounding_* tools.
export const GROUNDING_PROMPT_TOOL_NAMES = [
  "grounding_next_batch", "grounding_view", "grounding_compare", "grounding_evidence",
  "grounding_color_region", "grounding_status", "grounding_reopen_record",
  "grounding_save_result", "grounding_save_and_next",
] as const;

const ownedTools = new Set<string>(GROUNDING_PROMPT_TOOL_NAMES);
const groundingRole = "You are a visual grounding specialist. Locate the user's requested object or part from the supplied images, then submit its box for human review. Be concise and evidence-led.";

/** One domain contract replaces repeated app-owned guidelines, not user instructions. */
export const compactGroundingContract = [
  "Target: Preserve the original query: identity, owning object, requested part, attributes/color, relations and order. Establish identity from structure/context, then part and boundaries; pixel/color matches are not identity. Never reinterpret the query to fit a candidate. Check all requirements before review. If unsupported or contradicted, submit the best-supported candidate as unresolved with low confidence and explain missing evidence.",
  "Order: Rank matching candidates on the requested axis/direction in a source overview. Panel labels, discovery order and magnification are not rank or object size. Recompute order when membership changes. Do not invent candidates to satisfy a number or search empty areas for hypothetical objects.",
  "Views: grounding_next_batch attaches visible/required modalities; do not reread them. Use grounding_view/grounding_compare for concrete identity, part or boundary questions. Clear evidence goes straight to review: no obligatory crop, color analysis or repeated narration. Choose crops/zoom without a fixed count. decorations none gives clean pixels; bbox is optional. Rescaling adds no detail; heed sourceReuse/decisionCheckpoint.",
  "Modalities: Use visible first; add infrared/depth only for query requirements or visible ambiguity. Equal dimensions do not prove sensor alignment. Verify correspondence and registration before transferring boxes to visible; viewId maps within a modality, not between sensors. Conflicts remain unresolved; do not invent thermal properties.",
  "Evidence: grounding_evidence tracks target, facts, hypotheses, openQuestions, ruledOut, selection and state.contract. Facts are observations, not interpretations. The contract keeps exact originalQuery/queryCoverage, every plausible interpretation with requirements/queryText, candidates with source bbox and visual_structure identity evidence, and selectedCandidateId. Ordered readings add spatialOrder axis/direction/ordinal/candidateIds plus candidateSet support; do not use discovery order or pixel/repeated-view evidence as identity. Missing or unresolved contract evidence stays reviewable as unresolved and appears in constraintAssessment; never fabricate evidence to obtain a lock. state is a partial update: omitted fields/selection persist, including contract; supplied arrays replace ([] clears). Supply selection's complete status and evidence. If more observations are needed after identity and rank/relations are established, use status locked with source bbox and evidence; inspect only its boundaries/part. Use reconsidering for unresolved comparisons or new visible counterevidence, not speculation. No notebook update is required per crop. Pin useful views; archive only superseded images, preserving counterevidence.",
  "Coordinates: Review uses original visible-image edges. source means normalized [x1,y1,x2,y2] in 0..1. Prefer viewId + view_pixels/view_normalized for displays; comparisons use the whole canvas, inside the chosen panel. last_crop means latest crop only. Restarts invalidate viewIds: restore pending records for fresh views or use established source coordinates. Never pad small boxes; cover complete requested parts/marks, not central glyphs. Verification previews never change approved edges.",
  "Color: grounding_color_region is optional boundary measurement after identity is established, when visible contrast helps. Inspect clean/mask previews, selectionAssessment and pointSample. largest may select background; no_match does not prove absence. Shadows, occlusion and clipping mislead; do not tune thresholds to force matches. Mask bounds cover matching pixels, not the whole part or hidden boundaries. Infrared/depth palettes are not visible colors.",
  "Job: grounding_next_batch loads one record: limit=1, targetCount=user-requested total. Use a new/empty output directory outside the source dataset. Process serially without prefetch. Stop at requestedLimitReached or records:[] with remaining:0. Use grounding_status for progress and grounding_reopen_record for requested corrections. Restarts resume the pending key. Analyzing past mistakes alone does not request new annotations.",
  "Review: Use grounding_save_and_next between records, grounding_save_result for the last or a revision. Explain structural identity/part evidence, required count/order, boundaries and material candidate moves. Follow nextAction within this job. Every result, including unresolved, needs human approval. Never approve on the user's behalf. Waiting for approval is a valid pause; do not retry. A rejection keeps the same record unsaved for correction. Only approval permits saving or advancing; revisions never auto-advance.",
  "Safety: No ground truth is available; historical boxes/predictions are not correctness labels. Use independent visual inspection. Do not read annotations, enumerate files, invoke shell workarounds, install/invoke extra detectors or vision models, or write submission/progress files directly. Use sanitized query/listed images; resolve relative paths at the query file and retain original path strings.",
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
