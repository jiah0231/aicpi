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
  "Target: Preserve the original query: identity, owning object, requested part, attributes/color, relations and order. Establish identity from structure/context before boundaries; pixels/color are not identity. Never reinterpret the query to fit a candidate. Unsupported requirements stay unresolved with low confidence and missing evidence explained.",
  "Order: Rank matching candidates on the requested axis/direction in a source overview. Panel labels, discovery order and magnification are not rank or object size. Recompute when membership changes. Never invent candidates or search empty areas for hypothetical objects.",
  "Views: grounding_next_batch attaches visible/required modalities; do not reread them. grounding_view/grounding_compare answer concrete identity, part or boundary questions. Clear evidence goes straight to review: no obligatory crop or color call. No fixed crop count. A clean crop needs region+reason; zoom is optional. region never changes the target bbox. Legacy bbox+zoom without region is crop-only. Rescaling adds no detail; heed sourceReuse/decisionCheckpoint.",
  "Modalities: Add infrared/depth only for query requirements or visible ambiguity. Equal dimensions do not prove sensor alignment or registration. IR/depth viewIds and last_crop cannot map to visible; remeasure on an existing visible view. A visible lock cannot constrain unregistered sensor geometry. Conflicts stay unresolved; never invent thermal properties.",
  "Evidence: grounding_evidence tracks target, facts, hypotheses, openQuestions, ruledOut, selection and state.contract. Submit contract inline in a save call to avoid a separate checkpoint. Facts are observations, not interpretations. originalQuery may be omitted to use the exact loaded query; supplied mismatches fail. Keep queryCoverage, each plausible interpretation with requirements[].queryText (only inside requirements), candidates with source bbox/visual_structure identity evidence, and selectedCandidateId. Ordered readings add spatialOrder axis/direction/ordinal/candidateIds and candidateSet support. Pixel/repeated-view claims cannot establish identity. Missing support remains unresolved in constraintAssessment; never fabricate evidence for a lock. state is a partial update: omitted fields/selection persist, including contract; arrays replace ([] clears). Supply selection's complete status and evidence. If more observations are needed after identity and rank/relations are established, use status locked with source bbox; inspect only boundaries/parts. Use reconsidering for unresolved comparisons or new visible counterevidence. No notebook update is required per crop. Pin useful views; archive superseded images, preserving counterevidence.",
  "Coordinates: source means original visible-image normalized [x1,y1,x2,y2] in 0..1. Prefer actual viewId + view_pixels/view_normalized; comparisons use the whole canvas inside its panel. last_crop only works if the latest crop is visible. Restarts invalidate viewIds: restore fresh views or use established source coordinates. Never pad small boxes. Check all four edges and complete visible silhouette (tail, peel, dome, faint outer plate), unless asked for a subpart. A reason is not a measurement. Small boxes get a clean human-review zoom before approval, not automatic visual verification. Previews never change approved edges.",
  "Color: grounding_color_region is optional boundary measurement after identity, when visible contrast helps. Inspect clean/mask, selectionAssessment and pointSample. largest may pick background; no_match does not prove absence. Shadows/occlusion/clipping mislead; never tune to force matches. Matching pixels need not cover the whole part; IR/depth palettes are not visible colors.",
  "Job: grounding_next_batch loads limit=1, targetCount=user-requested total, into a new/empty output directory outside the dataset. Work serially without prefetch. Stop at requestedLimitReached or records:[] with remaining:0. Use grounding_status for progress, grounding_reopen_record for requested corrections. Restarts resume the pending key; discussing past mistakes does not request new annotations.",
  "Review: grounding_save_and_next between records; grounding_save_result last or revision. Explain identity/part, count/order, boundaries and material moves. Follow nextAction. For a questionable/tiny box, optional previewOnly returns its actual overlay and clean pixels before review, without saving/advancing; inspect, then submit without the flag. Every result, including unresolved, needs human approval. Never approve on the user's behalf. Waiting for approval is a valid pause; do not retry. A rejection keeps the same record unsaved. Only approval permits saving/advancing; revisions never auto-advance.",
  "Safety: No ground truth is available; prior predictions are not labels. Use independent visual inspection. No annotation reads, file enumeration, shell workarounds, extra detectors or vision models, or direct submission/progress writes. Use sanitized query/listed images; resolve paths at the query file and retain original path strings.",
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
