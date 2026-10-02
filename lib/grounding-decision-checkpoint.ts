import type { GroundingWorkingState } from "./grounding-evidence";
import type { GroundingConstraintAssessment } from "./grounding-constraints";

/** Per-record, ephemeral counters. No image interpretation and no learned query data. */
export type GroundingDecisionProgress = {
  inspections: number;
  unchangedConditions: number;
  conditionSignature: string;
  recentActions: string[];
  lastActionRepeated: boolean;
};

/** Prose edits and reworded reasons are not evidence of changed condition support. */
export function groundingConditionSignature(state?: GroundingWorkingState): string {
  const contract = state?.contract;
  return JSON.stringify(contract ? {
    coverage: contract.queryCoverage.status,
    selected: contract.selectedCandidateId,
    candidates: contract.candidates.map((item) => [item.id, item.bbox, item.identity.status, item.identity.basis]).sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
    readings: contract.interpretations.map((item) => [item.id, item.status,
      item.requirements.map((condition) => [condition.id, condition.queryText, condition.status]).sort((a, b) => a[0].localeCompare(b[0])),
      item.spatialOrder ? { ...item.spatialOrder, candidateIds: [...item.spatialOrder.candidateIds].sort(), candidateSet: item.spatialOrder.candidateSet.status } : null,
    ]).sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
  } : null);
}

export function advanceGroundingDecision(previous: GroundingDecisionProgress | undefined, state: GroundingWorkingState | undefined, action: string): GroundingDecisionProgress {
  const conditionSignature = groundingConditionSignature(state);
  return {
    inspections: (previous?.inspections ?? 0) + 1,
    unchangedConditions: previous?.conditionSignature === conditionSignature ? previous.unchangedConditions + 1 : 0,
    conditionSignature,
    lastActionRepeated: previous?.recentActions.includes(action) ?? false,
    recentActions: [...(previous?.recentActions ?? []).filter((item) => item !== action), action].slice(-12),
  };
}

export function groundingDecisionCheckpoint(state: GroundingWorkingState | undefined, assessment: GroundingConstraintAssessment, progress?: GroundingDecisionProgress) {
  const pendingConditions = state?.contract?.interpretations.filter((item) => item.status !== "contradicted")
    .flatMap((item) => item.requirements.filter((condition) => condition.status !== "supported")
      .map((condition) => ({ reading: item.id, condition: condition.id, queryText: condition.queryText, status: condition.status }))) ?? [];
  const unchanged = progress?.conditionSignature === groundingConditionSignature(state) ? progress.unchangedConditions : 0;
  return {
    verification: "not_verified" as const,
    declaredConstraints: assessment.status,
    unresolvedConditions: pendingConditions,
    unresolvedChecks: assessment.issues.map((issue) => ({ code: issue.code, message: issue.message })),
    order: assessment.orders.map((order) => ({ reading: order.interpretationId, requestedRank: order.ordinal,
      declaredCount: order.possibleCount, supportedCount: order.supportedCount, rankWithinDeclaredSet: order.selectedRank ?? null,
      selectedRank: state?.contract?.interpretations.find((item) => item.id === order.interpretationId)?.spatialOrder?.candidateSet.status === "supported" ? order.selectedRank ?? null : null })),
    inspections: progress?.inspections ?? 0,
    inspectionsWithoutDeclaredChange: unchanged,
    repeatedSourceAction: progress?.lastActionRepeated ?? false,
    decisionRequired: !assessment.canLock && Boolean(progress && (unchanged > 0 || progress.lastActionRepeated)),
    nextDecision: assessment.canLock
      ? "Declared checks permit a proposal, not verified truth. Request human review after any concrete remaining boundary check."
      : "Give a concise checkpoint: what remains unresolved, which specific observable evidence could resolve it, and the next action. Use one genuinely useful inspection (name its condition and expected visible distinction), submit an unresolved low-confidence proposal for human review, or ask clarification and pause. Rephrasing/zoom is not new evidence; invisible/occluded parts cannot establish a required attribute or pose. Do not repeat a narrative search or call a preview verified.",
    limitation: "Counts track declared support changes and source actions only, not semantic progress or visual truth. No view cap or automatic approval.",
  };
}
