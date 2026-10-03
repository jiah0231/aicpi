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
    candidates: contract.candidates.map((item) => [item.id, item.identity.status, item.identity.basis]).sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
    readings: contract.interpretations.map((item) => [item.id, item.status,
      item.requirements.map((condition) => [condition.id, condition.status]).sort((a, b) => a[0].localeCompare(b[0])),
      item.spatialOrder ? { axis: item.spatialOrder.axis, direction: item.spatialOrder.direction, ordinal: item.spatialOrder.ordinal, selectedCandidateId: item.spatialOrder.selectedCandidateId, candidateIds: [...item.spatialOrder.candidateIds].sort(), candidateSet: item.spatialOrder.candidateSet.status } : null,
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
    lastTrial: state?.lastTrial,
    decisionRequired: state?.lastTrial?.nextObservation === null,
    nextDecision: assessment.canLock
      ? "Declared checks permit a proposal, not verified truth. Request human review after any concrete remaining boundary check."
      : "Use tools only when a specific remaining difficulty has a plausible observable test. Assess the result: useful, inconclusive, contradictory or failed; keep what remains unknown explicit. An unsuccessful trial can be reasonable, and a distinct hypothesis or recovery can justify another observation. If no useful next observation remains, submit an unresolved low-confidence proposal for human review, or ask clarification and pause. Rewording, tool switching and rendered pixels alone establish no support; never call a preview verified.",
    limitation: "Counts track declared support changes and source actions only, not semantic progress or visual truth. No view cap or automatic approval. Stable support, repeated actions and tool success do not determine usefulness; lastTrial is the model’s assessment, not verified truth.",
  };
}
