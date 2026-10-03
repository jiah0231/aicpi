import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";
const jiti = createJiti(import.meta.url);
const { advanceGroundingDecision, groundingConditionSignature, groundingDecisionCheckpoint } = await jiti.import("./grounding-decision-checkpoint.ts");
const { assessGroundingConstraints } = await jiti.import("./grounding-constraints.ts");

function state() {
  return { contract: { originalQuery: "third girl with both hands behind back", queryCoverage: {status: "supported", evidence: "All query conditions represented"}, selectedCandidateId: "a",
    candidates: [{id: "a", bbox: [.1,.1,.2,.5], identity: {label: "girl", status: "supported", basis: "visual_structure", evidence: "Visible human silhouette"}}],
    interpretations: [{id: "all", reading: "Third girl, additionally hands behind back", status: "unresolved", evidence: "Pose and rank uncertain", requirements: [
      {id: "pose", queryText: "both hands behind back", description: "Pose", status: "unresolved", evidence: "Hands are not visible"},
    ], spatialOrder: {axis: "x", direction: "ascending", ordinal: 3, candidateIds: ["a"], selectedCandidateId: "a", candidateSet: {status: "unresolved", evidence: "Insufficient supported members"}}}],
  }};
}

test("prose and fact rewrites do not reset condition stagnation; actual declarations do", () => {
  const original = state();
  const changed = structuredClone(original);
  changed.facts = ["New wording without new evidence"];
  changed.contract.candidates[0].bbox = [.101,.101,.201,.501];
  changed.contract.interpretations[0].spatialOrder.candidateSet.evidence = "The same unresolved membership with rewritten prose";
  changed.openQuestions = ["Could the hands be behind the back?"];
  changed.contract.interpretations[0].requirements[0].evidence = "Same invisible hands, differently described";
  assert.equal(groundingConditionSignature(original), groundingConditionSignature(changed));
  let progress = advanceGroundingDecision(undefined, original, "visible:roi-a");
  progress = advanceGroundingDecision(progress, changed, "visible:roi-b");
  progress = advanceGroundingDecision(progress, changed, "visible:roi-a");
  assert.equal(progress.unchangedConditions, 2);
  assert.equal(progress.lastActionRepeated, true);
  changed.contract.interpretations[0].requirements[0].status = "contradicted";
  progress = advanceGroundingDecision(progress, changed, "visible:roi-a");
  assert.equal(progress.unchangedConditions, 0);
  assert.equal(progress.lastActionRepeated, true, "a changed declaration never makes repeated pixels novel");
  for (let n = 0; n < 40; n++) progress = advanceGroundingDecision(progress, changed, `action-${n}`);
  assert.equal(progress.recentActions.length, 12);
});

test("checkpoint exposes missing pose and rank without inventing support or limiting views", () => {
  const current = state();
  const assessment = assessGroundingConstraints(current.contract, current.contract.originalQuery);
  let progress = advanceGroundingDecision(undefined, current, "source-a");
  progress = advanceGroundingDecision(progress, current, "source-a");
  const checkpoint = groundingDecisionCheckpoint(current, assessment, progress);
  assert.equal(checkpoint.verification, "not_verified");
  assert.equal(checkpoint.decisionRequired, false, "unchanged declarations do not judge whether a trial is useful");
  assert.equal(checkpoint.unresolvedConditions[0].queryText, "both hands behind back");
  assert.equal(checkpoint.order[0].requestedRank, 3);
  assert.equal(checkpoint.order[0].selectedRank, null);
  assert.match(checkpoint.nextDecision, /unresolved low-confidence proposal/);
  assert.match(checkpoint.nextDecision, /clarification and pause/);
  assert.match(checkpoint.limitation, /not semantic progress or visual truth/);
  assert.match(checkpoint.limitation, /No view cap or automatic approval/);
});


test("trial outcomes do not assert truth, force tool use, or punish a reasonable failed trial", () => {
  const current = state();
  const assessment = assessGroundingConstraints(current.contract, current.contract.originalQuery);
  let progress;
  for (const action of ["view", "process", "color", "compare", "view"]) progress = advanceGroundingDecision(progress, current, action);
  for (const outcome of ["useful", "inconclusive", "contradictory", "failed"]) {
    current.lastTrial = { question: "Can the visible sleeve establish hand position?", outcome,
      observation: "The hand itself remains hidden", remainingUnknown: "Both hands behind back",
      nextObservation: "Inspect the other candidate's visible wrist" };
    const checkpoint = groundingDecisionCheckpoint(current, assessment, progress);
    assert.equal(checkpoint.decisionRequired, false);
    assert.equal(checkpoint.lastTrial.outcome, outcome);
    assert.equal(checkpoint.verification, "not_verified");
    assert.equal(checkpoint.unresolvedConditions[0].status, "unresolved");
  }
  current.lastTrial.nextObservation = null;
  assert.equal(groundingDecisionCheckpoint(current, assessment, progress).decisionRequired, true);
  current.lastTrial.nextObservation = "Recover unavailable source pixels to inspect the wrist";
  assert.equal(groundingDecisionCheckpoint(current, assessment, progress).decisionRequired, false);
});
