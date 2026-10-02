import assert from "node:assert/strict";
import test from "node:test";
import {
  GROUNDING_CONSTRAINT_LIMITS,
  assessGroundingConstraints,
  assessGroundingSpatialOrder,
  validateGroundingConstraintContract,
} from "./grounding-constraints.ts";

const supported = (evidence = "The original overview supplies the stated structural evidence.") => ({ status: "supported", evidence });
const candidate = (id, bbox) => ({
  id, bbox,
  identity: { label: "alpaca", ...supported("Long neck, ears and visible body structure support alpaca identity."), basis: "visual_structure" },
});

function contract() {
  return {
    originalQuery: "The second alpaca from left to right",
    queryCoverage: supported("The object identity, requested ordinal and left-to-right direction are represented."),
    interpretations: [{
      id: "left-to-right", reading: "Select the second alpaca in left-to-right source-image order.",
      ...supported("The query explicitly specifies a single horizontal ordering."),
      requirements: [
        { id: "identity", queryText: "alpaca", description: "The selected object is an alpaca.", ...supported() },
        { id: "rank", queryText: "second", description: "The selected object is second in the declared ordering.", ...supported() },
        { id: "direction", queryText: "from left to right", description: "Order is ascending source x-coordinate.", ...supported() },
      ],
      spatialOrder: {
        axis: "x", direction: "ascending", ordinal: 2,
        candidateIds: ["right", "left", "middle"], selectedCandidateId: "middle",
        candidateSet: supported("The common overview shows these three members of the relevant alpaca group."),
      },
    }],
    // Intentionally stored in discovery order rather than spatial order.
    candidates: [candidate("right", [.7, .3, .9, .8]), candidate("left", [.1, .3, .3, .8]), candidate("middle", [.4, .3, .6, .8])],
    selectedCandidateId: "middle",
  };
}
const assess = (value) => assessGroundingConstraints(value, value.originalQuery);
const codes = (assessment) => assessment.issues.map((issue) => issue.code);

test("a bounded contract preserves the exact query and nested evidence without mutating input", () => {
  const value = contract();
  value.originalQuery = `  ${value.originalQuery}\n`;
  value.interpretations[0].requirements[2].queryText = "from left to right\n";
  const before = structuredClone(value);
  assert.deepEqual(validateGroundingConstraintContract(value, value.originalQuery), value);
  assert.deepEqual(value, before);
  const copied = validateGroundingConstraintContract(value);
  copied.candidates[0].bbox[0] = .65;
  copied.interpretations[0].spatialOrder.candidateIds.reverse();
  assert.deepEqual(value, before);
  assert.throws(() => validateGroundingConstraintContract(value, value.originalQuery.trim()), /exactly match/);
});

test("an omitted query anchors to the loaded record without replacing an explicit mismatch", () => {
  const value = contract();
  const query = value.originalQuery;
  delete value.originalQuery;
  const before = structuredClone(value);
  assert.equal(validateGroundingConstraintContract(value, query).originalQuery, query);
  assert.deepEqual(value, before);
  assert.throws(() => validateGroundingConstraintContract(value), /originalQuery/);
  assert.throws(() => validateGroundingConstraintContract({ ...value, originalQuery: "other query" }, query), /exactly match/);
  assert.throws(() => validateGroundingConstraintContract({ ...value, originalQuery: undefined }, query), /originalQuery/);
});

test("misplaced query excerpts report the exact repair instead of dropping evidence", () => {
  const value = contract();
  value.interpretations[0].queryText = "alpaca";
  assert.throws(() => validateGroundingConstraintContract(value), /Put queryText inside each requirements\[\] item, not on the interpretation/);
});

test("the serialized contract budget includes a server-filled original query", () => {
  const value = contract();
  const query = value.originalQuery + " ".repeat(11900);
  delete value.originalQuery;
  for (let index = 0; index < 24; index++) {
    const extra = candidate(`extra-${index}`, [.1, .1, .2, .2]);
    extra.identity.evidence = "x".repeat(800);
    value.candidates.push(extra);
  }
  assert.ok(Buffer.byteLength(JSON.stringify(value)) < GROUNDING_CONSTRAINT_LIMITS.serializedBytes);
  assert.ok(Buffer.byteLength(JSON.stringify({ ...value, originalQuery: query })) > GROUNDING_CONSTRAINT_LIMITS.serializedBytes);
  assert.throws(() => validateGroundingConstraintContract(value, query), /serialized UTF-8 limit/);
});

test("the total serialized UTF-8 budget bounds many individually valid claims", () => {
  assert.equal(GROUNDING_CONSTRAINT_LIMITS.serializedBytes, 32 * 1024);
  const value = contract();
  for (let index = 0; index < 16; index++) {
    const extra = candidate(`other-${index}`, [.1, .1, .2, .2]);
    extra.identity.evidence = "x".repeat(800);
    value.candidates.push(extra);
  }
  assert.doesNotThrow(() => validateGroundingConstraintContract(value, value.originalQuery));
  for (const extra of value.candidates.slice(3)) extra.identity.evidence = "界".repeat(800);
  const before = structuredClone(value);
  assert.throws(() => validateGroundingConstraintContract(value, value.originalQuery), /serialized UTF-8 limit.*preserve originalQuery exactly/);
  assert.deepEqual(value, before, "oversize contracts must never be truncated into apparent support");
});

test("requirement excerpts stay short without truncating the exact original query", () => {
  assert.equal(GROUNDING_CONSTRAINT_LIMITS.queryExcerptCharacters, 600);
  const value = contract();
  value.originalQuery = "alpaca ".repeat(100);
  value.interpretations[0].requirements = [{
    id: "identity", queryText: value.originalQuery.slice(0, 600),
    description: "The repeated query still requests an alpaca.", ...supported(),
  }];
  const normalized = validateGroundingConstraintContract(value, value.originalQuery);
  assert.equal(normalized.originalQuery, value.originalQuery);
  assert.equal(normalized.interpretations[0].requirements[0].queryText.length, 600);
  value.interpretations[0].requirements[0].queryText = value.originalQuery.slice(0, 601);
  assert.throws(() => validateGroundingConstraintContract(value), /queryText.*at most 600/);
});

test("geometry independently ranks candidates instead of trusting discovery or submitted list order", () => {
  const value = contract();
  const before = structuredClone(value);
  const result = assess(value);
  assert.equal(result.status, "supported");
  assert.equal(result.canLock, true);
  assert.equal(result.requiresHumanReview, true);
  assert.deepEqual(result.orders[0].orderedCandidateIds, ["left", "middle", "right"]);
  assert.equal(result.orders[0].selectedRank, 2);
  assert.deepEqual(result.orders[0].selectedRankRange, [2, 2]);
  assert.deepEqual(result.selectedCandidate, { id: "middle", sourceBbox: [.4, .3, .6, .8], identityStatus: "supported" });
  assert.match(result.limitations.join(" "), /model-declared.*not machine-verified/);
  assert.deepEqual(value, before);
  value.candidates.reverse();
  value.interpretations[0].spatialOrder.candidateIds.reverse();
  assert.deepEqual(assess(value).orders, result.orders);
});

test("adding a previously missed middle candidate recomputes rank and invalidates a premature lock", () => {
  const value = contract();
  value.selectedCandidateId = "right";
  value.interpretations[0].spatialOrder.selectedCandidateId = "right";
  value.candidates = value.candidates.filter((item) => item.id !== "middle");
  value.interpretations[0].spatialOrder.candidateIds = ["right", "left"];
  assert.equal(assess(value).canLock, true);
  value.candidates.push(candidate("middle", [.4, .3, .6, .8]));
  value.interpretations[0].spatialOrder.candidateIds.push("middle");
  const result = assess(value);
  assert.equal(result.canLock, false);
  assert.equal(result.status, "contradicted");
  assert.equal(result.orders[0].selectedRank, 3);
  assert.ok(codes(result).includes("rank_mismatch"));
});

test("horizontal reverse and vertical ordering use the declared axis and direction", () => {
  const value = contract();
  const order = value.interpretations[0].spatialOrder;
  order.direction = "descending";
  order.ordinal = 1;
  order.selectedCandidateId = "right";
  const horizontal = assessGroundingSpatialOrder(order, value.candidates);
  assert.deepEqual(horizontal.orderedCandidateIds, ["right", "middle", "left"]);
  assert.equal(horizontal.selectedRank, 1);
  const vertical = [candidate("bottom", [.1, .7, .3, .9]), candidate("top", [.7, .1, .9, .3])];
  const bottomFirst = assessGroundingSpatialOrder({ ...order, axis: "y", candidateIds: ["top", "bottom"], selectedCandidateId: "bottom" }, vertical);
  assert.deepEqual(bottomFirst.orderedCandidateIds, ["bottom", "top"]);
  assert.equal(bottomFirst.selectedRank, 1);
});

test("the exported geometry helper never silently drops or double-counts candidate references", () => {
  const value = contract();
  const order = value.interpretations[0].spatialOrder;
  assert.throws(() => assessGroundingSpatialOrder({ ...order, candidateIds: ["left", "missing"] }, value.candidates), /unknown candidate/);
  assert.throws(() => assessGroundingSpatialOrder({ ...order, candidateIds: ["left", "left"] }, value.candidates), /unique ids/);
});

test("equal or nearly equal source centers cannot establish a unique ordinal", () => {
  for (const offset of [0, .0000005]) {
    const value = contract();
    value.candidates.find((item) => item.id === "left").bbox = [.4 + offset, .01, .6 + offset, .2];
    const result = assess(value);
    assert.equal(result.canLock, false);
    assert.equal(result.orders[0].selectedRank, undefined);
    assert.deepEqual(result.orders[0].selectedRankRange, [1, 2]);
    assert.deepEqual(result.orders[0].tiedCandidateIds, ["left"]);
    assert.ok(codes(result).includes("spatial_tie"));
  }
});

test("an unresolved identity is a possible member, never promoted to a supported count", () => {
  const value = contract();
  value.candidates.find((item) => item.id === "left").identity.status = "unresolved";
  const result = assess(value);
  assert.equal(result.canLock, false);
  assert.equal(result.orders[0].supportedCount, 2);
  assert.equal(result.orders[0].possibleCount, 3);
  assert.equal(result.orders[0].selectedRank, undefined);
  assert.deepEqual(result.orders[0].selectedRankRange, [1, 2]);
  assert.ok(codes(result).includes("ordering_identity_unresolved"));
  assert.equal(result.requiresHumanReview, true);
});

test("insufficient or contradicted candidate counts do not establish the requested ordinal", () => {
  const value = contract();
  value.interpretations[0].spatialOrder.ordinal = 4;
  assert.ok(codes(assess(value)).includes("insufficient_candidates"));
  value.interpretations[0].spatialOrder.ordinal = 2;
  value.candidates.find((item) => item.id === "left").identity.status = "contradicted";
  const result = assess(value);
  assert.deepEqual(result.orders[0].orderedCandidateIds, ["middle", "right"]);
  assert.equal(result.orders[0].supportedCount, 2);
  assert.equal(result.canLock, false);
  assert.ok(codes(result).includes("ordering_identity_unresolved"));
});

test("an incomplete model-declared candidate set is explicit uncertainty", () => {
  const value = contract();
  value.interpretations[0].spatialOrder.candidateSet = { status: "unresolved", evidence: "A partly occluded object may change the relevant membership." };
  value.interpretations[0].spatialOrder.ordinal = 3;
  const result = assess(value);
  assert.equal(result.canLock, false);
  assert.equal(result.status, "unresolved");
  assert.ok(codes(result).includes("candidate_set_unresolved"));
});

test("The second first alpaca preserves incompatible readings instead of silently repairing the query", () => {
  const value = contract();
  value.originalQuery = "The second first alpaca from left to right";
  value.queryCoverage = { status: "unresolved", evidence: "Both second and first occur; the intended ordinal has not been established." };
  const second = value.interpretations[0];
  second.status = "unresolved";
  second.evidence = "Second is present, but this reading does not resolve the word first.";
  second.requirements.push({ id: "conflicting-first", queryText: "first", description: "Resolve the additional conflicting ordinal.", status: "unresolved", evidence: "The word first cannot be silently discarded." });
  const first = structuredClone(second);
  first.id = "first-reading";
  first.reading = "Select the first alpaca; second remains unexplained.";
  first.spatialOrder.ordinal = 1;
  first.spatialOrder.selectedCandidateId = "left";
  value.interpretations.push(first);
  const normalized = validateGroundingConstraintContract(value, value.originalQuery);
  const result = assess(normalized);
  assert.equal(normalized.originalQuery, "The second first alpaca from left to right");
  assert.equal(normalized.interpretations.length, 2);
  assert.equal(result.canLock, false);
  assert.ok(codes(result).includes("ambiguous_interpretations"));
  assert.ok(codes(result).includes("requirement_unresolved"));
  assert.deepEqual(result.orders.map((item) => item.ordinal), [2, 1]);
  assert.equal(result.requiresHumanReview, true);
});

test("rejected alternatives remain recorded without forcing a resolved reading back to uncertainty", () => {
  const value = contract();
  const alternative = structuredClone(value.interpretations[0]);
  alternative.id = "rejected-reading";
  alternative.status = "contradicted";
  alternative.evidence = "The original query explicitly states left to right, which rejects right-to-left ordering.";
  alternative.spatialOrder.direction = "descending";
  alternative.spatialOrder.selectedCandidateId = "right";
  value.interpretations.push(alternative);
  const result = assess(value);
  assert.equal(result.canLock, true);
  assert.equal(result.orders.length, 2);
});

test("unresolved selection evidence stays reviewable and repeated crops never promote identity", () => {
  for (const basis of ["pixel_measurement", "repeated_view", "unknown"]) {
    const value = contract();
    value.candidates.find((item) => item.id === "middle").identity.basis = basis;
    const before = structuredClone(value);
    const first = assess(value);
    const repeated = assess(value);
    assert.equal(first.canLock, false);
    assert.equal(first.status, "unresolved");
    assert.ok(codes(first).includes("identity_basis"));
    assert.equal(first.requiresHumanReview, true);
    assert.deepEqual(first, repeated);
    assert.deepEqual(value, before);
  }
  const value = contract();
  value.candidates.find((item) => item.id === "middle").identity.status = "unresolved";
  const result = assess(value);
  assert.equal(result.canLock, false);
  assert.equal(result.selectedCandidate.identityStatus, "unresolved");
  assert.ok(codes(result).includes("candidate_identity"));
});

test("unsupported requirements and original-query drift prevent lock even when geometry agrees", () => {
  const value = contract();
  value.interpretations[0].requirements[0].status = "contradicted";
  assert.equal(assess(value).canLock, false);
  assert.equal(assess(value).status, "contradicted");
  value.interpretations[0].requirements[0].status = "supported";
  const mismatch = assessGroundingConstraints(value, "The second first alpaca from left to right");
  assert.ok(codes(mismatch).includes("original_query_mismatch"));
  assert.equal(mismatch.canLock, false);
});

test("selection must reference the same ordered candidate and an intersecting source region", () => {
  const value = contract();
  value.selectedCandidateId = "left";
  assert.ok(codes(assess(value)).includes("selected_candidate_mismatch"));
  value.selectedCandidateId = "middle";
  const outside = assessGroundingConstraints(value, value.originalQuery, [.01, .01, .1, .1]);
  assert.ok(codes(outside).includes("selection_bbox_mismatch"));
  assert.equal(outside.canLock, false);
  const refinedPart = assessGroundingConstraints(value, value.originalQuery, [.45, .4, .5, .5]);
  assert.equal(refinedPart.canLock, true, "boundary/part refinement is allowed within the selected source region");
  value.interpretations[0].spatialOrder.candidateIds = ["left", "right"];
  assert.ok(codes(assess(value)).includes("selected_candidate_not_in_order"));
});

test("legacy and incomplete contracts are reviewable but cannot create a new lock", () => {
  const legacy = assessGroundingConstraints(undefined, "The first alpaca");
  assert.equal(legacy.canLock, false);
  assert.equal(legacy.requiresHumanReview, true);
  assert.ok(codes(legacy).includes("missing_contract"));
  const incomplete = { originalQuery: "", queryCoverage: { status: "unresolved", evidence: "The record has no target query." }, interpretations: [], candidates: [] };
  assert.doesNotThrow(() => validateGroundingConstraintContract(incomplete, ""));
  assert.equal(assess(incomplete).canLock, false);
  const value = contract();
  value.interpretations[0].requirements = [];
  assert.ok(codes(assess(value)).includes("missing_requirements"));
});

test("malformed, unbounded and unanchored claims are rejected instead of silently repaired", () => {
  const invalidMutations = [
    (value) => { value.originalQuery = "x".repeat(12001); },
    (value) => { value.extra = true; },
    (value) => { value.queryCoverage.evidence = " "; },
    (value) => { value.queryCoverage.evidence = "x".repeat(801); },
    (value) => { value.queryCoverage.status = "verified"; },
    (value) => { value.candidates[0].bbox = [0, 0, NaN, 1]; },
    (value) => { value.candidates[0].bbox = [0, 0, 2, 1]; },
    (value) => { value.candidates[0].bbox = [.5, .1, .1, .9]; },
    (value) => { value.candidates[0].identity.basis = "more_crops"; },
    (value) => { value.candidates[0].identity.evidence = ""; },
    (value) => { value.candidates.push(structuredClone(value.candidates[0])); },
    (value) => { value.candidates = Array(33).fill(value.candidates[0]); },
    (value) => { value.selectedCandidateId = "missing"; },
    (value) => { value.interpretations = Array(7).fill(value.interpretations[0]); },
    (value) => { value.interpretations.push(structuredClone(value.interpretations[0])); },
    (value) => { value.interpretations[0].requirements = Array(13).fill(value.interpretations[0].requirements[0]); },
    (value) => { value.interpretations[0].requirements[0].queryText = "sheep"; },
    (value) => { value.interpretations[0].requirements[0].queryText = ""; },
    (value) => { value.interpretations[0].requirements[0].status = "certain"; },
    (value) => { value.interpretations[0].spatialOrder.candidateIds.push("missing"); },
    (value) => { value.interpretations[0].spatialOrder.candidateIds.push("left"); },
    (value) => { value.interpretations[0].spatialOrder.axis = "z"; },
    (value) => { value.interpretations[0].spatialOrder.direction = "left"; },
    (value) => { value.interpretations[0].spatialOrder.ordinal = 0; },
    (value) => { value.interpretations[0].spatialOrder.ordinal = 2.5; },
    (value) => { value.interpretations[0].spatialOrder.ordinal = Infinity; },
  ];
  for (const mutate of invalidMutations) {
    const value = contract();
    mutate(value);
    assert.throws(() => validateGroundingConstraintContract(value), /[Gg]rounding contract/);
  }
  for (const value of [null, [], "contract", {}]) assert.throws(() => validateGroundingConstraintContract(value), /[Gg]rounding contract/);
});

test("the reported three-drone layout orders the ground candidate between the two flyers", () => {
  const value = contract();
  value.originalQuery = "third drone from the left";
  value.candidates = [candidate("leftflyer", [.18, .2, .30, .4]), candidate("rightflyer", [.68, .2, .80, .4]), candidate("ground", [.455, .7, .525, .8])];
  for (const item of value.candidates) item.identity.label = "drone";
  value.selectedCandidateId = "ground";
  value.interpretations[0].requirements = [{ id: "query", queryText: value.originalQuery, description: "Third matching drone from the left", ...supported() }];
  Object.assign(value.interpretations[0].spatialOrder, { ordinal: 3, candidateIds: ["leftflyer", "rightflyer", "ground"], selectedCandidateId: "ground" });
  const result = assess(value);
  assert.deepEqual(result.orders[0].orderedCandidateIds, ["leftflyer", "ground", "rightflyer"]);
  assert.equal(result.orders[0].selectedRank, 2);
  assert.equal(result.canLock, false);
  assert.ok(codes(result).includes("rank_mismatch"));
});
