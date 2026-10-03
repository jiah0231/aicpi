import { assessGroundingCandidateProvenance, type GroundingCandidateProvenanceContext, type GroundingCandidateGeometryIssue } from "./grounding-candidate-provenance";

/**
 * A model-declared grounding contract, not an object recognizer or language
 * parser. The runtime checks its structure, original-query anchor and geometry;
 * a human still reviews every proposal, including apparently supported ones.
 */
export type GroundingSupportStatus = "supported" | "unresolved" | "contradicted";
export type GroundingConstraintBox = [number, number, number, number];
export type GroundingSupport = { status: GroundingSupportStatus; evidence: string };

export type GroundingConstraintCandidate = {
  id: string;
  /** Full visible-source normalized edges, never display/crop coordinates. */
  bbox: GroundingConstraintBox;
  /** Explicit reference to the current-record visible view used for this bbox. */
  measurementViewId?: string;
  identity: GroundingSupport & {
    label: string;
    basis: "visual_structure" | "pixel_measurement" | "repeated_view" | "unknown";
  };
};

export type GroundingQueryRequirement = GroundingSupport & {
  id: string;
  /** An exact excerpt of originalQuery; it is not silently normalized. */
  queryText: string;
  description: string;
};

export type GroundingSpatialOrder = {
  axis: "x" | "y";
  /** Ascending: left-to-right / top-to-bottom in the visible source frame. */
  direction: "ascending" | "descending";
  ordinal: number;
  /** Membership is declared by the model. Array/discovery order is irrelevant. */
  candidateIds: string[];
  selectedCandidateId: string;
  /** Whether all relevant members of this ordering set have been established. */
  candidateSet: GroundingSupport;
};

export type GroundingQueryInterpretation = GroundingSupport & {
  id: string;
  reading: string;
  requirements: GroundingQueryRequirement[];
  spatialOrder?: GroundingSpatialOrder;
};

export type GroundingConstraintContract = {
  /** Must equal the loaded record's query byte-for-byte, including whitespace. */
  originalQuery: string;
  /** Model attestation that no attribute, relation, order or reading was dropped. */
  queryCoverage: GroundingSupport;
  interpretations: GroundingQueryInterpretation[];
  candidates: GroundingConstraintCandidate[];
  selectedCandidateId?: string;
};

export const GROUNDING_CONSTRAINT_LIMITS = {
  serializedBytes: 32 * 1024,
  queryCharacters: 12000,
  queryExcerptCharacters: 600,
  textCharacters: 600,
  evidenceCharacters: 800,
  idCharacters: 80,
  candidates: 32,
  interpretations: 6,
  requirements: 12,
} as const;

export type GroundingConstraintIssue = { code: string; message: string };
/** Declared rectangle geometry only, never detected object/foreground coverage. */
export type GroundingProposalGeometry = {
  selectedCandidateId: string;
  proposedBbox: GroundingConstraintBox;
  selectedBoxCoverage: number;
  proposalInsideSelectedBox: number;
  otherCandidateOverlaps: { id: string; candidateBoxCoverage: number; proposalBoxCoverage: number }[];
};
export type GroundingOrderAssessment = {
  interpretationId: string;
  axis: GroundingSpatialOrder["axis"];
  direction: GroundingSpatialOrder["direction"];
  ordinal: number;
  orderedCandidateIds: string[];
  supportedCandidateIds: string[];
  supportedCount: number;
  possibleCount: number;
  selectedCandidateId: string;
  /** Present only when source-center ordering and every counted identity agree. */
  selectedRank?: number;
  /** Range among declared candidates, not a claim that the scene is exhaustive. */
  selectedRankRange?: [number, number];
  tiedCandidateIds: string[];
  /** Runtime-resolved evidence; never accepted from a model contract. */
  geometryIssues: GroundingCandidateGeometryIssue[];
};

export type GroundingConstraintAssessment = {
  status: GroundingSupportStatus;
  canLock: boolean;
  requiresHumanReview: true;
  issues: GroundingConstraintIssue[];
  orders: GroundingOrderAssessment[];
  /** Advisory: overlap is legitimate for occluded objects and never blocks review. */
  proposalGeometry?: GroundingProposalGeometry;
  selectedCandidate?: {
    id: string;
    sourceBbox: GroundingConstraintBox;
    identityStatus: GroundingSupportStatus;
  };
  limitations: string[];
};

function object(input: unknown, field: string, keys: readonly string[]): Record<string, unknown> {
  if (input === null || typeof input !== "object" || Array.isArray(input)) {
    throw new Error(`Grounding contract ${field} must be an object.`);
  }
  const value = input as Record<string, unknown>;
  for (const key of Object.keys(value)) {
    if (!keys.includes(key)) {
      const interpretationPath = field.match(/^interpretations\[\d+\]/)?.[0] ?? "interpretations[0]";
      const hint = key === "spatialOrder"
        ? ` Put spatialOrder at contract.${interpretationPath}.spatialOrder, beside requirements (not inside requirements or at contract root). Shape example only: {"axis":"x","direction":"ascending","ordinal":2,"candidateIds":["existing-id"],"selectedCandidateId":"existing-id","candidateSet":{"status":"unresolved","evidence":"Membership still uncertain"}}. Use actual existing candidate IDs and query-supported axis/direction/ordinal; this example is not evidence. Keep observations and unresolved questions; repair only the nesting.`
        : key === "queryText" && /^interpretations\[\d+\]$/.test(field)
          ? ` Put queryText at contract.${field}.requirements[i].queryText, not on the interpretation. Requirement fields: id, queryText, description, status, evidence. Keep the existing observations; repair only this field.`
          : ` Allowed fields: ${keys.join(", ")}.`;
      throw new Error(`Unknown grounding contract ${field} field: ${key}.${hint}`);
    }
  }
  return value;
}

function text(input: unknown, field: string, limit: number, preserve = false, allowEmpty = false): string {
  if (typeof input !== "string" || input.length > limit || (!allowEmpty && !input.trim())) {
    throw new Error(`Grounding contract ${field} must be ${allowEmpty ? "a" : "a nonempty"} string of at most ${limit} characters.`);
  }
  return preserve ? input : input.trim();
}

function list(input: unknown, field: string, limit: number): unknown[] {
  if (!Array.isArray(input) || input.length > limit) {
    throw new Error(`Grounding contract ${field} must be an array of at most ${limit} items.`);
  }
  return input;
}

function status(input: unknown, field: string): GroundingSupportStatus {
  if (input !== "supported" && input !== "unresolved" && input !== "contradicted") {
    throw new Error(`Grounding contract ${field} must be supported, unresolved or contradicted.`);
  }
  return input;
}

function support(input: unknown, field: string): GroundingSupport {
  const value = object(input, field, ["status", "evidence"]);
  return {
    status: status(value.status, `${field}.status`),
    evidence: text(value.evidence, `${field}.evidence`, GROUNDING_CONSTRAINT_LIMITS.evidenceCharacters),
  };
}

function identifier(input: unknown, field: string): string {
  return text(input, field, GROUNDING_CONSTRAINT_LIMITS.idCharacters);
}

function box(input: unknown, field: string): GroundingConstraintBox {
  if (!Array.isArray(input) || input.length !== 4
    || input.some((edge) => typeof edge !== "number" || !Number.isFinite(edge) || edge < 0 || edge > 1)
    || input[0] >= input[2] || input[1] >= input[3]) {
    throw new Error(`Grounding contract ${field} must be a valid normalized source bbox.`);
  }
  return [...input] as GroundingConstraintBox;
}

function unique(ids: readonly string[], field: string): void {
  if (new Set(ids).size !== ids.length) throw new Error(`Grounding contract ${field} must contain unique ids.`);
}

/** Strict bounded structure; incomplete knowledge uses explicit unresolved states. */
export function validateGroundingConstraintContract(input: unknown, originalQuery?: string): GroundingConstraintContract {
  const value = object(input, "root", ["originalQuery", "queryCoverage", "interpretations", "candidates", "selectedCandidateId"]);
  // Tool callers may omit the redundant query: only the loaded record supplies
  // it. A provided mismatching (or malformed) query is never silently replaced.
  const query = text(Object.hasOwn(value, "originalQuery") ? value.originalQuery : originalQuery,
    "originalQuery", GROUNDING_CONSTRAINT_LIMITS.queryCharacters, true, true);
  let serialized: string;
  try {
    serialized = JSON.stringify({ ...value, originalQuery: query });
  } catch {
    throw new Error("Grounding contract must be JSON-serializable.");
  }
  if (new TextEncoder().encode(serialized).byteLength > GROUNDING_CONSTRAINT_LIMITS.serializedBytes) {
    throw new Error(`Grounding contract exceeds the ${GROUNDING_CONSTRAINT_LIMITS.serializedBytes}-byte serialized UTF-8 limit. Shorten evidence, excerpts or candidate declarations; preserve originalQuery exactly and keep unresolved gaps explicit.`);
  }
  if (originalQuery !== undefined && query !== originalQuery) {
    throw new Error("Grounding contract originalQuery must exactly match the loaded original query; preserve every word and ambiguity.");
  }
  const candidates = list(value.candidates, "candidates", GROUNDING_CONSTRAINT_LIMITS.candidates).map((input, index) => {
    const field = `candidates[${index}]`;
    const candidate = object(input, field, ["id", "bbox", "identity", "measurementViewId"]);
    const identity = object(candidate.identity, `${field}.identity`, ["label", "status", "evidence", "basis"]);
    const basis = identity.basis;
    if (basis !== "visual_structure" && basis !== "pixel_measurement" && basis !== "repeated_view" && basis !== "unknown") {
      throw new Error(`Grounding contract ${field}.identity.basis must be visual_structure, pixel_measurement, repeated_view or unknown.`);
    }
    return {
      id: identifier(candidate.id, `${field}.id`),
      bbox: box(candidate.bbox, `${field}.bbox`),
      ...(Object.hasOwn(candidate, "measurementViewId") ? { measurementViewId: identifier(candidate.measurementViewId, `${field}.measurementViewId`) } : {}),
      identity: {
        label: text(identity.label, `${field}.identity.label`, GROUNDING_CONSTRAINT_LIMITS.textCharacters),
        status: status(identity.status, `${field}.identity.status`),
        evidence: text(identity.evidence, `${field}.identity.evidence`, GROUNDING_CONSTRAINT_LIMITS.evidenceCharacters),
        basis,
      },
    } satisfies GroundingConstraintCandidate;
  });
  unique(candidates.map((item) => item.id), "candidates");
  const candidateIds = new Set(candidates.map((item) => item.id));
  const reference = (input: unknown, field: string) => {
    const id = identifier(input, field);
    if (!candidateIds.has(id)) throw new Error(`Grounding contract ${field} references unknown candidate ${id}.`);
    return id;
  };
  const interpretations = list(value.interpretations, "interpretations", GROUNDING_CONSTRAINT_LIMITS.interpretations).map((input, index) => {
    const field = `interpretations[${index}]`;
    const interpretation = object(input, field, ["id", "reading", "status", "evidence", "requirements", "spatialOrder"]);
    const requirements = list(interpretation.requirements, `${field}.requirements`, GROUNDING_CONSTRAINT_LIMITS.requirements).map((input, index) => {
      const path = `${field}.requirements[${index}]`;
      const requirement = object(input, path, ["id", "queryText", "description", "status", "evidence"]);
      const queryText = text(requirement.queryText, `${path}.queryText`, GROUNDING_CONSTRAINT_LIMITS.queryExcerptCharacters, true, !query.trim());
      if (!query.includes(queryText)) {
        // Suggest the source's exact spelling without rewriting the model's
        // contract or asking it to inspect the same image again.
        const escaped = queryText.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const sourceMatch = new RegExp(escaped, "iu").exec(query)?.[0];
        const repair = sourceMatch === undefined
          ? "Copy a contiguous, case-sensitive excerpt from originalQuery."
          : `Replace only this queryText with ${JSON.stringify(sourceMatch)}; capitalization must match the source.`;
        throw new Error(`Grounding contract ${path}.queryText must be an exact excerpt of originalQuery. ${repair} Keep the current observations and proposal; no new image call is needed for this field repair.`);
      }
      return {
        id: identifier(requirement.id, `${path}.id`),
        queryText,
        description: text(requirement.description, `${path}.description`, GROUNDING_CONSTRAINT_LIMITS.textCharacters),
        status: status(requirement.status, `${path}.status`),
        evidence: text(requirement.evidence, `${path}.evidence`, GROUNDING_CONSTRAINT_LIMITS.evidenceCharacters),
      };
    });
    unique(requirements.map((item) => item.id), `${field}.requirements`);
    let spatialOrder: GroundingSpatialOrder | undefined;
    if (Object.hasOwn(interpretation, "spatialOrder")) {
      const path = `${field}.spatialOrder`;
      const order = object(interpretation.spatialOrder, path, ["axis", "direction", "ordinal", "candidateIds", "selectedCandidateId", "candidateSet"]);
      if (order.axis !== "x" && order.axis !== "y") throw new Error(`Grounding contract ${path}.axis must be x or y.`);
      if (order.direction !== "ascending" && order.direction !== "descending") throw new Error(`Grounding contract ${path}.direction must be ascending or descending.`);
      if (typeof order.ordinal !== "number" || !Number.isSafeInteger(order.ordinal) || order.ordinal < 1) {
        throw new Error(`Grounding contract ${path}.ordinal must be a positive safe integer.`);
      }
      const members = list(order.candidateIds, `${path}.candidateIds`, GROUNDING_CONSTRAINT_LIMITS.candidates)
        .map((id, index) => reference(id, `${path}.candidateIds[${index}]`));
      unique(members, `${path}.candidateIds`);
      spatialOrder = {
        axis: order.axis, direction: order.direction, ordinal: order.ordinal, candidateIds: members,
        selectedCandidateId: reference(order.selectedCandidateId, `${path}.selectedCandidateId`),
        candidateSet: support(order.candidateSet, `${path}.candidateSet`),
      };
    }
    return {
      id: identifier(interpretation.id, `${field}.id`),
      reading: text(interpretation.reading, `${field}.reading`, GROUNDING_CONSTRAINT_LIMITS.textCharacters),
      status: status(interpretation.status, `${field}.status`),
      evidence: text(interpretation.evidence, `${field}.evidence`, GROUNDING_CONSTRAINT_LIMITS.evidenceCharacters),
      requirements,
      ...(spatialOrder ? { spatialOrder } : {}),
    };
  });
  unique(interpretations.map((item) => item.id), "interpretations");
  return {
    originalQuery: query, queryCoverage: support(value.queryCoverage, "queryCoverage"), interpretations, candidates,
    ...(Object.hasOwn(value, "selectedCandidateId") ? { selectedCandidateId: reference(value.selectedCandidateId, "selectedCandidateId") } : {}),
  };
}

const ORDER_EPSILON = 1e-6;
const establishedIdentity = (candidate: GroundingConstraintCandidate) => candidate.identity.status === "supported"
  && candidate.identity.basis === "visual_structure";

export function assessGroundingProposalGeometry(
  candidates: readonly GroundingConstraintCandidate[],
  selectedCandidateId: string | undefined,
  proposedBbox: readonly number[],
): GroundingProposalGeometry | undefined {
  const selected = candidates.find((candidate) => candidate.id === selectedCandidateId);
  if (!selected) return undefined;
  const proposed = box(proposedBbox, "selectionBbox");
  const area = (bbox: readonly number[]) => (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
  const intersection = (bbox: readonly number[]) => Math.max(0, Math.min(proposed[2], bbox[2]) - Math.max(proposed[0], bbox[0]))
    * Math.max(0, Math.min(proposed[3], bbox[3]) - Math.max(proposed[1], bbox[1]));
  const selectedIntersection = intersection(selected.bbox);
  return {
    selectedCandidateId: selected.id, proposedBbox: proposed,
    selectedBoxCoverage: selectedIntersection / area(selected.bbox),
    proposalInsideSelectedBox: selectedIntersection / area(proposed),
    otherCandidateOverlaps: candidates.filter((candidate) => candidate.id !== selected.id)
      .flatMap((candidate) => {
        const overlap = intersection(candidate.bbox);
        return overlap > 0 ? [{ id: candidate.id, candidateBoxCoverage: overlap / area(candidate.bbox),
          proposalBoxCoverage: overlap / area(proposed) }] : [];
      }),
  };
}

/** Deterministic ordering of a validated contract; no image recognition or language parsing. */
export function assessGroundingSpatialOrder(
  order: GroundingSpatialOrder,
  candidates: readonly GroundingConstraintCandidate[],
  interpretationId = "",
  provenanceContext?: GroundingCandidateProvenanceContext,
): GroundingOrderAssessment {
  unique(candidates.map((candidate) => candidate.id), "candidates");
  unique(order.candidateIds, "spatialOrder.candidateIds");
  const candidateMap = new Map(candidates.map((candidate) => [candidate.id, candidate]));
  const coordinate = (candidate: GroundingConstraintCandidate) => {
    const axis = order.axis === "x" ? 0 : 1;
    return (candidate.bbox[axis] + candidate.bbox[axis + 2]) / 2 * (order.direction === "ascending" ? 1 : -1);
  };
  const members = order.candidateIds.map((id) => {
    const candidate = candidateMap.get(id);
    if (!candidate) throw new Error(`Grounding contract spatialOrder references unknown candidate ${id}.`);
    return candidate;
  });
  const possible = members.filter((item) => item.identity.status !== "contradicted")
    .sort((a, b) => coordinate(a) - coordinate(b) || a.id.localeCompare(b.id));
  const geometryIssues = assessGroundingCandidateProvenance(members, provenanceContext);
  const geometryResolved = geometryIssues.length === 0;
  const unresolvedGeometryIds = new Set(geometryIssues.map((issue) => issue.candidateId));
  const supported = possible.filter((candidate) => establishedIdentity(candidate) && !unresolvedGeometryIds.has(candidate.id));
  const selected = possible.find((item) => item.id === order.selectedCandidateId);
  const tied = selected && geometryResolved ? possible.filter((item) => item.id !== selected.id && Math.abs(coordinate(item) - coordinate(selected)) <= ORDER_EPSILON) : [];
  const range: [number, number] | undefined = selected && geometryResolved ? [
    1 + supported.filter((item) => coordinate(item) < coordinate(selected) - ORDER_EPSILON).length,
    1 + possible.filter((item) => item.id !== selected.id && coordinate(item) <= coordinate(selected) + ORDER_EPSILON).length,
  ] : undefined;
  const selectedRank = geometryResolved && selected && establishedIdentity(selected) && supported.length === possible.length && tied.length === 0
    ? possible.findIndex((item) => item.id === selected.id) + 1 : undefined;
  return {
    interpretationId, axis: order.axis, direction: order.direction, ordinal: order.ordinal,
    orderedCandidateIds: geometryResolved ? possible.map((item) => item.id) : [], supportedCandidateIds: supported.map((item) => item.id),
    supportedCount: supported.length, possibleCount: possible.length, selectedCandidateId: order.selectedCandidateId,
    ...(selectedRank === undefined ? {} : { selectedRank }), ...(range ? { selectedRankRange: range } : {}),
    tiedCandidateIds: tied.map((item) => item.id), geometryIssues,
  };
}

/**
 * Absent legacy contracts and unresolved evidence remain eligible for HUMAN
 * review. Only a fully supported declared contract can establish a new lock.
 * Nothing here changes model evidence because a view was repeated or enlarged.
 */
export function assessGroundingConstraints(
  contract: GroundingConstraintContract | undefined,
  originalQuery: string,
  selectionBbox?: readonly number[],
  provenanceContext?: GroundingCandidateProvenanceContext,
): GroundingConstraintAssessment {
  const issues: GroundingConstraintIssue[] = [];
  const orders: GroundingOrderAssessment[] = [];
  let contradicted = false;
  const issue = (code: string, message: string, contradiction = false) => {
    issues.push({ code, message });
    contradicted ||= contradiction;
  };
  const inspectSupport = (claim: GroundingSupport, code: string, label: string) => {
    if (claim.status !== "supported") issue(code, `${label} is ${claim.status}: ${claim.evidence}`, claim.status === "contradicted");
  };
  const limitations = [
    "Only declared source-box geometry and contract consistency are checked mechanically; object identity is model-declared, not machine-verified.",
    "Query coverage, semantic interpretation of direction/order and candidate-set completeness are model-declared; no language parser or detector proves that every query requirement or scene object was included or interpreted correctly.",
    "Pixel measurements, magnification and repeated same-source-pixel views do not establish identity or resolve query ambiguity. Human review is always required.",
  ];
  if (!contract) {
    issue("missing_contract", "No original-query contract is recorded. Attach contract directly to the save call (or state.contract to grounding_evidence); originalQuery may be omitted and will use the loaded query. Keep requirements[].queryText as exact query excerpts. No extra image call is needed; unsupported claims remain unresolved for human review.");
    return { status: "unresolved", canLock: false, requiresHumanReview: true, issues, orders, limitations };
  }
  // Revalidate callers using persisted or plain JavaScript data, without ever
  // correcting the query to fit the candidate.
  contract = validateGroundingConstraintContract(contract);
  if (contract.originalQuery !== originalQuery) issue("original_query_mismatch", "The contract does not exactly match the loaded original query. Preserve its original wording and all plausible readings.", true);
  if (!originalQuery.trim()) issue("missing_query", "The original query is empty; the requested target remains unresolved.");
  inspectSupport(contract.queryCoverage, "query_coverage", "Coverage of the complete original query");
  const plausible = contract.interpretations.filter((item) => item.status !== "contradicted");
  if (plausible.length !== 1) issue("ambiguous_interpretations", `There are ${plausible.length} plausible readings; preserve alternatives until the requested reading is established.`, plausible.length === 0 && contract.interpretations.length > 0);
  const selected = contract.candidates.find((item) => item.id === contract.selectedCandidateId);
  if (!selected) issue("missing_selection", "No selected candidate is recorded in the contract.");
  else {
    inspectSupport(selected.identity, "candidate_identity", `Identity of selected candidate ${selected.id}`);
    if (selected.identity.basis !== "visual_structure") issue("identity_basis", `Candidate ${selected.id} relies on ${selected.identity.basis}; color measurements and repeated views cannot establish object identity.`);
    if (selectionBbox !== undefined) {
      const proposed = box(selectionBbox, "selectionBbox");
      if (Math.min(proposed[2], selected.bbox[2]) <= Math.max(proposed[0], selected.bbox[0])
        || Math.min(proposed[3], selected.bbox[3]) <= Math.max(proposed[1], selected.bbox[1])) {
        issue("selection_bbox_mismatch", `The proposed source bbox does not intersect selected candidate ${selected.id}; reconcile the selected identity before locking.`, true);
      }
    }
  }
  for (const interpretation of contract.interpretations) {
    const order = interpretation.spatialOrder;
    const assessment = order ? assessGroundingSpatialOrder(order, contract.candidates, interpretation.id, provenanceContext) : undefined;
    if (assessment) orders.push(assessment);
    // Rejected alternative readings remain visible without invalidating a
    // supported reading. Their rejection evidence is required by the schema.
    if (interpretation.status === "contradicted") continue;
    inspectSupport(interpretation, "interpretation_unresolved", `Reading ${interpretation.id}`);
    if (interpretation.requirements.length === 0) issue("missing_requirements", `Reading ${interpretation.id} has no explicit query requirements.`);
    for (const requirement of interpretation.requirements) {
      inspectSupport(requirement, "requirement_unresolved", `Requirement ${interpretation.id}/${requirement.id} (${requirement.queryText})`);
    }
    if (!order || !assessment) continue;
    inspectSupport(order.candidateSet, "candidate_set_unresolved", `Candidate membership/completeness for reading ${interpretation.id}`);
    if (order.selectedCandidateId !== contract.selectedCandidateId) issue("selected_candidate_mismatch", `Reading ${interpretation.id} selects ${order.selectedCandidateId}, but the contract selects ${contract.selectedCandidateId ?? "none"}.`, true);
    for (const geometryIssue of assessment.geometryIssues) issue("ordering_geometry_unresolved", `Reading ${interpretation.id}: ${geometryIssue.message}`);
    if (!order.candidateIds.includes(order.selectedCandidateId) || contract.candidates.find((candidate) => candidate.id === order.selectedCandidateId)?.identity.status === "contradicted") issue("selected_candidate_not_in_order", `Selected candidate ${order.selectedCandidateId} is absent from the viable ordering set.`, true);
    if (assessment.possibleCount < order.ordinal) issue("insufficient_candidates", `Reading ${interpretation.id} requests rank ${order.ordinal}, but only ${assessment.possibleCount} possible candidates (${assessment.supportedCount} supported) are declared. Do not relabel the last visible candidate as the missing rank or invent hidden members. Inspect further only for a specific observed ambiguity; otherwise submit unresolved for human review, or use grounding_evidence clarification to ask about the missing referent without proposing a box.`);
    if (order.candidateIds.some((id) => !establishedIdentity(contract.candidates.find((candidate) => candidate.id === id)!))) issue("ordering_identity_unresolved", `Reading ${interpretation.id} has unresolved, contradicted or measurement-only candidate identities; they cannot establish the requested count/order.`);
    if (assessment.tiedCandidateIds.length > 0) issue("spatial_tie", `Selected candidate ${order.selectedCandidateId} has indistinguishable ${order.axis}-centers with ${assessment.tiedCandidateIds.join(", ")}; no unique requested rank is established.`);
    if (assessment.selectedRank !== undefined && assessment.selectedRank !== order.ordinal) {
      issue("rank_mismatch", `Source-center geometry places ${order.selectedCandidateId} at rank ${assessment.selectedRank}, not requested rank ${order.ordinal}, among the declared candidates.`,
        order.candidateSet.status === "supported" || assessment.selectedRank > order.ordinal);
    } else if (assessment.selectedRankRange && (order.ordinal < assessment.selectedRankRange[0]
      || (order.candidateSet.status === "supported" && order.ordinal > assessment.selectedRankRange[1]))) {
      issue("rank_mismatch", `Requested rank ${order.ordinal} lies outside ${order.selectedCandidateId}'s declared-candidate rank range ${assessment.selectedRankRange.join("–")}.`, true);
    }
  }
  return {
    status: contradicted ? "contradicted" : issues.length ? "unresolved" : "supported",
    canLock: issues.length === 0, requiresHumanReview: true, issues, orders, limitations,
    ...(selected && selectionBbox !== undefined ? {
      proposalGeometry: assessGroundingProposalGeometry(contract.candidates, selected.id, selectionBbox),
    } : {}),
    ...(selected ? { selectedCandidate: { id: selected.id, sourceBbox: [...selected.bbox] as GroundingConstraintBox, identityStatus: selected.identity.status } } : {}),
  };
}
