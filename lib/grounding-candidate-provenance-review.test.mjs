import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";
const jiti = createJiti(import.meta.url);
const { assessGroundingConstraints, validateGroundingConstraintContract } = await jiti.import("./grounding-constraints.ts");
const { GroundingViewRegistry } = await jiti.import("./grounding-views.ts");
const supported = { status: "supported", evidence: "The original visible structure supports this declaration; this is not machine recognition." };
const view = (modality = "visible", extra = {}) => ({ modality, region: [0, 0, 1, 1], sourceWidth: 1000, sourceHeight: 800, width: 500, height: 400, ...extra });
function fixture() {
  const registry = new GroundingViewRegistry();
  const visible = registry.register(view());
  const value = {
    originalQuery: "The second drone from left to right", queryCoverage: { ...supported }, selectedCandidateId: "right",
    candidates: [
      { id: "left", bbox: [.1, .2, .3, .4], measurementViewId: visible.id, identity: { ...supported, label: "drone", basis: "visual_structure" } },
      { id: "right", bbox: [.6, .2, .8, .4], measurementViewId: visible.id, identity: { ...supported, label: "drone", basis: "visual_structure" } },
    ],
    interpretations: [{ id: "reading", reading: "Second visible drone in ascending horizontal order", ...supported,
      requirements: [{ id: "rank", queryText: "second drone from left to right", description: "Identity and spatial order", ...supported }],
      spatialOrder: { axis: "x", direction: "ascending", ordinal: 2, candidateIds: ["right", "left"], selectedCandidateId: "right", candidateSet: { ...supported } },
    }],
  };
  return { registry, visible, value };
}
const assess = (value, registry) => assessGroundingConstraints(value, value.originalQuery, undefined, registry);

test("malformed measurement references fail strict parsing rather than become trusted provenance", () => {
  for (const invalid of [null, 5, {}, [], "", " ", "x".repeat(81)]) {
    const { value } = fixture();
    value.candidates[0].measurementViewId = invalid;
    assert.throws(() => validateGroundingConstraintContract(value), /measurementViewId/);
  }
});

test("candidate-supplied modality, record, or transform cannot authorize source geometry", () => {
  for (const [field, data] of Object.entries({ modality: "visible", recordId: "current", transform: [1, 0, 0, 1], provenance: { registered: true } })) {
    const { value, registry } = fixture();
    value.candidates[0][field] = data;
    assert.throws(() => assess(value, registry), new RegExp(`Unknown grounding contract candidates\\[0\\] field: ${field}`));
  }
});

test("a syntactically valid fabricated view reference remains unresolved", () => {
  const { value, registry } = fixture();
  value.candidates[0].measurementViewId = "claimed-visible-current-record";
  const result = assess(value, registry);
  assert.equal(result.canLock, false);
  assert.equal(result.orders[0].selectedRank, undefined);
  assert.equal(result.orders[0].geometryIssues[0].code, "unavailable_measurement_view");
});

test("supplementary IR identity discussion preserves visible-only positive ordering", () => {
  const { value, registry } = fixture();
  registry.register(view("infrared"));
  value.candidates[0].identity.evidence = "Visible rotor and body structure identify a drone; infrared is supplementary and is not registered geometry.";
  const result = assess(value, registry);
  assert.equal(result.canLock, true);
  assert.equal(result.orders[0].selectedRank, 2);
  assert.deepEqual(result.orders[0].geometryIssues, []);
});

test("source-mapped derived panels support measurement, never upgrade identity", () => {
  const { value, registry } = fixture();
  const mask = registry.register(view("visible", { region: [.05, .1, .35, .5], derived: { kind: "image_processing", operation: "threshold", role: "measurement_only", establishesObjectIdentity: false } }));
  value.candidates[0].measurementViewId = mask.id;
  assert.equal(assess(value, registry).canLock, true);
  value.candidates[0].identity.basis = "pixel_measurement";
  const measuredOnly = assess(value, registry);
  assert.equal(measuredOnly.canLock, false);
  assert.ok(measuredOnly.issues.some((issue) => issue.code === "ordering_identity_unresolved"));
  value.candidates[0].identity.basis = "visual_structure";
  value.candidates[0].bbox = [.01, .2, .3, .4];
  const beyondMask = assess(value, registry);
  assert.equal(beyondMask.canLock, false);
  assert.equal(beyondMask.orders[0].geometryIssues[0].code, "measurement_view_coverage");
});
