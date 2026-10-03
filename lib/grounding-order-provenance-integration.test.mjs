import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import sharp from "sharp";
import { createJiti } from "jiti";
const { createGroundingSafetyExtension } = await createJiti(import.meta.url).import("./grounding-safety-extension.ts");

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "grounding-order-provenance-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, "source"), outputDir = join(root, "output");
  await mkdir(source);
  const pixels = await sharp({ create: { width: 80, height: 60, channels: 3, background: "#778899" } }).png().toBuffer();
  await writeFile(join(source, "visible.png"), pixels);
  await writeFile(join(source, "infrared.png"), pixels);
  const queryPath = join(source, "queries.json"), query = "the third drone from the left";
  await writeFile(queryPath, JSON.stringify({ one: { query, visible: "visible.png", infrared: "infrared.png" } }));
  const handlers = new Map(), tools = new Map();
  let active = ["read", "write", "bash"], calls = 0;
  createGroundingSafetyExtension({ cwd: root, sessionId: "provenance-test" }).factory({
    on(name, handler) { handlers.set(name, handler); },
    registerTool(tool) { tools.set(tool.name, tool); active.push(tool.name); },
    getActiveTools: () => active, setActiveTools(names) { active = names; },
    appendEntry() {}, sendMessage() {}, sendUserMessage() {}, setModel: async () => true,
  });
  const call = (name, params, context) => tools.get(name).execute(`${name}-${++calls}`, params, undefined, undefined, context);
  await handlers.get("before_agent_start")({ prompt: `批处理 1 条图像定位样本\n${queryPath}`, systemPromptOptions: { sections: {} } });
  const loaded = await call("grounding_next_batch", { queryPath, outputDir, targetCount: 1 });
  const visibleId = loaded.details.evidenceViewIds[0];
  const infrared = await call("grounding_view", { modality: "infrared", reason: "Check independent sensor identity structure in its original overview." });
  const infraredId = infrared.details.evidenceViewIds[0];
  const support = { status: "supported", evidence: "Synthetic declaration for validator testing, not actual image recognition." };
  const contract = {
    originalQuery: query, queryCoverage: support, selectedCandidateId: "right",
    candidates: [["left", .2], ["middle", .45], ["right", .7]].map(([id, x]) => ({ id, bbox: [x, .2, x + .1, .4],
      measurementViewId: id === "middle" ? infraredId : visibleId,
      identity: { label: "drone", basis: "visual_structure", ...support } })),
    interpretations: [{ id: "order", reading: query, ...support,
      requirements: [{ id: "rank", queryText: query, description: "Third drone in visible x order", ...support }],
      spatialOrder: { axis: "x", direction: "ascending", ordinal: 3, candidateIds: ["right", "left", "middle"],
        selectedCandidateId: "right", candidateSet: support } }],
  };
  const params = { queryPath, outputDir, key: "one", bbox: [.7, .2, .8, .4], status: "ok", confidence: .95,
    reason: "Synthetic candidate for testing that unregistered sensor geometry remains unresolved." };
  return { call, query, queryPath, outputDir, visibleId, infraredId, contract, params };
}

function review(callback, response = {}) {
  return { isIdle: () => true, abort() {}, ui: { custom: async (factory) => {
    const details = factory({}, {}, {}, () => {}).groundingReview;
    callback(details);
    return { type: "grounding_review_response", action: "confirm", bbox: details.bbox, status: "unresolved", confidence: .4,
      targetFound: true, candidateCount: 3, candidateRank: 3, reason: "Human test reviewer retains unresolved sensor provenance.", ...response };
  } } };
}

const assertUnresolved = (assessment) => {
  assert.equal(assessment.canLock, false);
  assert.equal(assessment.status, "unresolved");
  assert.equal(assessment.orders[0].selectedRank, undefined);
  assert.ok(assessment.issues.some((issue) => issue.code === "ordering_geometry_unresolved"));
};

test("evidence and preview defer cross-modal locks; an existing visible overview repairs provenance", async (t) => {
  const f = await fixture(t);
  const evidence = await f.call("grounding_evidence", { state: { contract: f.contract,
    selection: { status: "locked", bbox: f.params.bbox, evidence: "The submitted declarations claim third rank." } } });
  assert.equal(evidence.details.lockDeferred, true);
  assert.equal(evidence.details.state.selection.status, "reconsidering");
  assertUnresolved(evidence.details.constraintAssessment);
  const preview = await f.call("grounding_save_result", { ...f.params, contract: f.contract, previewOnly: true });
  assertUnresolved(preview.details.constraintAssessment);
  assert.equal(preview.details.status, "unresolved");
  assert.ok(preview.details.confidence <= .49);
  f.contract.candidates[1].measurementViewId = f.visibleId;
  const repaired = await f.call("grounding_evidence", { state: { contract: f.contract,
    selection: { status: "locked", bbox: f.params.bbox, evidence: "All measurements reference the inspected visible overview." } } });
  assert.equal(repaired.details.lockDeferred, false);
  assert.equal(repaired.details.constraintAssessment.canLock, true);
  assert.equal(repaired.details.constraintAssessment.orders[0].selectedRank, 3);
});

test("save review retains authoritative provenance issues and unresolved confidence", async (t) => {
  const f = await fixture(t);
  let sawReview = false;
  await f.call("grounding_save_result", { ...f.params, contract: f.contract }, review((details) => {
    sawReview = true;
    assertUnresolved(details.constraintAssessment);
    assert.equal(details.status, "unresolved");
    assert.ok(details.confidence <= .49);
  }));
  assert.equal(sawReview, true);
  const output = JSON.parse((await readFile(join(f.outputDir, "progress.jsonl"), "utf8")).trim());
  assertUnresolved(output.constraintAssessment);
  assert.equal(output.status, "unresolved");
  assert.ok(output.confidence <= .49);
});

test("a browser response cannot silently promote unresolved provenance without human confirmation", async (t) => {
  const f = await fixture(t);
  await assert.rejects(f.call("grounding_save_result", { ...f.params, contract: f.contract }, review((details) => {
    assertUnresolved(details.constraintAssessment);
  }, { status: "ok", confidence: .95 })), /unless the human reviewer explicitly confirms/);
});
