import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import sharp from "sharp";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url);
const { createGroundingSafetyExtension } = await jiti.import("./grounding-safety-extension.ts");
const { validateGroundingWorkingState } = await jiti.import("./grounding-evidence.ts");
const { advanceGroundingDecision, groundingDecisionCheckpoint } = await jiti.import("./grounding-decision-checkpoint.ts");
const { assessGroundingConstraints } = await jiti.import("./grounding-constraints.ts");

function trial(outcome = "inconclusive", nextObservation = "Check whether the lower edge joins the visible outline") {
  return { question: "Does the local mask separate the target from its background?", outcome,
    observation: "The mask includes both the target and adjacent background",
    remainingUnknown: "The complete lower boundary is unresolved", nextObservation };
}

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "grounding-trial-outcomes-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sourceDir = join(root, "source");
  const outputDir = join(root, "output");
  await mkdir(sourceDir);
  const pixels = Buffer.from(Array.from({ length: 80 * 60 * 3 }, (_, i) => i % 251));
  await writeFile(join(sourceDir, "visible.png"), await sharp(pixels, { raw: { width: 80, height: 60, channels: 3 } }).png().toBuffer());
  const queryPath = join(sourceDir, "queries.json");
  await writeFile(queryPath, JSON.stringify(Object.fromEntries(["one", "two"].map(key => [key, {
    query: `the synthetic object ${key}`, visible: "visible.png",
  }]))));
  const handlers = new Map();
  const tools = new Map();
  let active = [];
  let id = 0;
  createGroundingSafetyExtension({ cwd: root, sessionId: `trials-${root}` }).factory({
    on(name, handler) { handlers.set(name, handler); },
    registerTool(tool) { tools.set(tool.name, tool); active.push(tool.name); },
    getActiveTools: () => [...active], setActiveTools(names) { active = [...names]; },
    appendEntry() {}, sendMessage() {}, sendUserMessage() {}, setModel: async () => true,
  });
  const call = (name, params = {}, context) => tools.get(name).execute(`trial-${++id}`, params, undefined, undefined, context);
  await handlers.get("before_agent_start")({ prompt: `批处理 2 条图像定位样本 ${queryPath}`, systemPromptOptions: { sections: {} } });
  const loaded = await call("grounding_next_batch", { queryPath, outputDir });
  return { handlers, tools, call, loaded, outputDir, params: { queryPath, outputDir, key: "one", bbox: [.25, .2, .75, .8],
    status: "ok", confidence: .99, reason: "Synthetic candidate proposed for explicit human review." } };
}

function conversation(id, name, result) {
  return [
    { role: "assistant", content: [{ type: "toolCall", id, name, arguments: {} }] },
    { role: "toolResult", toolCallId: id, toolName: name, content: result.content, details: result.details, isError: false },
  ];
}
const images = result => result.content.filter(block => block.type === "image");
const measurementCalls = [
  ["grounding_process_image", { region: [.1,.1,.9,.9], operations: [{ kind: "edges" }], reason: "Check the visible lower boundary against local edges" }],
  ["grounding_refine_box", { region: [.1,.1,.9,.9], coarseBox: [.25,.2,.75,.8], reason: "Check the visible lower boundary against edge candidates" }],
  ["grounding_color_region", { region: [.1,.1,.9,.9], color: "black" }],
];

for (const outcome of ["useful", "inconclusive", "contradictory", "failed"]) {
  test(`a ${outcome} trial is an optional bounded declaration`, () => {
    assert.deepEqual(validateGroundingWorkingState({ lastTrial: trial(outcome) }), { lastTrial: trial(outcome) });
  });
}

test("trial validation rejects malformed receipts without making ordinary notes mandatory", () => {
  assert.deepEqual(validateGroundingWorkingState({}), {});
  assert.deepEqual(validateGroundingWorkingState({ facts: [] }), { facts: [] });
  assert.equal(validateGroundingWorkingState({ lastTrial: trial("failed", null) }).lastTrial.nextObservation, null);
  for (const patch of [{ outcome: "verified" }, { question: "" }, { observation: " " }, { nextObservation: "" },
    { nextObservation: 42 }, { remainingUnknown: "x".repeat(401) }, { toolSuccess: true }]) {
    assert.throws(() => validateGroundingWorkingState({ lastTrial: { ...trial(), ...patch } }));
  }
  const missing = trial();
  delete missing.outcome;
  assert.throws(() => validateGroundingWorkingState({ lastTrial: missing }));
});

test("counts, action switches and failed trials do not decide whether further evidence is useful", () => {
  const assessment = assessGroundingConstraints(undefined, "the synthetic object");
  const state = { lastTrial: trial("failed") };
  let progress;
  for (let i = 0; i < 50; i++) progress = advanceGroundingDecision(progress, state, i % 2 ? "same-source" : "other-source");
  const checkpoint = groundingDecisionCheckpoint(state, assessment, progress);
  assert.equal(checkpoint.decisionRequired, false);
  assert.equal(checkpoint.verification, "not_verified");
  const noNext = groundingDecisionCheckpoint({ lastTrial: trial("failed", null) }, assessment, progress);
  assert.equal(noNext.decisionRequired, true);
  assert.equal(groundingDecisionCheckpoint(undefined, assessment, progress).decisionRequired, false);
});

for (const [toolName, params] of measurementCalls) {
  test(`${toolName} accepts inline trial feedback without an extra notebook call or invented lock`, async (t) => {
    const f = await fixture(t);
    assert.ok(f.tools.get(toolName).parameters.properties.state);
    assert.equal(f.tools.get(toolName).parameters.required?.includes("state") ?? false, false);
    const result = await f.call(toolName, { ...params, state: { lastTrial: trial("useful"),
      selection: { status: "locked", bbox: f.params.bbox, evidence: "The model claims a useful measurement establishes the target." } } });
    assert.equal(result.details.saved, false);
    assert.equal(result.details.inlineStateUpdate.constraintAssessment.canLock, false);
    assert.equal(result.details.inlineStateUpdate.selection.status, "reconsidering");
    const evidence = await f.call("grounding_evidence");
    assert.deepEqual(evidence.details.state.lastTrial, trial("useful"));
    assert.equal(evidence.details.state.contract, undefined);
    const preview = await f.call("grounding_save_result", { ...f.params, previewOnly: true });
    assert.equal(preview.details.status, "unresolved");
    assert.ok(preview.details.confidence <= .49);
    assert.equal(preview.details.verification, "not_verified");
    await assert.rejects(readFile(join(f.outputDir, "progress.jsonl")), { code: "ENOENT" });
  });

  test(`${toolName} rejected input never commits trial feedback`, async (t) => {
    const f = await fixture(t);
    await f.call("grounding_evidence", { state: { lastTrial: trial("inconclusive") } });
    await assert.rejects(f.call(toolName, { ...params, region: [.9,.1,.2,.9], state: { lastTrial: trial("useful") } }));
    assert.deepEqual((await f.call("grounding_evidence")).details.state.lastTrial, trial("inconclusive"));
  });

  test(`${toolName} cannot mutate a record paused for clarification`, async (t) => {
    const f = await fixture(t);
    await f.call("grounding_evidence", { state: { lastTrial: trial("failed", null) }, clarification: "Which of these two visible objects do you mean?" });
    await assert.rejects(f.call(toolName, { ...params, state: { lastTrial: trial("useful") } }), /clarification|waiting|awaiting/i);
  });
}

test("a null next observation remains revisable by a new concrete observable, without disabling evidence recovery", async (t) => {
  const f = await fixture(t);
  const params = { region: [.1,.1,.6,.6], reason: "Inspect the candidate boundary against visible background" };
  const first = await f.call("grounding_view", params);
  const project = result => f.handlers.get("context")({ messages: [
    ...conversation("load", "grounding_next_batch", f.loaded), ...conversation("inspect", "grounding_view", result),
  ] });
  await project(first);
  await f.call("grounding_evidence", { state: { lastTrial: trial("failed", null) } });
  const testA = { condition: "query", observable: "Check whether the leftmost outline closes around the body" };
  const testB = { condition: "query", observable: "Check whether the separate lower tip joins the body outline" };
  const a = await f.call("grounding_view", { ...params, inspectionIntent: "counterevidence", inspectionTest: testA });
  assert.ok(images(a).length);
  await project(a);
  const b = await f.call("grounding_view", { ...params, inspectionIntent: "counterevidence", inspectionTest: testB });
  assert.ok(images(b).length, "a different concrete observable is not capped by the earlier trial");
  await project(b);
  assert.equal(images(await f.call("grounding_view", { ...params, inspectionIntent: "counterevidence", inspectionTest: testB })).length, 0);
  await f.handlers.get("context")({ messages: [] });
  assert.ok(images(await f.call("grounding_view", { ...params, inspectionIntent: "recover_evidence" })).length);
});

test("trial receipts do not transfer to the next record", async (t) => {
  const f = await fixture(t);
  await f.call("grounding_evidence", { state: { lastTrial: trial("failed", null) } });
  const context = { ui: { custom: async factory => {
    const details = factory({}, {}, {}, () => {}).groundingReview;
    return { type: "grounding_review_response", action: "confirm", constraintsResolved: true, bbox: details.bbox,
      status: "ok", confidence: .95, targetFound: true, candidateCount: 1, candidateRank: 1,
      reason: "Human review explicitly confirms this synthetic object." };
  } }, isIdle: () => true, abort() {} };
  await f.call("grounding_save_and_next", f.params, context);
  const next = await f.call("grounding_evidence");
  assert.equal(next.details.key, "two");
  assert.equal(next.details.state.lastTrial, undefined);
});

for (const [toolName, params] of measurementCalls) {
  test(`${toolName} cannot overwrite a newer notebook update while rendering`, async (t) => {
    const f = await fixture(t);
    const pending = f.call(toolName, { ...params, state: { lastTrial: trial("useful") } });
    const rejected = assert.rejects(pending, /changed|newer evidence/i);
    await f.call("grounding_evidence", { state: { lastTrial: trial("contradictory") } });
    await rejected;
    assert.deepEqual((await f.call("grounding_evidence")).details.state.lastTrial, trial("contradictory"));
  });
}

test("pending human review freezes trial state and optional measurement calls", async (t) => {
  const f = await fixture(t);
  await f.call("grounding_evidence", { state: { lastTrial: trial("inconclusive") } });
  let present;
  const opened = new Promise(resolve => { present = resolve; });
  const context = { ui: { custom: factory => new Promise(resolve => {
    const component = factory({}, {}, {}, () => {});
    present({ details: component.groundingReview, respond: resolve });
  }) }, isIdle: () => true, abort() {} };
  const rejected = assert.rejects(f.call("grounding_save_result", f.params, context), /review rejected/i);
  const panel = await opened;
  try {
    assert.equal(panel.details.status, "unresolved");
    for (const [name, params] of measurementCalls) {
      await assert.rejects(f.call(name, { ...params, state: { lastTrial: trial("useful") } }), /human review/i);
    }
    await assert.rejects(f.call("grounding_evidence", { state: { lastTrial: trial("useful") } }), /human review/i);
    assert.deepEqual((await f.call("grounding_evidence")).details.state.lastTrial, trial("inconclusive"));
    await assert.rejects(readFile(join(f.outputDir, "progress.jsonl")), { code: "ENOENT" });
  } finally {
    panel.respond({ type: "grounding_review_response", action: "reject", reason: "The human has not established the target boundary." });
    await rejected;
  }
});
