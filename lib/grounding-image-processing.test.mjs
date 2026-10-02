import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import sharp from "sharp";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url);
const { processGroundingImage, validateGroundingImageOperations } = await jiti.import("./grounding-image-processing.ts");
const { createGroundingSafetyExtension } = await jiti.import("./grounding-safety-extension.ts");
const { GroundingViewRegistry } = await jiti.import("./grounding-views.ts");

const solid = (width = 80, height = 60, background = "#666666") => sharp({ create: { width, height, channels: 3, background } }).png().toBuffer();
const nearBox = (actual, expected) => actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-12));

test("processing validates bounded operation-specific parameters without silent clamping", () => {
  assert.deepEqual(validateGroundingImageOperations([{ kind: "blur" }, { kind: "median" }, { kind: "threshold" }]),
    [{ kind: "blur", sigma: 1 }, { kind: "median", size: 3 }, { kind: "threshold", level: 128 }]);
  for (const operations of [[], Array(4).fill({ kind: "edges" }), [{ kind: "edges", sigma: 1 }],
    [{ kind: "median", size: 4 }], [{ kind: "blur", sigma: NaN }], [{ kind: "sharpen", sigma: 4 }],
    [{ kind: "contrast", gain: Infinity }], [{ kind: "threshold", level: 12.5 }], [{ kind: "threshold", level: 256 }],
    [{ kind: "toString" }], [null]]) assert.throws(() => validateGroundingImageOperations(operations));
});

test("all operations return original plus independent labeled derived panels with exact ROI", async () => {
  const bytes = await solid(192, 108);
  for (const operations of [[{ kind: "edges" }, { kind: "blur" }, { kind: "median" }],
    [{ kind: "sharpen" }, { kind: "contrast" }, { kind: "threshold" }]]) {
    const result = await processGroundingImage(bytes, { region: [13 / 192, 7 / 108, 89 / 192, 73 / 108], operations });
    assert.deepEqual(result.regionPixels, [13, 7, 89, 73]);
    assert.equal(result.panels.length, 4);
    assert.equal(result.panels[0].operation, undefined);
    assert.match(result.panels[0].label, /ORIGINAL/);
    assert.equal(result.panels.every((panel) => panel.rect[2] - panel.rect[0] === 76 && panel.rect[3] - panel.rect[1] === 66), true);
    assert.match(result.warnings.join(" "), /identity.*occluded/);
    const original = result.panels[0].rect;
    const originalPixels = await sharp(result.image).extract({ left: original[0], top: original[1], width: 76, height: 66 }).raw().toBuffer();
    const metadata = await sharp(result.image).metadata();
    assert.ok(metadata.channels === 3 || metadata.channels === 4);
    assert.equal(originalPixels.every((value, index) => value === (metadata.channels === 4 && index % 4 === 3 ? 255 : 102)), true);
  }
});

test("Sobel on a constant ROI is zero while threshold independently reads original intensity", async () => {
  const result = await processGroundingImage(await solid(), { region: [0, 0, 1, 1], operations: [{ kind: "edges" }, { kind: "threshold", level: 100 }] });
  for (const [index, expected] of [[1, 0], [2, 255]]) {
    const [left, top, right, bottom] = result.panels[index].rect;
    const pixels = await sharp(result.image).extract({ left, top, width: right - left, height: bottom - top }).removeAlpha().raw().toBuffer();
    assert.equal(pixels.every((value) => value === expected), true);
  }
});

test("invalid coordinates, oversized ROI and aborts fail before rendering", async () => {
  const bytes = await solid();
  for (const region of [[0, 0, 2, 1], [0, 0, 0, 1], [0, NaN, 1, 1], [0, 1, 1, 0]]) {
    await assert.rejects(processGroundingImage(bytes, { region, operations: [{ kind: "edges" }] }), /region/);
  }
  for (const region of [[0, 0, 1e-12, 1], [0, 1 - 1e-12, 1, 1]]) {
    await assert.rejects(processGroundingImage(bytes, { region, operations: [{ kind: "edges" }] }), /no pixels after edge snapping/);
  }
  await assert.rejects(processGroundingImage(await solid(2001, 2000), { region: [0, 0, 1, 1], operations: [{ kind: "edges" }] }), /4 million/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(processGroundingImage(bytes, { region: [0, 0, 1, 1], operations: [{ kind: "edges" }] }, controller.signal), /abort/i);
});

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "grounding-processing-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, "source"); await mkdir(source);
  await writeFile(join(source, "visible.png"), await solid());
  await writeFile(join(source, "infrared.png"), await solid());
  const queryPath = join(source, "queries.json"), outputDir = join(root, "output");
  await writeFile(queryPath, JSON.stringify({ one: { query: "the synthetic object", visible: "visible.png", infrared: "infrared.png" } }));
  const tools = new Map(), handlers = new Map(); let active = ["read", "write", "bash"];
  createGroundingSafetyExtension({ cwd: root, sessionId: "processing-regression" }).factory({
    on: (name, handler) => handlers.set(name, handler), registerTool: (tool) => tools.set(tool.name, tool),
    getActiveTools: () => active, setActiveTools: (names) => { active = names; },
    appendEntry() {}, sendMessage() {}, sendUserMessage() {}, setModel: async () => true,
  });
  const call = (name, params, context) => tools.get(name).execute(name, params, undefined, undefined, context);
  await handlers.get("before_agent_start")({ prompt: `图像定位 ${queryPath}`, systemPromptOptions: { sections: {} } });
  await call("grounding_next_batch", { queryPath, outputDir });
  return { call, handlers, queryPath, outputDir, active: () => active };
}

const processingParams = { region: [.2, .2, .8, .8], operations: [{ kind: "edges" }, { kind: "threshold", level: 100 }], reason: "Compare the remaining visible boundary against clean pixels." };

test("processing activates, preserves source provenance and maps each composite panel exactly", async (t) => {
  const f = await fixture(t);
  assert.ok(f.active().includes("grounding_process_image"));
  const result = await f.call("grounding_process_image", processingParams);
  assert.equal(result.content.filter((block) => block.type === "image").length, 1);
  assert.equal(result.details.saved, false);
  assert.equal(result.details.establishesObjectIdentity, false);
  assert.equal(result.details.panels.length, 3);
  assert.equal(result.details.evidenceViewIds.length, 3);
  for (const [index, panel] of result.details.panels.entries()) {
    const registry = new GroundingViewRegistry();
    const view = registry.register(panel);
    nearBox(registry.toVisibleSource(view.id, view.displayRect, "view_pixels"), result.details.region);
    if (index) {
      assert.equal(panel.derived.establishesObjectIdentity, false);
      assert.equal(panel.derived.role, "measurement_only");
      const returned = registry.get(view.id); returned.derived.operation = "corrupted";
      assert.notEqual(registry.get(view.id).derived.operation, "corrupted");
    } else assert.equal(panel.derived, undefined);
  }
});

test("same processing render still returns pixels and fresh IDs after archive; derived recall is original", async (t) => {
  const f = await fixture(t);
  const first = await f.call("grounding_process_image", processingParams);
  await f.call("grounding_evidence", { archive: first.details.evidenceViewIds });
  const second = await f.call("grounding_process_image", processingParams);
  assert.equal(second.details.renderReused, true);
  assert.equal(second.content.filter((block) => block.type === "image").length, 1);
  assert.notEqual(first.details.panels[0].id, second.details.panels[0].id);
  const recalled = await f.call("grounding_view", { viewId: second.details.panels[1].id, reason: "Check this filtered boundary against original visible pixels." });
  assert.ok(recalled.details.viewId);
  const evidence = await f.call("grounding_evidence", { limit: 20 });
  const recalledView = evidence.details.views.find((view) => view.id === recalled.details.viewId);
  assert.ok(recalledView);
  assert.equal(recalledView.derived, undefined);
  const changed = await f.call("grounding_process_image", { ...processingParams, operations: [{ kind: "threshold", level: 101 }] });
  assert.equal(changed.details.renderReused, false);
});

test("processing rejects mismatched sensor coordinates and both clarification entry points", async (t) => {
  const f = await fixture(t);
  const thermal = await f.call("grounding_view", { modality: "infrared", reason: "Check the genuine thermal ambiguity in this image." });
  await assert.rejects(f.call("grounding_process_image", { ...processingParams, viewId: thermal.details.viewId, coordinateSpace: "view_normalized" }), /infrared.*visible|registration/i);
  await assert.rejects(f.call("grounding_process_image", { ...processingParams, coordinateSpace: "view_pixels" }), /viewId/i);
  await f.call("grounding_evidence", { clarification: "Which of these objects did you mean?" });
  await assert.rejects(f.call("grounding_process_image", processingParams), /clarification/i);
  const blocked = await f.handlers.get("tool_call")({ toolName: "grounding_process_image", input: processingParams });
  assert.equal(blocked.block, true);
  assert.equal(blocked.terminate, true);
});

test("processing cannot move outside a locked target until explicit reconsideration", async (t) => {
  const f = await fixture(t);
  const query = "the synthetic object", bbox = [.2, .2, .5, .5];
  const evidence = "The candidate's structure and surrounding context establish the requested object.";
  const locked = await f.call("grounding_evidence", { state: {
    contract: {
      originalQuery: query, queryCoverage: { status: "supported", evidence },
      candidates: [{ id: "target", bbox, identity: { label: "synthetic object", basis: "visual_structure", status: "supported", evidence } }],
      selectedCandidateId: "target",
      interpretations: [{ id: "reading", reading: query, status: "supported", evidence,
        requirements: [{ id: "identity", queryText: query, description: "The visibly identified synthetic object", status: "supported", evidence }] }],
    },
    selection: { status: "locked", bbox, evidence },
  } });
  assert.equal(locked.details.state.selection.status, "locked");
  const outside = { ...processingParams, region: [.7, .7, .9, .9] };
  await assert.rejects(f.call("grounding_process_image", outside), /locked target/);
  const measured = await f.call("grounding_process_image", { ...processingParams, region: [.2, .2, .55, .55] });
  assert.equal(measured.details.saved, false);
  await f.call("grounding_evidence", { state: { selection: { status: "reconsidering", evidence: "A newly visible neighboring structure contradicts the earlier object identification." } } });
  const reconsidered = await f.call("grounding_process_image", outside);
  assert.equal(reconsidered.details.saved, false);
});

test("processing waits while human review is open and resumes after review cancellation", async (t) => {
  const f = await fixture(t);
  let enteredReview = false;
  const context = { ui: { custom: async () => {
    enteredReview = true;
    await assert.rejects(f.call("grounding_process_image", processingParams), /Wait for the current human review/);
    throw new Error("Synthetic review cancelled without approval");
  } } };
  await assert.rejects(f.call("grounding_save_result", {
    queryPath: f.queryPath, outputDir: f.outputDir, key: "one", bbox: [.2, .2, .8, .8],
    status: "unresolved", confidence: .2, reason: "Ask the human to resolve this synthetic object proposal.",
  }, context), /Synthetic review cancelled without approval/);
  assert.equal(enteredReview, true);
  const resumed = await f.call("grounding_process_image", processingParams);
  assert.equal(resumed.details.saved, false);
});
