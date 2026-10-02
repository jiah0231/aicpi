// Regression source only; not executed as part of this change.
import assert from "node:assert/strict";
import test from "node:test";
import sharp from "sharp";
import { createJiti } from "jiti";
const jiti = createJiti(import.meta.url);
const { proposeGroundingContours, validateGroundingContourOptions } = await jiti.import("./grounding-contours.ts");
const { GroundingViewRegistry } = await jiti.import("./grounding-views.ts");
const options = { region: [0, 0, 1, 1], coarseBox: [.1, .1, .9, .9] };
const solid = (width = 100, height = 100) => sharp({ create: { width, height, channels: 3, background: "#777777" } }).png().toBuffer();

test("edge options reject invalid geometry, thresholds and point filters", () => {
  assert.equal(validateGroundingContourOptions(options).lowThreshold, 20);
  for (const patch of [{ region: [0, 0, .5, .5] }, { coarseBox: [0, 0, NaN, 1] },
    { lowThreshold: 0 }, { highThreshold: Infinity }, { lowThreshold: 60, highThreshold: 20 },
    { point: [.9, .9] }, { point: [NaN, .4] }]) {
    assert.throws(() => validateGroundingContourOptions({ ...options, ...patch }));
  }
});

test("constant pixels return unresolved without invented boundaries", async () => {
  const result = await proposeGroundingContours(await solid(), options);
  assert.deepEqual(result.candidates, []);
  assert.equal(result.status, "unresolved");
  assert.equal(result.panels.length, 3);
  assert.match(result.warnings.join(" "), /not filled objects/);
});

test("edge candidate pixel geometry stays source mapped and never selects a winner", async () => {
  const bytes = await sharp(Buffer.from('<svg width="160" height="120"><rect width="160" height="120" fill="white"/><rect x="40" y="30" width="80" height="60" fill="black"/></svg>')).png().toBuffer();
  const result = await proposeGroundingContours(bytes, { region: [10 / 160, 10 / 120, 150 / 160, 110 / 120], coarseBox: [.2, .2, .8, .8], point: [.5, .5] });
  assert.deepEqual(result.regionPixels, [10, 10, 150, 110]);
  assert.ok(result.candidates.length > 0 && result.candidates.length <= 3);
  assert.equal(result.selectedCandidateId, undefined);
  for (const candidate of result.candidates) {
    assert.ok(candidate.bbox[0] <= .5 && candidate.bbox[2] > .5);
    assert.ok(candidate.rankScore >= 0 && candidate.rankScore <= 1);
    assert.equal(candidate.confidence, undefined);
  }
  for (const panel of result.panels) {
    const registry = new GroundingViewRegistry();
    const view = registry.register({ modality: "visible", region: result.region, sourceWidth: 160, sourceHeight: 120,
      width: result.width, height: result.height, displayRect: panel.rect });
    const mapped = registry.toVisibleSource(view.id, panel.rect, "view_pixels");
    mapped.forEach((v, i) => assert.ok(Math.abs(v - result.region[i]) < 1e-12));
  }
});

test("transparent pixels do not manufacture edge candidates", async () => {
  const bytes = await sharp({ create: { width: 100, height: 100, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } }).png().toBuffer();
  assert.deepEqual((await proposeGroundingContours(bytes, options)).candidates, []);
});

test("ROI budgets and asynchronous cancellation fail rather than silently resizing", async () => {
  await assert.rejects(proposeGroundingContours(await solid(1001, 1000), options), /1 million/);
  const controller = new AbortController();
  const bytes = await solid(800, 800);
  const pending = proposeGroundingContours(bytes, options, controller.signal);
  setImmediate(() => controller.abort());
  await assert.rejects(pending, /abort/i);
});

async function fixture(t) {
  const { mkdtemp, mkdir, writeFile, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const { createGroundingSafetyExtension } = await jiti.import("./grounding-safety-extension.ts");
  const root = await mkdtemp(join(tmpdir(), "grounding-contours-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, "source"); await mkdir(source);
  await writeFile(join(source, "visible.png"), await solid());
  const queryPath = join(source, "queries.json"), outputDir = join(root, "output");
  await writeFile(queryPath, JSON.stringify({ one: { query: "the synthetic object", visible: "visible.png" } }));
  const tools = new Map(), handlers = new Map(); let active = ["read", "write", "bash"];
  createGroundingSafetyExtension({ cwd: root, sessionId: "contour-regression" }).factory({
    on: (name, handler) => handlers.set(name, handler), registerTool: tool => tools.set(tool.name, tool),
    getActiveTools: () => active, setActiveTools: names => { active = names; },
    appendEntry() {}, sendMessage() {}, sendUserMessage() {}, setModel: async () => true,
  });
  const call = (name, params, context) => tools.get(name).execute(name, params, undefined, undefined, context);
  await handlers.get("before_agent_start")({ prompt: `图像定位 ${queryPath}`, systemPromptOptions: { sections: {} } });
  await call("grounding_next_batch", { queryPath, outputDir });
  return { call, handlers, queryPath, outputDir, active: () => active };
}
const toolInput = { ...options, reason: "Check the remaining boundary against the original pixels." };

test("refinement registers measurement provenance without selecting or saving; clarification blocks both entry points", async t => {
  const f = await fixture(t);
  assert.ok(f.active().includes("grounding_refine_box"));
  const result = await f.call("grounding_refine_box", toolInput);
  assert.equal(result.details.saved, false);
  assert.equal(result.details.requiresHumanReview, true);
  assert.equal(result.details.establishesObjectIdentity, false);
  assert.equal(result.details.panels[0].derived, undefined);
  for (const panel of result.details.panels.slice(1)) assert.equal(panel.derived.role, "measurement_only");
  assert.equal(result.details.panels[2].decorations, "hypothesis");
  await assert.rejects(f.call("grounding_refine_box", { ...toolInput, viewId: result.details.panels[0].id }), /source coordinates only/);
  const recalled = await f.call("grounding_view", { viewId: result.details.panels[1].id, reason: "Compare the edge proposal with original image pixels." });
  assert.ok(recalled.details.viewId);
  const evidence = await f.call("grounding_evidence", { limit: 20 });
  assert.equal(evidence.details.views.find(v => v.id === recalled.details.viewId).derived, undefined);
  await f.call("grounding_evidence", { clarification: "Which visible object do you mean?" });
  await assert.rejects(f.call("grounding_refine_box", toolInput), /clarification/i);
  const blocked = await f.handlers.get("tool_call")({ toolName: "grounding_refine_box", input: toolInput });
  assert.equal(blocked.block, true);
  assert.equal(blocked.terminate, true);
});

test("refinement checks coarseBox against a locked target, not just a broad search ROI", async t => {
  const f = await fixture(t), query = "the synthetic object", bbox = [.1, .1, .4, .4];
  const evidence = "The candidate's visible structure and context establish the requested object.";
  await f.call("grounding_evidence", { state: {
    contract: { originalQuery: query, queryCoverage: { status: "supported", evidence },
      candidates: [{ id: "target", bbox, identity: { label: "synthetic object", basis: "visual_structure", status: "supported", evidence } }], selectedCandidateId: "target",
      interpretations: [{ id: "reading", reading: query, status: "supported", evidence,
        requirements: [{ id: "identity", queryText: query, description: "The visually identified target", status: "supported", evidence }] }],
    }, selection: { status: "locked", bbox, evidence },
  } });
  await assert.rejects(f.call("grounding_refine_box", { ...toolInput, coarseBox: [.6, .6, .9, .9] }), /locked target/);
});

test("refinement waits while ordinary human review is open", async t => {
  const f = await fixture(t);
  let entered = false;
  const context = { ui: { custom: async () => {
    entered = true;
    await assert.rejects(f.call("grounding_refine_box", toolInput), /Wait for the current human review/);
    throw new Error("Synthetic review cancellation");
  } } };
  await assert.rejects(f.call("grounding_save_result", { queryPath: f.queryPath, outputDir: f.outputDir, key: "one", bbox: [.2, .2, .5, .5], confidence: .3, status: "unresolved", reason: "Synthetic incomplete identity evidence." }, context));
  assert.equal(entered, true);
});
