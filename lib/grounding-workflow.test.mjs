import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import sharp from "sharp";
import { createJiti } from "jiti";

const { createGroundingSafetyExtension } = await createJiti(import.meta.url).import("./grounding-safety-extension.ts");

function install(root) {
  const handlers = new Map();
  const tools = new Map();
  const messages = [];
  const activeChanges = [];
  let active = ["read", "write", "edit", "bash", "powershell", "ls", "find", "grep"];
  createGroundingSafetyExtension({ cwd: root, sessionId: `workflow-${Date.now()}` }).factory({
    on(name, handler) { handlers.set(name, handler); },
    registerTool(tool) { tools.set(tool.name, tool); active.push(tool.name); },
    getActiveTools: () => [...active],
    setActiveTools(names) { active = [...names]; activeChanges.push([...names]); },
    sendMessage(message, options) { messages.push({ message, options }); },
    sendUserMessage() {},
    setModel: async () => true,
  });
  return {
    tools, handlers, messages, activeChanges,
    activeTools: () => [...active],
    call: (name, params, context) => tools.get(name).execute(`${name}-call`, params, undefined, undefined, context),
  };
}

function confirm(details, bbox = details.bbox) {
  return {
    type: "grounding_review_response", action: "confirm", bbox,
    status: "ok", confidence: .95, targetFound: true, candidateCount: 1, candidateRank: 1,
    reason: "The synthetic target and its complete border were manually verified.",
  };
}

function immediateReview(responseFactory = confirm) {
  return {
    ui: { custom: async (factory) => responseFactory(factory({}, {}, {}, () => {}).groundingReview) },
    isIdle: () => true,
    abort() {},
  };
}

function deferredReview() {
  let present;
  const opened = new Promise((resolve) => { present = resolve; });
  const context = {
    ui: {
      custom: (factory) => new Promise((resolve) => {
        let component;
        const done = (value) => { component?.dispose(); resolve(value); };
        component = factory({}, {}, {}, done);
        present({ details: component.groundingReview, respond: (value) => component.handleInput(JSON.stringify(value)) });
      }),
    },
    isIdle: () => true,
    abort() {},
  };
  return { context, opened };
}

async function fixture(t, count = 3, load = true) {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-workflow-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sourceDir = join(root, "source");
  const outputDir = join(root, "output");
  await mkdir(sourceDir);
  const raw = Buffer.alloc(80 * 60 * 3, 255);
  for (let y = 20; y < 28; y += 1) for (let x = 30; x < 38; x += 1) raw.fill(0, (y * 80 + x) * 3, (y * 80 + x) * 3 + 3);
  const sourceImage = await sharp(raw, { raw: { width: 80, height: 60, channels: 3 } }).png().toBuffer();
  await writeFile(join(sourceDir, "synthetic.png"), sourceImage);
  const sourceQueries = Object.fromEntries(["one", "two", "three"].slice(0, count).map((key) => [key, {
    query: `the black synthetic square ${key}`, visible: "synthetic.png",
    bbox: [.91, .92, .97, .99], annotations: "SOURCE_ANNOTATION_SENTINEL",
  }]));
  const queryPath = join(sourceDir, "queries.json");
  await writeFile(queryPath, JSON.stringify(sourceQueries));
  const runtime = install(root);
  await runtime.handlers.get("before_agent_start")({
    prompt: `批处理 ${count || 1} 条 ${queryPath}`,
    systemPromptOptions: { sections: {} },
  });
  if (load) await runtime.call("grounding_next_batch", { queryPath, outputDir });
  return {
    ...runtime, root, queryPath, outputDir, sourceImage, sourceQueries,
    params: { queryPath, outputDir, key: "one", bbox: [.3, .25, .6, .6], status: "ok", confidence: .9,
      reason: "The black synthetic square is visibly enclosed by these edges." },
  };
}

function compareBox(actual, expected) {
  assert.equal(actual.length, 4);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-12, `${actual} != ${expected}`));
}

async function outputSnapshot(f) {
  return Promise.all(["progress.jsonl", "queries.json"].map((name) => readFile(join(f.outputDir, name), "utf8")));
}

test("grounding color tool infers current record, measures clean native pixels, and never saves", async (t) => {
  const f = await fixture(t);
  const result = await f.call("grounding_color_region", { region: [.25, .2, .7, .7], color: "black" });
  assert.equal(result.details.key, "one");
  assert.equal(result.details.coordinateSpace, "source");
  assert.equal(result.details.saved, false);
  assert.equal(result.details.selectedAreaPixels, 64);
  compareBox(result.details.bbox, [30 / 80, 20 / 60, 38 / 80, 28 / 60]);
  assert.equal(result.content.filter((block) => block.type === "image").length, 2);
  assert.equal(result.content.some((block) => block.type === "text" && /rawPreview|maskPreview/.test(block.text)), false);
  const rawPreview = result.content.find((block) => block.type === "image");
  const metadata = await sharp(Buffer.from(rawPreview.data, "base64")).metadata();
  assert.equal(metadata.width, result.details.previewWidth);
  assert.equal(metadata.height, result.details.previewHeight);
  await assert.rejects(readFile(join(f.outputDir, "progress.jsonl")), { code: "ENOENT" });
  assert.deepEqual(JSON.parse(await readFile(f.queryPath, "utf8")), f.sourceQueries);
  await assert.rejects(f.call("grounding_color_region", {
    queryPath: join(f.root, "another", "queries.json"), region: [0, 0, 1, 1], color: "black",
  }), /currently loaded record/);
});

test("grounding color maps ROI and point from a stable crop view", async (t) => {
  const f = await fixture(t);
  const cropRegion = [.25, .2, .7, .7];
  const crop = await f.call("grounding_view", {
    region: cropRegion, zoom: 2, decorations: "none", reason: "Inspect the synthetic square before sampling its center color.",
  });
  const result = await f.call("grounding_color_region", {
    region: [0, 0, 1, 1],
    point: [(34 / 80 - cropRegion[0]) / (cropRegion[2] - cropRegion[0]), (24 / 60 - cropRegion[1]) / (cropRegion[3] - cropRegion[1])],
    color: "black",
    selection: "point",
    coordinateSpace: "view_normalized",
    viewId: crop.details.viewId,
  });
  assert.equal(result.details.inputCoordinateSpace, "view_normalized");
  assert.equal(result.details.inputViewId, crop.details.viewId);
  compareBox(result.details.region, crop.details.image.cropNormalized);
  compareBox(result.details.bbox, [30 / 80, 20 / 60, 38 / 80, 28 / 60]);
  assert.deepEqual(result.details.pointSample.sourcePixel, [34, 24]);
  assert.equal(result.details.pointSample.matchesRequestedColor, true);
  await assert.rejects(f.call("grounding_color_region", {
    region: [0, 0, 1, 1], color: "black", coordinateSpace: "view_pixels",
  }), /viewId is required/);
});

test("color inspection does not establish or replace grounding_view last_crop coordinates", async (t) => {
  const f = await fixture(t);
  await f.call("grounding_color_region", { region: [.25, .2, .7, .7], color: "black" });
  await assert.rejects(f.call("grounding_save_result", {
    ...f.params, coordinateSpace: "last_crop",
  }, immediateReview()), /requires a successful grounding_view crop/);
  await f.call("grounding_view", {
    queryPath: f.queryPath, key: "one", modality: "visible", bbox: f.params.bbox,
    region: [.2, .2, .8, .8], zoom: 3, reason: "Inspect the whole synthetic square before refining its black pixels.",
  });
  await f.call("grounding_color_region", {
    key: "one", queryPath: f.queryPath, region: [.35, .3, .55, .55], color: "black",
  });
  let reviewBox;
  await f.call("grounding_save_result", {
    ...f.params, bbox: [.2, .2, .5, .5], coordinateSpace: "last_crop",
  }, immediateReview((details) => { reviewBox = details.bbox; return confirm(details); }));
  compareBox(reviewBox, [.32, .32, .5, .5]);
});

test("status exposes this run's approved prediction and sanitized query, never source annotations", async (t) => {
  const f = await fixture(t);
  await f.call("grounding_save_result", f.params, immediateReview());
  const result = await f.call("grounding_status", {});
  assert.equal(result.details.completed, 1);
  assert.equal(result.details.remaining, 2);
  assert.equal(result.details.queryPath, f.queryPath);
  assert.equal(result.details.outputDir, f.outputDir);
  const first = result.details.records[0];
  assert.equal(first.key, "one");
  assert.equal(first.query, f.sourceQueries.one.query);
  assert.equal(first.state, "approved");
  compareBox(first.approvedPrediction.bbox, f.params.bbox);
  assert.equal(Object.hasOwn(first, "bbox"), false);
  assert.equal(JSON.stringify(result).includes("SOURCE_ANNOTATION_SENTINEL"), false);
  assert.equal(JSON.stringify(result).includes("0.91,0.92,0.97,0.99"), false);
  const page = await f.call("grounding_status", { offset: 1, limit: 1 });
  assert.deepEqual(page.details.records.map((item) => item.key), ["two"]);
  assert.equal(page.details.nextOffset, 2);
});

test("reopening and rejection preserve approved files; confirmed revision changes only its key and never advances", async (t) => {
  const f = await fixture(t);
  await f.call("grounding_save_result", f.params, immediateReview());
  await f.call("grounding_next_batch", { queryPath: f.queryPath, outputDir: f.outputDir });
  await f.call("grounding_save_result", { ...f.params, key: "two", bbox: [.2, .2, .7, .7] }, immediateReview());
  const before = await outputSnapshot(f);
  const reopened = await f.call("grounding_reopen_record", { key: "one" });
  assert.equal(reopened.details.revision, true);
  compareBox(reopened.details.previousPrediction.bbox, f.params.bbox);
  assert.equal(JSON.stringify(reopened.details).includes("SOURCE_ANNOTATION_SENTINEL"), false);
  assert.deepEqual(await outputSnapshot(f), before);
  const pending = deferredReview();
  const rejected = assert.rejects(f.call("grounding_save_and_next", {
    ...f.params, bbox: [.35, .3, .5, .5],
  }, pending.context), /review rejected/);
  const panel = await pending.opened;
  assert.equal(panel.details.canContinue, false);
  assert.deepEqual(await outputSnapshot(f), before);
  panel.respond({ type: "grounding_review_response", action: "reject", reason: "The candidate omits part of the target." });
  await rejected;
  assert.deepEqual(await outputSnapshot(f), before);
  const refined = [30 / 80, 20 / 60, 38 / 80, 28 / 60];
  const saved = await f.call("grounding_save_and_next", {
    ...f.params, bbox: refined,
  }, immediateReview((details) => { assert.equal(details.canContinue, false); return confirm(details, refined); }));
  assert.equal(saved.terminate, true);
  assert.equal(saved.details.revision, true);
  assert.equal(saved.details.next, null);
  assert.equal(saved.details.saved.processed, 2);
  const after = await outputSnapshot(f);
  const beforeProgress = before[0].trim().split("\n").map(JSON.parse);
  const afterProgress = after[0].trim().split("\n").map(JSON.parse);
  assert.equal(afterProgress.length, 2);
  compareBox(afterProgress.find((item) => item.key === "one").bbox, refined);
  assert.deepEqual(afterProgress.find((item) => item.key === "two"), beforeProgress.find((item) => item.key === "two"));
  assert.deepEqual(JSON.parse(after[1]).two, JSON.parse(before[1]).two);
  const status = await f.call("grounding_status", {});
  assert.equal(status.details.completed, 2);
  assert.equal(status.details.remaining, 1);
  assert.equal(status.details.records.find((item) => item.key === "three").state, "unfinished");
  assert.equal(f.handlers.get("agent_before_settle")({ outcome: "completed", context: { canContinue: true } }), undefined);
  assert.deepEqual(JSON.parse(await readFile(f.queryPath, "utf8")), f.sourceQueries);
});

test("exhausted next emits a visible completion and terminates without startup retries", async (t) => {
  const f = await fixture(t, 1);
  await f.call("grounding_save_result", f.params, immediateReview());
  f.messages.length = 0;
  const result = await f.call("grounding_next_batch", { queryPath: f.queryPath, outputDir: f.outputDir });
  assert.equal(result.terminate, true);
  assert.deepEqual(result.details.records, []);
  assert.equal(f.messages.length, 1);
  assert.equal(f.messages[0].message.customType, "grounding-complete");
  assert.equal(f.messages[0].message.display, true);
  assert.match(f.messages[0].message.content, /已完成 1\/1 条，没有下一条/);
  assert.equal(f.messages[0].options.triggerTurn, false);
  assert.equal(f.handlers.get("agent_before_settle")({ outcome: "completed", context: { canContinue: true } }), undefined);
});

test("an active batch only exposes dedicated grounding tools, avoiding blocked directory and shell loops", async (t) => {
  const f = await fixture(t);
  assert.ok(f.activeChanges.length > 0);
  const active = f.activeTools();
  for (const name of ["read", "write", "edit", "bash", "powershell", "ls", "find", "grep"]) assert.equal(active.includes(name), false, name);
  for (const name of ["grounding_status", "grounding_reopen_record", "grounding_color_region", "grounding_next_batch", "grounding_view", "grounding_save_result", "grounding_save_and_next"]) assert.ok(active.includes(name), name);
});

test("session_start restores last job paths from prior grounding tool calls for default status", async (t) => {
  const f = await fixture(t, 1);
  await f.call("grounding_save_result", f.params, immediateReview());
  const restored = install(f.root);
  await restored.handlers.get("session_start")({}, {
    sessionManager: { getEntries: () => [
      { type: "message", message: { role: "user", content: `批处理 ${f.queryPath}` } },
      { type: "message", message: { role: "assistant", content: [
        { type: "toolCall", id: "historical-next", name: "grounding_next_batch", arguments: { queryPath: f.queryPath, outputDir: f.outputDir } },
      ] } },
    ] },
  });
  const result = await restored.call("grounding_status", {});
  assert.equal(result.details.queryPath, f.queryPath);
  assert.equal(result.details.outputDir, f.outputDir);
  assert.equal(result.details.completed, 1);
  assert.equal(result.details.remaining, 0);
  assert.equal(restored.activeTools().includes("ls"), false);
});
