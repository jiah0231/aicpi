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
  const entries = [];
  const messages = [];
  let active = ["read", "write", "bash"];
  let callNumber = 0;
  createGroundingSafetyExtension({ cwd: root, sessionId: `improvements-${Date.now()}` }).factory({
    on(name, handler) { handlers.set(name, handler); },
    registerTool(tool) { tools.set(tool.name, tool); active.push(tool.name); },
    getActiveTools: () => [...active],
    setActiveTools(names) { active = [...names]; },
    appendEntry(customType, data) { entries.push({ type: "custom", customType, data: structuredClone(data) }); },
    sendMessage(message, options) { messages.push({ message, options }); },
    sendUserMessage() {},
    setModel: async () => true,
  });
  return {
    handlers, tools, entries, messages,
    call: (name, params, context) => tools.get(name).execute(`${name}-${++callNumber}`, params, undefined, undefined, context),
  };
}

function immediateReview(onReview = () => {}) {
  return {
    ui: { custom: async (factory) => {
      const details = factory({}, {}, {}, () => {}).groundingReview;
      onReview(details);
      return {
        type: "grounding_review_response", action: "confirm", bbox: details.bbox,
        status: "ok", confidence: .95, targetFound: true, candidateCount: 1, candidateRank: 1,
        reason: "The synthetic candidate is confirmed by the mocked human review.",
      };
    } },
    isIdle: () => true,
    abort() {},
  };
}

async function fixture(t, { count = 3, prompt, targetCount, load = true, visibleImage, visibleFilename = "visible.png" } = {}) {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-improvements-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sourceDir = join(root, "source");
  const outputDir = join(root, "output");
  await mkdir(sourceDir);
  await writeFile(join(sourceDir, visibleFilename), visibleImage ?? await sharp({
    create: { width: 80, height: 60, channels: 3, background: { r: 67, g: 103, b: 149 } },
  }).png().toBuffer());
  await writeFile(join(sourceDir, "infrared.png"), await sharp({
    create: { width: 80, height: 60, channels: 3, background: { r: 110, g: 110, b: 110 } },
  }).png().toBuffer());
  const queryPath = join(sourceDir, "queries.json");
  await writeFile(queryPath, JSON.stringify(Object.fromEntries(["one", "two", "three"].slice(0, count).map((key) => [key, {
    query: `the synthetic object ${key}`, visible: visibleFilename, infrared: "infrared.png",
    bbox: [.91, .92, .97, .99], annotations: "SOURCE_ANNOTATION_SENTINEL",
  }]))));
  const runtime = install(root);
  const userPrompt = `${prompt ?? `批处理 ${count} 条图像定位样本`}\n${queryPath}`;
  await runtime.handlers.get("before_agent_start")({ prompt: userPrompt, systemPromptOptions: { sections: {} } });
  const loaded = load ? await runtime.call("grounding_next_batch", { queryPath, outputDir, ...(targetCount === undefined ? {} : { targetCount }) }) : undefined;
  return {
    ...runtime, root, queryPath, outputDir, loaded, userPrompt,
    params: { queryPath, outputDir, key: "one", bbox: [.25, .2, .75, .8], status: "ok", confidence: .9,
      reason: "The synthetic object is visibly bounded by the selected rectangle." },
  };
}

function compareBox(actual, expected) {
  assert.equal(actual.length, 4);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-12, `${actual} != ${expected}`));
}

const images = (result) => result.content.filter((block) => block.type === "image");

function conversationTool(id, toolName, result, args = {}) {
  return [
    { role: "assistant", content: [{ type: "toolCall", id, name: toolName, arguments: args }] },
    { role: "toolResult", toolCallId: id, toolName, content: result.content, details: result.details, isError: false },
  ];
}

test("a clean crop omits hypotheses, grid and labels without requiring bbox", async (t) => {
  const f = await fixture(t);
  const result = await f.call("grounding_view", {
    region: [.2, .2, .8, .8], zoom: 3, decorations: "none", reason: "Inspect clean pixels before choosing the target identity.",
  });
  assert.equal(result.details.currentBbox, null);
  assert.equal(typeof result.details.viewId, "string");
  assert.deepEqual(result.details.evidenceViewIds, [result.details.viewId]);
  assert.equal(images(result).length, 1);
  const { data, info } = await sharp(Buffer.from(images(result)[0].data, "base64")).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  assert.deepEqual([info.width, info.height, info.channels], [144, 108, 3]);
  for (let offset = 0; offset < data.length; offset += 3) {
    assert.deepEqual([...data.subarray(offset, offset + 3)], [67, 103, 149], `unexpected decoration at pixel ${offset / 3}`);
  }
  assert.equal(JSON.stringify(result).includes("SOURCE_ANNOTATION_SENTINEL"), false);
});

test("view_pixels uses a stable earlier view after a later crop changes last_crop", async (t) => {
  const f = await fixture(t);
  const first = await f.call("grounding_view", {
    region: [.25, .2, .75, .8], zoom: 2, decorations: "none", reason: "Measure the chosen part against this stable display.",
  });
  const second = await f.call("grounding_view", {
    region: [0, 0, .2, .2], zoom: 4, decorations: "none", reason: "Inspect a different candidate without invalidating the first view.",
  });
  assert.notEqual(first.details.viewId, second.details.viewId);
  let reviewed;
  await f.call("grounding_save_result", {
    ...f.params, viewId: first.details.viewId, coordinateSpace: "view_pixels", bbox: [20, 18, 60, 54],
  }, immediateReview((details) => { reviewed = details.bbox; }));
  compareBox(reviewed, [.375, .35, .625, .65]);
  const progress = JSON.parse((await readFile(join(f.outputDir, "progress.jsonl"), "utf8")).trim());
  compareBox(progress.bbox, reviewed);
});

test("view IDs from an approved prior record cannot be recalled or used to save a new record", async (t) => {
  const f = await fixture(t);
  const view = await f.call("grounding_view", {
    region: [.25, .2, .75, .8], zoom: 2, decorations: "none", reason: "Keep this record's view for a stale-reference regression.",
  });
  await f.call("grounding_save_and_next", f.params, immediateReview());
  await assert.rejects(f.call("grounding_view", { viewId: view.details.viewId, reason: "Attempt to recall the previous record's view." }), /Unknown or stale viewId/);
  let reviewOpened = false;
  await assert.rejects(f.call("grounding_save_result", {
    ...f.params, key: "two", viewId: view.details.viewId, coordinateSpace: "view_pixels", bbox: [20, 18, 60, 54],
  }, immediateReview(() => { reviewOpened = true; })), /Unknown or stale viewId/);
  assert.equal(reviewOpened, false);
  const progress = (await readFile(join(f.outputDir, "progress.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
  assert.deepEqual(progress.map((item) => item.key), ["one"]);
});

test("candidate comparison returns one overview canvas and maps panel coordinates from that whole canvas", async (t) => {
  const f = await fixture(t);
  const result = await f.call("grounding_compare", {
    regions: [
      { label: "Left candidate", region: [.1, .1, .3, .5] },
      { label: "Right candidate", region: [.5, .2, .9, .8] },
    ], reason: "Compare two candidate identities in full-image context.",
  });
  assert.equal(images(result).length, 1);
  assert.equal(result.details.panels.length, 2);
  assert.equal(result.details.overview.label, "Overview");
  assert.equal(result.details.evidenceViewIds.length, 3);
  const panel = result.details.panels[1];
  const metadata = await sharp(Buffer.from(images(result)[0].data, "base64")).metadata();
  assert.deepEqual([panel.width, panel.height], [metadata.width, metadata.height]);
  const [left, top, right, bottom] = panel.displayRect;
  assert.ok(left > 0 && top > 0);
  await assert.rejects(f.call("grounding_save_result", {
    ...f.params, viewId: panel.id, coordinateSpace: "view_pixels", bbox: [left - 1, top, right, bottom],
  }, immediateReview()), /crosses the image content boundary/);
  let reviewed;
  await f.call("grounding_save_result", {
    ...f.params, viewId: panel.id, coordinateSpace: "view_pixels",
    bbox: [left + (right - left) / 4, top + (bottom - top) / 4, left + 3 * (right - left) / 4, top + 3 * (bottom - top) / 4],
  }, immediateReview((details) => { reviewed = details.bbox; }));
  compareBox(reviewed, [.6, .35, .8, .65]);
});

test("explicit evidence archival changes context images only and pinning wins over archival", async (t) => {
  const f = await fixture(t);
  const first = await f.call("grounding_view", { region: [.2, .2, .5, .6], zoom: 2, decorations: "none", reason: "Keep the visible counterexample available for comparison." });
  const second = await f.call("grounding_view", { region: [.5, .2, .8, .6], zoom: 2, decorations: "none", reason: "Inspect a region that will later be superseded." });
  const correction = { role: "user", content: [{ type: "text", text: "这个候选身份不对，请保留左边的反例。" }] };
  const history = [
    { role: "user", content: [{ type: "text", text: f.userPrompt }] },
    ...conversationTool("load", "grounding_next_batch", f.loaded),
    ...conversationTool("first", "grounding_view", first),
    correction,
    ...conversationTool("second", "grounding_view", second),
  ];
  const original = structuredClone(history);
  const listed = await f.call("grounding_evidence", {});
  const originalView = listed.details.views.find((view) => view.pinned);
  assert.ok(originalView);
  const state = { target: "synthetic object", facts: ["The left candidate is a counterexample."], openQuestions: ["Which candidate matches the query?"], ruledOut: ["Right region was superseded."] };
  const evidence = await f.call("grounding_evidence", {
    pin: [first.details.viewId], archive: [originalView.id, first.details.viewId, second.details.viewId], state,
  });
  assert.deepEqual(evidence.details.state, state);
  const context = await f.handlers.get("context")({ messages: history });
  const results = context.messages.filter((message) => message.role === "toolResult");
  assert.equal(images(results.find((message) => message.toolCallId === "load")).length, images(f.loaded).length);
  assert.equal(images(results.find((message) => message.toolCallId === "first")).length, 1);
  assert.equal(images(results.find((message) => message.toolCallId === "second")).length, 0);
  assert.ok(context.messages.includes(correction));
  assert.deepEqual(context.messages.map((message) => ({ ...message, content: message.content.filter((block) => block.type !== "image") })),
    history.map((message) => ({ ...message, content: message.content.filter((block) => block.type !== "image") })));
  assert.deepEqual(history, original, "the persisted transcript must remain unchanged");
  await f.call("grounding_evidence", { unpin: [first.details.viewId], restore: [second.details.viewId] });
  const changed = (await f.handlers.get("context")({ messages: history })).messages.filter((message) => message.role === "toolResult");
  assert.equal(images(changed.find((message) => message.toolCallId === "first")).length, 0);
  assert.equal(images(changed.find((message) => message.toolCallId === "second")).length, 1);
});

test("the first infrared crop includes the full infrared image for context and later crops do not repeat it", async (t) => {
  const f = await fixture(t);
  const first = await f.call("grounding_view", {
    modality: "infrared", region: [.25, .2, .75, .8], zoom: 2, decorations: "none", reason: "Resolve a visible ambiguity with infrared evidence.",
  });
  assert.equal(images(first).length, 2);
  const dimensions = await Promise.all(images(first).map(async (block) => {
    const metadata = await sharp(Buffer.from(block.data, "base64")).metadata();
    return [metadata.width, metadata.height];
  }));
  assert.equal(dimensions[0][0] / dimensions[0][1], 80 / 60, "the first image must preserve the complete source aspect ratio");
  assert.deepEqual(dimensions[1], [80, 72]);
  const second = await f.call("grounding_view", {
    modality: "infrared", region: [.2, .2, .8, .8], zoom: 2, decorations: "none", reason: "Check the neighbouring infrared pixels after the overview.",
  });
  assert.equal(images(second).length, 1);
  await f.call("grounding_evidence", { archive: [first.details.viewId] });
  const history = [
    ...conversationTool("load", "grounding_next_batch", f.loaded),
    ...conversationTool("ir", "grounding_view", first),
  ];
  const context = (await f.handlers.get("context")({ messages: history }))?.messages ?? history;
  assert.equal(images(context.find((message) => message.toolCallId === "ir")).length, 2, "the first full-modality overview must survive crop archival");
});

test("the real Chinese three-item request keeps save_result running until the last approved record", async (t) => {
  const f = await fixture(t, { prompt: "请处理下面这组 3 条图像定位样本" });
  assert.equal(f.loaded.details.job.targetCount, 3);
  for (const [index, key] of ["one", "two", "three"].entries()) {
    if (index > 0) await f.call("grounding_next_batch", { queryPath: f.queryPath, outputDir: f.outputDir });
    const saved = await f.call("grounding_save_result", { ...f.params, key }, immediateReview());
    assert.equal(saved.terminate, index === 2);
    assert.equal(saved.details.nextAction, index === 2 ? "complete" : "grounding_next_batch");
    assert.equal(saved.details.job.approvedInJob, index + 1);
  }
});

test("explicit targetCount limits a three-record dataset without saving or loading the second record", async (t) => {
  const f = await fixture(t, { prompt: "请根据图像定位目标，逐条审核", targetCount: 1 });
  assert.equal(f.loaded.details.job.targetCount, 1);
  const saved = await f.call("grounding_save_and_next", f.params, immediateReview((details) => assert.equal(details.canContinue, false)));
  assert.equal(saved.terminate, true);
  assert.equal(saved.details.next, null);
  assert.equal(saved.details.requestedLimitReached, true);
  const status = await f.call("grounding_status", {});
  assert.equal(status.details.completed, 1);
  assert.equal(status.details.remaining, 2);
  assert.equal(status.details.job.approvedInJob, 1);
  const next = await f.call("grounding_next_batch", { queryPath: f.queryPath, outputDir: f.outputDir });
  assert.equal(next.terminate, true);
  assert.deepEqual(next.details.records, []);
  assert.equal(next.details.requestedLimitReached, true);
});

test("persisted grounding job restores the requested count and original progress baseline across wrapper restart", async (t) => {
  const f = await fixture(t, { prompt: "请根据图像定位目标，逐条审核", targetCount: 2 });
  await f.call("grounding_save_result", f.params, immediateReview());
  const jobEntry = f.entries.filter((entry) => entry.customType === "grounding:job").at(-1);
  assert.equal(jobEntry.data.targetCount, 2);
  assert.equal(jobEntry.data.startCompleted, 0);
  assert.equal(jobEntry.data.lastApprovedKey, "one");
  const restored = install(f.root);
  await restored.handlers.get("session_start")({}, { sessionManager: { getEntries: () => [
    { type: "message", message: { role: "user", content: f.userPrompt } },
    ...f.entries,
  ] } });
  const status = await restored.call("grounding_status", {});
  assert.equal(status.details.queryPath, f.queryPath);
  assert.equal(status.details.outputDir, f.outputDir);
  assert.equal(status.details.job.targetCount, 2);
  assert.equal(status.details.job.approvedInJob, 1);
  const next = await restored.call("grounding_next_batch", { queryPath: f.queryPath, outputDir: f.outputDir });
  assert.equal(next.details.records[0].key, "two");
  const saved = await restored.call("grounding_save_and_next", { ...f.params, key: "two" }, immediateReview());
  assert.equal(saved.terminate, true);
  assert.equal(saved.details.next, null);
  assert.equal(saved.details.job.approvedInJob, 2);
  assert.equal(saved.details.requestedLimitReached, true);
  const finalStatus = await restored.call("grounding_status", {});
  assert.equal(finalStatus.details.remaining, 1);
});

test("wrapper restart restores the pending record, bbox, evidence state and revision-safe source save", async (t) => {
  const f = await fixture(t, { prompt: "请根据图像定位目标，逐条审核", targetCount: 2 });
  const workingState = {
    target: "synthetic object one",
    facts: ["A bounded blue rectangle is visible in the selected area."],
    hypotheses: ["The rectangle may be the queried object."],
    openQuestions: ["Do its visible edges cover the complete target?"],
    ruledOut: ["The far right annotation is unrelated."],
  };
  await f.call("grounding_evidence", { state: workingState });
  const oldView = await f.call("grounding_view", {
    bbox: [.2, .25, .6, .75],
    region: [.1, .1, .8, .9],
    zoom: 2,
    decorations: "hypothesis",
    reason: "Confirm the complete visible boundary before requesting review.",
  });
  const pendingEntry = f.entries.filter((entry) => entry.customType === "grounding:job").at(-1);
  assert.equal(pendingEntry.data.version, 2);
  assert.equal(pendingEntry.data.pending.key, "one");
  compareBox(pendingEntry.data.pending.currentBbox, [.2, .25, .6, .75]);
  assert.deepEqual(pendingEntry.data.pending.workingState, workingState);

  const restored = install(f.root);
  await restored.handlers.get("session_start")({}, { sessionManager: { getEntries: () => [
    { type: "message", message: { role: "user", content: f.userPrompt } },
    ...f.entries,
  ] } });
  const status = await restored.call("grounding_status", {});
  assert.equal(status.details.job.currentKey, "one");
  const reloaded = await restored.call("grounding_next_batch", { queryPath: f.queryPath, outputDir: f.outputDir });
  assert.equal(reloaded.details.records[0].key, "one");
  assert.equal(reloaded.details.job.currentKey, "one");
  const evidence = await restored.call("grounding_evidence", {});
  assert.deepEqual(evidence.details.state, workingState);
  await assert.rejects(
    restored.call("grounding_view", { viewId: oldView.details.viewId, reason: "Check that a stale wrapper-local view cannot be reused." }),
    /Unknown or stale viewId/,
  );

  const restartedAgain = install(f.root);
  await restartedAgain.handlers.get("session_start")({}, { sessionManager: { getEntries: () => [
    { type: "message", message: { role: "user", content: f.userPrompt } },
    ...f.entries,
    ...restored.entries,
  ] } });
  const saved = await restartedAgain.call("grounding_save_result", f.params, immediateReview());
  assert.equal(saved.details.saved, "one");
  assert.equal(saved.details.job.currentKey, null);
  const clearedEntry = restartedAgain.entries.filter((entry) => entry.customType === "grounding:job").at(-1);
  assert.equal(clearedEntry.data.version, 2);
  assert.equal(clearedEntry.data.pending, null);
  const progress = JSON.parse((await readFile(join(f.outputDir, "progress.jsonl"), "utf8")).trim());
  compareBox(progress.bbox, f.params.bbox);
});

test("status inspection of another dataset does not change the active job or report same-named foreign keys as loaded", async (t) => {
  const f = await fixture(t, { prompt: "请根据图像定位目标，逐条审核", targetCount: 2 });
  await f.call("grounding_save_and_next", f.params, immediateReview());
  const other = await fixture(t, { load: false });
  const persistedBefore = structuredClone(f.entries);
  const foreign = await f.call("grounding_status", { queryPath: other.queryPath, outputDir: other.outputDir });
  assert.equal(foreign.details.job, null);
  assert.equal(foreign.details.completed, 0);
  assert.ok(foreign.details.records.every((record) => record.state === "unfinished"), "loaded record identities must include the dataset, not just the key");
  const otherRun = await f.call("grounding_status", { queryPath: f.queryPath, outputDir: other.outputDir });
  assert.equal(otherRun.details.job, null);
  assert.ok(otherRun.details.records.every((record) => record.state === "unfinished"), "loaded record identities must also include the output run");
  assert.deepEqual(f.entries, persistedBefore);
  const current = await f.call("grounding_status", {});
  assert.equal(current.details.queryPath, f.queryPath);
  assert.equal(current.details.outputDir, f.outputDir);
  assert.equal(current.details.job.targetCount, 2);
  assert.equal(current.details.job.approvedInJob, 1);
  assert.equal(current.details.job.currentKey, "two");
  const saved = await f.call("grounding_save_and_next", { ...f.params, key: "two" }, immediateReview());
  assert.equal(saved.terminate, true);
  assert.equal(saved.details.requestedLimitReached, true);
  assert.equal(saved.details.next, null);
});

async function pixelAt(block, x, y) {
  const { data, info } = await sharp(Buffer.from(block.data, "base64")).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const offset = (Math.floor(y * info.height) * info.width + Math.floor(x * info.width)) * info.channels;
  return [...data.subarray(offset, offset + 3)];
}

function compareColor(actual, expected) {
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) <= 5, `${actual} differs from expected ${expected}`));
}

test("EXIF-rotated JPEG overview, comparison, crop and approved coordinates share raw source axes", async (t) => {
  const leftColor = [210, 40, 20];
  const rightColor = [20, 70, 200];
  const right = await sharp({ create: { width: 1200, height: 1200, channels: 3, background: { r: rightColor[0], g: rightColor[1], b: rightColor[2] } } }).png().toBuffer();
  const jpeg = await sharp({ create: { width: 2400, height: 1200, channels: 3, background: { r: leftColor[0], g: leftColor[1], b: leftColor[2] } } })
    .composite([{ input: right, left: 1200, top: 0 }]).withMetadata({ orientation: 6 }).jpeg({ quality: 100, chromaSubsampling: "4:4:4" }).toBuffer();
  assert.equal((await sharp(jpeg).metadata()).orientation, 6);
  const f = await fixture(t, { visibleImage: jpeg, visibleFilename: "oriented.jpg" });
  const full = images(f.loaded)[0];
  const fullMetadata = await sharp(Buffer.from(full.data, "base64")).metadata();
  assert.equal(fullMetadata.width / fullMetadata.height, 2, "overview must not auto-rotate raw source axes");
  assert.deepEqual([f.loaded.details.images[0].originalWidth, f.loaded.details.images[0].originalHeight], [2400, 1200]);
  compareColor(await pixelAt(full, .25, .5), leftColor);
  compareColor(await pixelAt(full, .75, .5), rightColor);
  const compared = await f.call("grounding_compare", {
    regions: [{ label: "Right blue half", region: [.5, 0, 1, 1] }], reason: "Compare the blue candidate against the raw source axes.",
  });
  assert.deepEqual([compared.details.overview.sourceWidth, compared.details.overview.sourceHeight], [2400, 1200]);
  const [left, top, rightEdge, bottom] = compared.details.overview.displayRect;
  const canvas = images(compared)[0];
  const { width, height } = compared.details.overview;
  compareColor(await pixelAt(canvas, (left + .25 * (rightEdge - left)) / width, (top + .5 * (bottom - top)) / height), leftColor);
  compareColor(await pixelAt(canvas, (left + .75 * (rightEdge - left)) / width, (top + .5 * (bottom - top)) / height), rightColor);
  const crop = await f.call("grounding_view", {
    region: [.5, 0, 1, 1], zoom: 1, decorations: "none", reason: "Measure a blue rectangle using the right half of the raw source.",
  });
  compareColor(await pixelAt(images(crop)[0], .5, .5), rightColor);
  assert.deepEqual([crop.details.image.originalWidth, crop.details.image.originalHeight], [2400, 1200]);
  let review;
  await f.call("grounding_save_result", {
    ...f.params, viewId: crop.details.viewId, coordinateSpace: "view_pixels", bbox: [300, 300, 900, 900],
  }, immediateReview((details) => { review = details; }));
  compareBox(review.bbox, [.625, .25, .875, .75]);
  assert.deepEqual([review.image.originalWidth, review.image.originalHeight], [2400, 1200]);
  assert.equal(review.image.width / review.image.height, 2);
  compareColor(await pixelAt(review.image, .75, .5), rightColor);
  const progress = JSON.parse((await readFile(join(f.outputDir, "progress.jsonl"), "utf8")).trim());
  compareBox(progress.bbox, [.625, .25, .875, .75]);
});

test("recalling a tiny crop preserves exact extracted pixel edges despite normalized floating-point roundoff", async (t) => {
  const visibleImage = await sharp({ create: { width: 1920, height: 1080, channels: 3, background: "#436795" } }).png().toBuffer();
  const f = await fixture(t, { visibleImage });
  const first = await f.call("grounding_view", {
    region: [123.1 / 1920, 39.1 / 1080, 124.9 / 1920, 78.9 / 1080], zoom: 3, decorations: "none",
    reason: "Inspect a narrow source strip with fractional requested bounds.",
  });
  const recalled = await f.call("grounding_view", {
    viewId: first.details.viewId, reason: "Recall exactly the same source strip without adding neighbouring pixels.",
  });
  const cropMetadata = (result) => result.content.filter((block) => block.type === "text").map((block) => JSON.parse(block.text)).find((details) => Array.isArray(details.cropPixels));
  assert.deepEqual(cropMetadata(first).cropPixels, [123, 39, 125, 79]);
  assert.deepEqual(cropMetadata(recalled).cropPixels, [123, 39, 125, 79]);
  assert.deepEqual(recalled.details.image.cropNormalized, first.details.image.cropNormalized);
  assert.deepEqual(cropMetadata(recalled).displayedSizePixels, cropMetadata(first).displayedSizePixels);
});

test("persisted job is authoritative over later unrelated status tool calls during session restoration", async (t) => {
  const f = await fixture(t, { prompt: "请根据图像定位目标，逐条审核", targetCount: 2 });
  await f.call("grounding_save_result", f.params, immediateReview());
  const other = await fixture(t, { load: false });
  const restored = install(f.root);
  await restored.handlers.get("session_start")({}, { sessionManager: { getEntries: () => [
    { type: "message", message: { role: "user", content: f.userPrompt } },
    ...f.entries,
    { type: "message", message: { role: "assistant", content: [{
      type: "toolCall", id: "unrelated-status", name: "grounding_status", arguments: { queryPath: other.queryPath, outputDir: other.outputDir },
    }] } },
  ] } });
  const status = await restored.call("grounding_status", {});
  assert.equal(status.details.queryPath, f.queryPath);
  assert.equal(status.details.outputDir, f.outputDir);
  assert.equal(status.details.job.targetCount, 2);
  assert.equal(status.details.job.approvedInJob, 1);
  assert.equal(status.details.job.lastApprovedKey, "one");
});

for (const approvedBeforeChange of [0, 1]) {
  test(`reducing the active job to one remaining record preserves the loaded record after ${approvedBeforeChange} prior approvals`, async (t) => {
    const f = await fixture(t, { prompt: "请处理下面这组 3 条图像定位样本" });
    if (approvedBeforeChange === 1) {
      const first = await f.call("grounding_save_and_next", f.params, immediateReview());
      assert.equal(first.terminate, false);
      assert.equal(first.details.next.records[0].key, "two");
    }
    const key = approvedBeforeChange === 0 ? "one" : "two";
    await f.handlers.get("before_agent_start")({
      prompt: approvedBeforeChange === 0 ? "接下来只处理 1 条" : "Only process 1 record next", systemPromptOptions: { sections: {} },
    });
    const status = await f.call("grounding_status", {});
    assert.equal(status.details.job.targetCount, 1);
    assert.equal(status.details.job.approvedInJob, 0, "the changed request begins at the current completed baseline");
    assert.equal(status.details.job.currentKey, key, "the loaded record must not be replaced by the changed count");
    const saved = await f.call("grounding_save_and_next", { ...f.params, key }, immediateReview((details) => {
      assert.equal(details.key, key);
      assert.equal(details.canContinue, false);
    }));
    assert.equal(saved.terminate, true);
    assert.equal(saved.details.next, null);
    assert.equal(saved.details.requestedLimitReached, true);
    assert.equal(saved.details.job.approvedInJob, 1);
    assert.equal(saved.details.job.currentKey, null);
    const progress = (await readFile(join(f.outputDir, "progress.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
    assert.deepEqual(progress.map((record) => record.key), approvedBeforeChange === 0 ? ["one"] : ["one", "two"]);
    const next = await f.call("grounding_next_batch", { queryPath: f.queryPath, outputDir: f.outputDir });
    assert.equal(next.terminate, true);
    assert.deepEqual(next.details.records, []);
    const finalStatus = await f.call("grounding_status", {});
    assert.equal(finalStatus.details.remaining, 2 - approvedBeforeChange);
  });
}
