import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import sharp from "sharp";
import { createJiti } from "jiti";

const { createGroundingSafetyExtension } = await createJiti(import.meta.url).import("./grounding-safety-extension.ts");
const { readGroundingLessons } = await createJiti(import.meta.url).import("./grounding-learning.ts");

function install(root, options = {}) {
  const handlers = new Map();
  const tools = new Map();
  const entries = [];
  const messages = [];
  let active = ["read", "write", "bash"];
  let callNumber = 0;
  createGroundingSafetyExtension({ cwd: root, sessionId: `improvements-${Date.now()}`, ...options }).factory({
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
        type: "grounding_review_response", action: "confirm", constraintsResolved: true, bbox: details.bbox,
        status: "ok", confidence: .95, targetFound: true, candidateCount: 1, candidateRank: 1,
        reason: "The synthetic candidate is confirmed by the mocked human review.",
      };
    } },
    isIdle: () => true,
    abort() {},
  };
}

async function fixture(t, { count = 3, prompt, targetCount, load = true, visibleImage, visibleFilename = "visible.png", query, learningPath } = {}) {
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
    query: query ?? `the synthetic object ${key}`, visible: visibleFilename, infrared: "infrared.png",
    bbox: [.91, .92, .97, .99], annotations: "SOURCE_ANNOTATION_SENTINEL",
  }]))));
  const runtime = install(root, { learningPath });
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

test("only human-reviewed generic procedures persist and reach later records without source context", async (t) => {
  const learningRoot = await mkdtemp(join(tmpdir(), "pi-grounding-learning-integration-"));
  t.after(() => rm(learningRoot, { recursive: true, force: true }));
  const learningPath = join(learningRoot, "lessons.jsonl");
  const first = await fixture(t, { count: 1, query: "the small wall switch", learningPath });
  const learning = { category: "boundary", applicability: "Small or low-contrast target boundaries",
    error: "A tight interior box can omit the outer silhouette.",
    method: "Inspect every visible outer edge before submitting the final measurement.",
    check: "Verify complete visible coverage without adding unsupported hidden parts.", sampleIndependent: true };
  await first.call("grounding_save_result", first.params, {
    ui: { custom: async (factory) => {
      const details = factory({}, {}, {}, () => {}).groundingReview;
      return {
        type: "grounding_review_response", action: "confirm", constraintsResolved: true,
        bbox: details.bbox, status: "ok", confidence: .9, targetFound: true, candidateCount: 1, candidateRank: 1,
        reason: "The browser confirms the complete visible synthetic target.",
        learning,
      };
    } },
    isIdle: () => true,
    abort() {},
  });
  assert.equal((await readGroundingLessons(learningPath)).length, 1);
  const stored = await readFile(learningPath, "utf8");
  assert.doesNotMatch(stored, /small wall switch|query|bbox|outcome|modalities|visible\.png/);
  assert.deepEqual(JSON.parse(stored), { version: 2, ...learning });

  const later = await fixture(t, { count: 1, query: "the rightmost visible drone", learningPath });
  const lessonBlock = later.loaded.content.find((block) => block.type === "text" && block.text.includes("groundingReviewLessons"));
  assert.ok(lessonBlock, "a later wrapper should receive persisted human guidance");
  const guidance = JSON.parse(lessonBlock.text).groundingReviewLessons;
  assert.equal(guidance.items[0].method, learning.method);
  assert.doesNotMatch(JSON.stringify(guidance), /small wall switch|queryPath|bbox|outcome|modalities/);
  assert.match(guidance.advisory, /never override/i);
});

function supportedContract(query, bbox) {
  return {
    originalQuery: query,
    queryCoverage: {
      status: "supported",
      evidence: "The exact original synthetic target query is represented by this contract.",
    },
    candidates: [{
      id: "selected",
      bbox,
      identity: {
        label: "synthetic object",
        status: "supported",
        basis: "visual_structure",
        evidence: "The selected synthetic object's visible structure establishes its identity.",
      },
    }],
    selectedCandidateId: "selected",
    interpretations: [{
      id: "literal",
      reading: query,
      status: "supported",
      evidence: "The literal reading matches the exact original query.",
      requirements: [{
        id: "target-identity",
        queryText: query,
        description: "Locate the requested synthetic object.",
        status: "supported",
        evidence: "The selected candidate's visible structure supports the requested identity.",
      }],
    }],
  };
}

test("high-entropy previews stay below the gateway payload cap without changing source coordinates", async (t) => {
  const width = 1920;
  const height = 1080;
  const noise = Buffer.alloc(width * height * 3);
  let state = 0x12345678;
  for (let index = 0; index < noise.length; index += 1) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    noise[index] = state & 0xff;
  }
  const visibleImage = await sharp(noise, { raw: { width, height, channels: 3 } }).png().toBuffer();
  assert.ok(visibleImage.toString("base64").length > 5_000_000, "fixture must reproduce an oversized PNG");
  const f = await fixture(t, { visibleImage });
  const firstCrop = await f.call("grounding_view", {
    region: [.1, .1, .6, .9], zoom: 1, decorations: "none",
    reason: "Inspect the first high-entropy region while preserving its source mapping.",
  });
  const secondCrop = await f.call("grounding_view", {
    region: [.4, .1, .9, .9], zoom: 1, decorations: "none",
    reason: "Inspect the second high-entropy region while preserving its source mapping.",
  });
  const activeImages = [images(f.loaded)[0], images(firstCrop)[0], images(secondCrop)[0]];
  for (const image of activeImages) {
    assert.ok(image.data.length <= 400_000, `preview contains ${image.data.length} base64 characters`);
  }
  assert.ok(activeImages.reduce((sum, image) => sum + image.data.length, 0) <= 1_200_000);
  const cropImage = images(firstCrop)[0];
  const cropMetadata = await sharp(Buffer.from(cropImage.data, "base64")).metadata();
  assert.deepEqual([firstCrop.details.image.width, firstCrop.details.image.height], [cropMetadata.width, cropMetadata.height]);
  let reviewed;
  await f.call("grounding_save_result", {
    ...f.params,
    viewId: firstCrop.details.viewId,
    coordinateSpace: "view_pixels",
    bbox: [cropMetadata.width * .25, cropMetadata.height * .25, cropMetadata.width * .75, cropMetadata.height * .75],
  }, immediateReview((details) => { reviewed = details.bbox; }));
  compareBox(reviewed, [.225, .3, .475, .7]);
  const saved = JSON.parse((await readFile(join(f.outputDir, "progress.jsonl"), "utf8")).trim());
  compareBox(saved.bbox, reviewed);
});

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

test("view recovers unambiguous display-pixel and source-pixel crop regions", async (t) => {
  const f = await fixture(t);
  const listed = await f.call("grounding_evidence", {});
  const overview = listed.details.views.find((view) => view.pinned);
  assert.ok(overview);
  const rect = overview.displayRect ?? [0, 0, overview.width, overview.height];
  const displayRegion = [
    rect[0] + (rect[2] - rect[0]) * .1,
    rect[1] + (rect[3] - rect[1]) * .1,
    rect[0] + (rect[2] - rect[0]) * .8,
    rect[1] + (rect[3] - rect[1]) * .8,
  ];
  assert.ok(displayRegion.every(Number.isSafeInteger), "fixture display fractions must land on exact pixels");

  const displayPixels = await f.call("grounding_view", {
    viewId: overview.id,
    region: displayRegion,
    zoom: 2,
    decorations: "none",
    reason: "Crop an unambiguous pixel rectangle measured on the overview display.",
  });
  assert.equal(displayPixels.details.inputCoordinateSpace, "view_pixels");
  compareBox(displayPixels.details.image.cropNormalized, [.1, .1, .8, .8]);

  const sourcePixels = await f.call("grounding_view", {
    region: [8, 6, 64, 48],
    zoom: 2,
    decorations: "none",
    reason: "Crop an unambiguous integer rectangle measured on the source image.",
  });
  assert.equal(sourcePixels.details.inputCoordinateSpace, "source_pixels");
  compareBox(sourcePixels.details.image.cropNormalized, [.1, .1, .8, .8]);

  await assert.rejects(f.call("grounding_view", {
    viewId: overview.id,
    region: [.1, .1, .8, .8],
    zoom: 2,
    reason: "Reject an ambiguous normalized region combined with a view reference.",
  }), /ambiguous with viewId/);
});

test("a selected candidate blocks unrelated candidate exploration until visible counterevidence reopens it", async (t) => {
  const query = "the second synthetic object from the left";
  const f = await fixture(t, { query });
  const selected = {
    target: query,
    contract: {
      originalQuery: query,
      queryCoverage: { status: "supported", evidence: "The original identity and left-to-right ordinal are both declared." },
      candidates: [["left", [.1, .1, .25, .3]], ["selected", [.4, .1, .55, .3]]].map(([id, bbox]) => ({ id, bbox,
        identity: { label: "synthetic object", status: "supported", basis: "visual_structure", evidence: "The synthetic candidate's structure is directly visible in the overview." } })),
      selectedCandidateId: "selected",
      interpretations: [{ id: "reading", reading: query, status: "supported", evidence: "This is the unambiguous original target description.",
        requirements: [{ id: "identity-and-order", queryText: query, description: "Second matching object from left to right", status: "supported", evidence: "Two matching structures are visible in the common source frame." }],
        spatialOrder: { axis: "x", direction: "ascending", ordinal: 2, candidateIds: ["selected", "left"], selectedCandidateId: "selected",
          candidateSet: { status: "supported", evidence: "Both relevant visible candidates are declared." } } }],
    },
    facts: ["Two visible candidates establish the requested left-to-right rank."],
    openQuestions: ["Where are the selected candidate's exact outer edges?"],
    selection: {
      status: "locked",
      bbox: [.4, .1, .55, .3],
      evidence: "The common overview establishes this candidate as the requested second object.",
    },
  };
  const evidence = await f.call("grounding_evidence", { state: selected });
  assert.deepEqual(evidence.details.state.selection.bbox, [.4, .1, .55, .3]);

  await assert.rejects(f.call("grounding_view", {
    region: [0, .2, .2, .5], zoom: 3,
    reason: "Inspect an unrelated left candidate after the target rank is already established.",
  }), /outside the selected target/);
  await assert.rejects(f.call("grounding_compare", {
    regions: [{ label: "Unrelated candidate", region: [0, .2, .2, .5] }],
    reason: "Restart candidate comparison without any new visible contradiction.",
  }), /target candidate is already selected/);
  await assert.rejects(f.call("grounding_save_result", {
    ...f.params,
    bbox: [.05, .2, .2, .5],
    reason: "Attempt to submit an unrelated candidate despite the locked target.",
  }, immediateReview()), /review bbox switches away from the locked target/);

  const target = await f.call("grounding_view", {
    region: [.35, .05, .6, .4], zoom: 3,
    reason: "Measure the selected candidate's complete visible outer boundary.",
  });
  assert.deepEqual(target.details.currentBbox, [.4, .1, .55, .3]);

  await f.call("grounding_evidence", { state: {
    ...selected,
    openQuestions: ["A newly visible shape contradicts the earlier candidate count."],
    selection: {
      status: "reconsidering",
      evidence: "A newly visible object-shaped region lies before the selected candidate and may change its rank.",
    },
  } });
  const reopened = await f.call("grounding_view", {
    region: [0, .2, .2, .5], zoom: 3,
    reason: "Inspect the newly visible contradictory shape before recomputing the ordinal rank.",
  });
  assert.equal(reopened.details.currentBbox, null);
});

test("distinct boundary crops and repeated source views stay advisory without evidence checkpoints", async (t) => {
  const f = await fixture(t);
  const bbox = [.25, .25, .55, .55];
  const query = "the synthetic object one";
  await f.call("grounding_evidence", { state: {
    contract: supportedContract(query, bbox),
    selection: { status: "locked", bbox, evidence: "The common overview establishes the requested synthetic target's identity." },
  } });
  await f.call("grounding_view", {
    region: [.1, .1, .7, .8], zoom: 2,
    reason: "Inspect the selected target's whole visible boundary.",
  });
  const left = await f.call("grounding_view", {
    region: [.2, .2, .5, .6], zoom: 4,
    reason: "Resolve where the left boundary separates from the adjacent background.",
  });
  assert.equal(left.details.sourceReuse.relation, "contained_rerender");
  const right = await f.call("grounding_view", {
    region: [.35, .2, .65, .6], zoom: 4,
    reason: "Resolve the independent right boundary question on this same target.",
  });
  assert.equal(right.details.sourceReuse.relation, "contained_rerender");
  const repeated = await f.call("grounding_view", {
    region: [.35, .2, .65, .6], zoom: 6,
    reason: "Check the right boundary at a readable display scale without treating it as new identity evidence.",
  });
  assert.equal(repeated.details.sourceReuse.relation, "exact");
  for (const result of [left, right, repeated]) {
    assert.equal(images(result).length, 1);
    const checkpoint = JSON.parse(result.content[0].text).decisionCheckpoint;
    assert.match(checkpoint, /Reuse the existing view when sufficient/);
    assert.match(checkpoint, /without an extra evidence checkpoint/);
    assert.deepEqual(result.details.currentBbox, [.25, .25, .55, .55]);
  }
});

test("partial evidence updates preserve selection and measured bounds while explicit arrays replace", async (t) => {
  const f = await fixture(t);
  const query = "the synthetic object one";
  const bbox = [.4, .1, .55, .3];
  const initial = {
    target: "the synthetic object",
    contract: supportedContract(query, bbox),
    facts: ["The visible shape connects to the requested owning object."],
    hypotheses: ["The lower dark strip may be a cast shadow rather than the target."],
    openQuestions: ["Where does the lower boundary end?"],
    ruledOut: ["The unrelated object at the left of the source frame."],
    selection: { status: "locked", bbox, evidence: "The shared overview establishes this target's identity and relation to its owner." },
  };
  await f.call("grounding_evidence", { state: initial });
  const measured = [.41, .11, .54, .29];
  await f.call("grounding_view", {
    region: [.35, .05, .6, .4], bbox: measured, zoom: 3,
    reason: "Refine the selected target's rough hypothesis to its measured visible boundary.",
  });
  const patch = {
    facts: ["The left and right boundaries are visibly separate from the background."],
    openQuestions: ["Does the lower strip belong to this target or its shadow?"],
  };
  const updated = await f.call("grounding_evidence", { state: patch });
  assert.deepEqual(updated.details.state, { ...initial, ...patch });
  assert.deepEqual(f.entries.at(-1).data.pending.currentBbox, measured);
  assert.deepEqual(f.entries.at(-1).data.pending.workingState, { ...initial, ...patch });
  const cleared = await f.call("grounding_evidence", { state: { hypotheses: [], openQuestions: [] } });
  const expected = { ...initial, ...patch, hypotheses: [], openQuestions: [] };
  assert.deepEqual(cleared.details.state, expected);
  for (const state of [undefined, {}]) {
    assert.deepEqual((await f.call("grounding_evidence", { state })).details.state, expected);
    assert.deepEqual(f.entries.at(-1).data.pending.currentBbox, measured, "listing or patching notes cannot revert refined bounds");
  }
  await assert.rejects(f.call("grounding_view", {
    region: [0, .2, .2, .5], zoom: 3,
    reason: "Try an unrelated candidate after updating only the evidence notes.",
  }), /outside the selected target/);
  await assert.rejects(f.call("grounding_compare", {
    regions: [{ label: "Unrelated candidate", region: [0, .2, .2, .5] }],
    reason: "Try to replace the target after updating only the evidence notes.",
  }), /target candidate is already selected/);
  await assert.rejects(f.call("grounding_save_result", {
    ...f.params, bbox: [.05, .2, .2, .5],
    reason: "Try to save an unrelated candidate after updating only the evidence notes.",
  }, immediateReview()), /review bbox switches away from the locked target/);
  for (const state of [null, { facts: "not a list" }, { selection: null }, { selection: { status: "locked", evidence: "Missing bounding box must fail." } }]) {
    await assert.rejects(f.call("grounding_evidence", { state }), /working state/);
    assert.deepEqual((await f.call("grounding_evidence", {})).details.state, expected);
    assert.deepEqual(f.entries.at(-1).data.pending.currentBbox, measured);
  }
  const selection = { status: "reconsidering", evidence: "A newly visible connected shape contradicts the selected target's identity." };
  const reopened = await f.call("grounding_evidence", { state: { selection } });
  assert.deepEqual(reopened.details.state, { ...expected, selection });
  assert.equal(f.entries.at(-1).data.pending.currentBbox, undefined);
  const candidate = await f.call("grounding_view", {
    region: [0, .2, .2, .5], zoom: 3,
    reason: "Inspect the concrete contradictory shape after explicitly reopening selection.",
  });
  assert.equal(candidate.details.currentBbox, null);
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

test("restart then direct source save restores all durable approvals and does not nudge a completed job", async (t) => {
  const f = await fixture(t);
  await f.call("grounding_save_and_next", f.params, immediateReview());
  await f.call("grounding_save_and_next", { ...f.params, key: "two" }, immediateReview());
  const restored = install(f.root);
  await restored.handlers.get("session_start")({}, { sessionManager: { getEntries: () => f.entries } });
  // Repeating the original count after restart must not start a new baseline
  // while its existing pending record still belongs to the original job.
  const prompt = { prompt: f.userPrompt, systemPromptOptions: { sections: {} } };
  await restored.handlers.get("before_agent_start")(prompt);
  assert.match(prompt.systemPromptOptions.sections.grounding_runtime_safety, /"approvedInJob":2/);
  const saved = await restored.call("grounding_save_result", { ...f.params, key: "three" }, immediateReview());
  assert.equal(saved.details.processed, 3);
  assert.equal(saved.details.job.approvedInJob, 3);
  assert.equal(saved.details.job.requestedLimitReached, true);
  assert.equal(saved.details.job.completionReached, true);
  assert.equal(saved.details.nextAction, "complete");
  assert.equal(await restored.handlers.get("agent_before_settle")({ outcome: "completed", context: { canContinue: true } }), undefined);
  const again = install(f.root);
  await again.handlers.get("session_start")({}, { sessionManager: { getEntries: () => [...f.entries, ...restored.entries] } });
  assert.equal(await again.handlers.get("agent_before_settle")({ outcome: "completed", context: { canContinue: true } }), undefined);
  assert.equal((await again.call("grounding_status", {})).details.job.approvedInJob, 3);
});

test("a stale pending entry for an already committed result cannot silently review or save it again", async (t) => {
  const f = await fixture(t, { count: 1 });
  const pendingEntries = structuredClone(f.entries);
  await f.call("grounding_save_result", f.params, immediateReview());
  const original = await readFile(join(f.outputDir, "progress.jsonl"), "utf8");
  const restored = install(f.root);
  await restored.handlers.get("session_start")({}, { sessionManager: { getEntries: () => pendingEntries } });
  await assert.rejects(restored.call("grounding_save_result", f.params, immediateReview(() => assert.fail("no second review"))), /Load this record|already has an approved/);
  assert.equal(await readFile(join(f.outputDir, "progress.jsonl"), "utf8"), original);
  assert.equal((await restored.call("grounding_status", {})).details.job.approvedInJob, 1);
  assert.equal(await restored.handlers.get("agent_before_settle")({ outcome: "completed", context: { canContinue: true } }), undefined);
});

test("a legacy unsupported lock remains a tentative candidate and can reach unresolved human review", async (t) => {
  const f = await fixture(t, { count: 1 });
  const evidence = await f.call("grounding_evidence", { state: {
    facts: ["The identity still needs evidence reload."],
    selection: { status: "locked", bbox: f.params.bbox, evidence: "The same dark pixels were enlarged several times." },
  } });
  assert.equal(evidence.details.lockDeferred, true);
  assert.equal(evidence.details.state.selection.status, "reconsidering");
  assert.equal(evidence.details.constraintAssessment.canLock, false);
  const context = { ui: { custom: async (factory) => {
    const details = factory({}, {}, {}, () => {}).groundingReview;
    assert.equal(details.status, "unresolved");
    assert.equal(details.confidence, .49);
    assert.equal(details.targetFound, false);
    assert.equal(details.modelProposal.confidence, .9);
    await assert.rejects(readFile(join(f.outputDir, "progress.jsonl")), { code: "ENOENT" });
    return { type: "grounding_review_response", action: "confirm", bbox: details.bbox,
      status: "unresolved", confidence: .25, targetFound: false, candidateCount: 0,
      reason: "Human accepts the tentative box while identity evidence remains unresolved." };
  } } };
  await f.call("grounding_save_result", f.params, context);
  const saved = JSON.parse((await readFile(join(f.outputDir, "progress.jsonl"), "utf8")).trim());
  assert.equal(saved.status, "unresolved");
  assert.equal(saved.confidence, .25);
  assert.equal(saved.reviewSource, "human");
  assert.equal(saved.constraintAssessment.canLock, false);
});

test("view-local subregions use an explicit coordinate frame instead of mutually exclusive viewId/region errors", async (t) => {
  const f = await fixture(t);
  const view = await f.call("grounding_view", { region: [.2, .2, .8, .8], zoom: 2, decorations: "none", reason: "Inspect the original candidate region." });
  await assert.rejects(f.call("grounding_view", { viewId: view.details.viewId, region: [.25, .25, .75, .75], zoom: 3,
    reason: "Inspect a subregion without an explicit coordinate frame." }), /explicit coordinateSpace/);
  const subview = await f.call("grounding_view", { viewId: view.details.viewId, region: [.25, .25, .75, .75], coordinateSpace: "view_normalized", zoom: 3,
    reason: "Inspect a clearly stated display-local subregion." });
  compareBox(subview.details.image.cropNormalized, [.35, .35, .65, .65]);
});

test("explicit model/automatic legacy results cannot masquerade as durable human approvals", async (t) => {
  const f = await fixture(t, { count: 1 });
  await f.call("grounding_save_result", f.params, immediateReview());
  const path = join(f.outputDir, "progress.jsonl");
  const humanRecord = JSON.parse((await readFile(path, "utf8")).trim());
  for (const reviewSource of ["model", "runtime_auto"]) {
    const legacy = JSON.stringify({ ...humanRecord, reviewSource }) + "\n";
    await writeFile(path, legacy);
    const restored = install(f.root);
    await restored.handlers.get("session_start")({}, { sessionManager: { getEntries: () => f.entries } });
    await assert.rejects(restored.call("grounding_status", {}), /not human-approved/);
    assert.equal(await readFile(path, "utf8"), legacy, "migration must not relabel or rewrite old results");
    assert.equal(await restored.handlers.get("agent_before_settle")({ outcome: "completed", context: { canContinue: true } }), undefined);
  }
});

test("artifact-only recovery repairs approved output without reapproval or double counting", async (t) => {
  const f = await fixture(t, { count: 1 });
  await f.call("grounding_save_result", f.params, immediateReview());
  const originalProgress = await readFile(join(f.outputDir, "progress.jsonl"), "utf8");
  const originalSubmission = await readFile(join(f.outputDir, "queries.json"), "utf8");
  await writeFile(join(f.outputDir, "queries.json"), "{}\n");
  await rm(join(f.outputDir, "queries.zip"));
  const restored = install(f.root);
  await restored.handlers.get("session_start")({}, { sessionManager: { getEntries: () => f.entries } });
  const status = await restored.call("grounding_status", {});
  assert.equal(status.details.job.approvedInJob, 1);
  assert.equal(status.details.job.completionReached, false);
  assert.match(status.details.job.restorationWarning, /missing or stale/);
  assert.equal(await readFile(join(f.outputDir, "queries.json"), "utf8"), "{}\n", "status stays read-only");
  const recovered = await restored.call("grounding_next_batch", { queryPath: f.queryPath, outputDir: f.outputDir });
  assert.deepEqual(recovered.details.records, []);
  assert.equal(recovered.details.job.approvedInJob, 1);
  assert.equal(recovered.details.job.completionReached, true);
  assert.equal(recovered.details.job.restorationWarning, undefined);
  assert.equal(await readFile(join(f.outputDir, "queries.json"), "utf8"), originalSubmission);
  assert.ok((await readFile(join(f.outputDir, "queries.zip"))).length > 0);
  assert.equal(await readFile(join(f.outputDir, "progress.jsonl"), "utf8"), originalProgress);
});

test("matching durable human approval clears a stale pending correction after restart", async (t) => {
  const f = await fixture(t, { count: 1 });
  await f.call("grounding_save_result", f.params, immediateReview());
  await f.call("grounding_reopen_record", { key: "one" });
  await f.call("grounding_save_result", { ...f.params, bbox: [.2, .2, .6, .6] }, immediateReview());
  const entriesBeforeClearing = f.entries.slice(0, -1);
  assert.ok(entriesBeforeClearing.at(-1).data.pending.humanApprovalId);
  const restored = install(f.root);
  await restored.handlers.get("session_start")({}, { sessionManager: { getEntries: () => entriesBeforeClearing } });
  const status = await restored.call("grounding_status", {});
  assert.equal(status.details.job.currentKey, null);
  assert.equal(status.details.job.pendingCorrectionKey, null);
  assert.equal(status.details.job.completionReached, true);
  assert.equal(await restored.handlers.get("agent_before_settle")({ outcome: "completed", context: { canContinue: true } }), undefined);
});

test("a known original ordinal cannot be omitted from an otherwise supported contract", async (t) => {
  const query = "the third drone from the left";
  const f = await fixture(t, { query });
  const result = await f.call("grounding_evidence", { state: {
    contract: { originalQuery: query, queryCoverage: { status: "supported", evidence: "The model claims the whole query is covered." },
      candidates: [{ id: "candidate", bbox: f.params.bbox, identity: { label: "drone", status: "supported", basis: "visual_structure", evidence: "The candidate has visible rotor and body structure." } }],
      selectedCandidateId: "candidate",
      interpretations: [{ id: "reading", reading: query, status: "supported", evidence: "A single reading is claimed.",
        requirements: [{ id: "target", queryText: query, description: "Requested drone identity and ordinal", status: "supported", evidence: "The model claims the chosen candidate matches." }] }] },
    selection: { status: "locked", bbox: f.params.bbox, evidence: "Attempt to lock without a candidate ordering declaration." },
  } });
  assert.equal(result.details.lockDeferred, true);
  assert.ok(result.details.constraintAssessment.issues.some((issue) => issue.code === "missing_spatial_order"));
});

test("save accepts inline contract with exact query default and preserves earlier facts", async (t) => {
  const f = await fixture(t, { count: 1 });
  await f.call("grounding_evidence", { state: { facts: ["Visible structure remains available."] } });
  const contract = supportedContract("the synthetic object one", f.params.bbox);
  delete contract.originalQuery;
  let review;
  await f.call("grounding_save_result", { ...f.params, contract }, immediateReview((details) => {
    review = details;
    const pending = f.entries.filter((entry) => entry.customType === "grounding:job").at(-1).data.pending;
    assert.deepEqual(pending.workingState.facts, ["Visible structure remains available."]);
  }));
  assert.equal(review.modelContract.originalQuery, "the synthetic object one");
  assert.equal(review.constraintAssessment.canLock, true);
  assert.equal(review.status, "ok");
  const saved = JSON.parse((await readFile(join(f.outputDir, "progress.jsonl"), "utf8")).trim());
  assert.equal(saved.modelContract.originalQuery, "the synthetic object one");
});

test("inline mismatched query is rejected before review, while unresolved claims stay reviewable", async (t) => {
  const f = await fixture(t, { count: 1 });
  const contract = supportedContract("wrong record", f.params.bbox);
  await assert.rejects(f.call("grounding_save_result", { ...f.params, contract }, immediateReview(() => assert.fail("must not review"))), /exactly match/);
  const actual = supportedContract("the synthetic object one", f.params.bbox);
  await f.call("grounding_evidence", { state: {
    contract: actual,
    selection: { status: "locked", bbox: f.params.bbox, evidence: "Earlier visible evidence supported the candidate." },
  } });
  actual.candidates[0].identity.status = "unresolved";
  await f.call("grounding_save_result", { ...f.params, contract: actual }, immediateReview((review) => {
    assert.equal(review.status, "unresolved");
    assert.equal(review.confidence, .49);
    assert.equal(review.constraintAssessment.canLock, false);
    const pending = f.entries.filter((entry) => entry.customType === "grounding:job").at(-1).data.pending;
    assert.equal(pending.workingState.selection.status, "reconsidering");
  }));
});

test("crop-only calls default zoom and recover bbox plus zoom without changing measured target", async (t) => {
  const f = await fixture(t);
  await f.call("grounding_view", { bbox: f.params.bbox, coordinateSpace: "source", reason: "Retain this visible source hypothesis." });
  const first = await f.call("grounding_view", { region: [.1, .1, .9, .9], decorations: "none", reason: "Inspect the chosen source region without restating the target." });
  assert.equal(first.details.action, "crop");
  compareBox(first.details.currentBbox, f.params.bbox);
  const subview = await f.call("grounding_view", { viewId: first.details.viewId, region: [.4, .4, .6, .6], coordinateSpace: "view_normalized", decorations: "none", reason: "Fit a smaller region to the display without inheriting an earlier zoom." });
  const subviewMetadata = subview.content.filter((block) => block.type === "text").map((block) => JSON.parse(block.text)).find((metadata) => metadata.zoomSource);
  assert.equal(subviewMetadata.zoomSource, "automatic_display");
  const recovered = await f.call("grounding_view", { bbox: [.2, .1, .8, .9], zoom: 2, coordinateSpace: "source", reason: "Inspect a local crop, not a new target hypothesis." });
  compareBox(recovered.details.currentBbox, f.params.bbox);
  compareBox(recovered.details.image.cropNormalized, [.2, .1, .8, .9]);
  assert.match(recovered.content[0].text, /target bbox unchanged/);
});

test("non-visible coordinates never enter visible save or color measurement, including last crop", async (t) => {
  const f = await fixture(t, { count: 1 });
  const visible = await f.call("grounding_view", { region: [.2, .2, .8, .8], decorations: "none", reason: "Measure this visible source candidate." });
  const thermal = await f.call("grounding_view", { modality: "infrared", region: [.1, .1, .9, .9], decorations: "none", reason: "Inspect thermal evidence without assuming registration." });
  const wrongFrame = { viewId: thermal.details.viewId, coordinateSpace: "view_normalized", bbox: [.2, .2, .8, .8] };
  await assert.rejects(f.call("grounding_save_result", { ...f.params, ...wrongFrame }, immediateReview(() => assert.fail("must not review"))), /infrared.*visible|registration/i);
  await assert.rejects(f.call("grounding_save_result", { ...f.params, coordinateSpace: "last_crop" }, immediateReview(() => assert.fail("must not review"))), /last crop.*not.*visible/i);
  await assert.rejects(f.call("grounding_color_region", { region: [.2, .2, .8, .8], color: "gray", viewId: thermal.details.viewId, coordinateSpace: "view_normalized" }), /infrared.*visible|registration/i);
  await f.call("grounding_save_result", { ...f.params, viewId: visible.details.viewId, coordinateSpace: "view_normalized", bbox: [0, 0, 1, 1] }, immediateReview((review) => {
    compareBox(review.bbox, [.2, .2, .8, .8]);
  }));
});

test("IR hypotheses do not overwrite the visible target or borrow its geometry lock", async (t) => {
  const f = await fixture(t);
  await f.call("grounding_evidence", { state: {
    contract: supportedContract("the synthetic object one", f.params.bbox),
    selection: { status: "locked", bbox: f.params.bbox, evidence: "Visible structure supports this source target." },
  } });
  const thermal = await f.call("grounding_view", { modality: "infrared", region: [0, 0, .1, .1], bbox: [0, 0, .1, .1], reason: "Compare displaced thermal evidence without registration." });
  compareBox(thermal.details.currentBbox, [0, 0, .1, .1]);
  const visible = await f.call("grounding_view", { region: [.2, .2, .8, .8], reason: "Return to the original visible candidate boundary." });
  compareBox(visible.details.currentBbox, f.params.bbox);
});

test("small boxes get clean contextual pixels before approval without a model crop call", async (t) => {
  const f = await fixture(t, { count: 1 });
  const bbox = [.45, .45, .5, .5];
  await f.call("grounding_save_result", { ...f.params, bbox }, immediateReview((review) => {
    assert.ok(review.boundaryPreview);
    const { region, image } = review.boundaryPreview;
    assert.ok(region[0] < bbox[0] && region[1] < bbox[1] && region[2] > bbox[2] && region[3] > bbox[3]);
    assert.ok(image.width > 0 && image.height > 0 && image.data.length > 0);
    assert.equal(review.image.originalWidth, 80);
    compareBox(review.bbox, bbox);
  }));
});

test("optional save preview returns measured proposal pixels without approval, persistence or advancement", async (t) => {
  const f = await fixture(t, { count: 2 });
  const crop = await f.call("grounding_view", { region: [.2, .2, .8, .8], reason: "Measure the source crop for an unsaved proposal." });
  const before = structuredClone(f.entries);
  const preview = await f.call("grounding_save_and_next", { ...f.params, previewOnly: true,
    viewId: crop.details.viewId, coordinateSpace: "view_normalized", bbox: [.4, .4, .5, .5],
    contract: supportedContract("the synthetic object one", f.params.bbox),
  }, immediateReview(() => assert.fail("preview must never request human review")));
  assert.equal(preview.details.previewOnly, true);
  assert.equal(preview.details.saved, false);
  assert.equal(preview.details.key, "one");
  compareBox(preview.details.bbox, [.44, .44, .5, .5]);
  assert.equal(images(preview).length, 2);
  assert.equal(preview.details.proposalView.modality, "visible");
  assert.equal(preview.details.cleanView.modality, "visible");
  assert.deepEqual(f.entries, before, "preview does not persist job, hypothesis or contract changes");
  await assert.rejects(readFile(join(f.outputDir, "progress.jsonl")), { code: "ENOENT" });
  const state = await f.call("grounding_evidence", {});
  assert.equal(state.details.state.contract, undefined);
  assert.equal(state.details.key, "one");
  await f.call("grounding_save_result", { ...f.params, viewId: preview.details.proposalView.id,
    coordinateSpace: "view_normalized", bbox: preview.details.bbox,
  }, immediateReview((review) => compareBox(review.bbox, preview.details.bbox)));
});

test("optional proposal preview cannot bypass a locked target's region", async (t) => {
  const f = await fixture(t);
  await f.call("grounding_evidence", { state: {
    contract: supportedContract("the synthetic object one", f.params.bbox),
    selection: { status: "locked", bbox: f.params.bbox, evidence: "The current visible structure establishes this target." },
  } });
  const before = structuredClone(f.entries);
  await assert.rejects(f.call("grounding_save_and_next", { ...f.params, previewOnly: true, bbox: [0, 0, .1, .1] },
    immediateReview(() => assert.fail("preview cannot open review"))), /outside the locked target.*reconsidering/);
  assert.deepEqual(f.entries, before);
});

test("comparison reports explicit object geometry without promoting inspection ROI to object rank", async (t) => {
  const f = await fixture(t);
  const result = await f.call("grounding_compare", {
    regions: [
      { label: "Discovered first", region: [.1, .1, .3, .5], bbox: [.7, .2, .9, .4] },
      { label: "Discovered second", region: [.5, .2, .9, .8], bbox: [.1, .2, .3, .4] },
      { label: "Uncertain extent", region: [.3, .3, .6, .6] },
    ], reason: "Compare spatial positions without treating ROI centers as object centers.",
  });
  assert.deepEqual(result.details.sourceGeometry.leftToRight, ["B", "A"]);
  assert.deepEqual(result.details.sourceGeometry.missingObjectBoxes, ["C"]);
  assert.deepEqual(result.details.sourceGeometry.tiedY, [["A", "B"]]);
  assert.equal(result.details.overview.decorations, "hypothesis");
  assert.ok(result.details.panels.every((panel) => panel.decorations === "none"));
});

test("optional boundary sheet preserves proposal and exposes clean mappable strips without saving", async (t) => {
  const f = await fixture(t);
  const before = structuredClone(f.entries);
  const bbox = [0, .2, .6, 1];
  for (const name of ["grounding_save_result", "grounding_save_and_next"]) {
    await assert.rejects(f.call(name, { ...f.params, boundaryStrips: true }, immediateReview(() => assert.fail("must not open review"))), /requires previewOnly/);
  }
  const result = await f.call("grounding_save_result", { ...f.params, bbox, previewOnly: true, boundaryStrips: true },
    immediateReview(() => assert.fail("preview cannot ask for approval")));
  compareBox(result.details.bbox, bbox);
  assert.equal(images(result).length, 3);
  assert.equal(result.details.boundaryStrips.length, 4);
  assert.equal(result.details.evidenceViewIds.length, 7);
  assert.equal(result.details.boundaryStrips.find((strip) => strip.edge === "left").outsideSourcePixels, 0);
  assert.deepEqual(f.entries, before);
  const strip = result.details.boundaryStrips[0];
  const mapped = await f.call("grounding_save_result", { ...f.params, bbox: strip.displayRect,
    viewId: strip.id, coordinateSpace: "view_pixels", previewOnly: true }, immediateReview(() => assert.fail("preview cannot approve")));
  compareBox(mapped.details.bbox, strip.region);
  await assert.rejects(readFile(join(f.outputDir, "progress.jsonl")), { code: "ENOENT" });
});

test("candidate comparison cannot inspect new images while human review is pending", async (t) => {
  const f = await fixture(t);
  const context = immediateReview();
  const review = context.ui.custom;
  context.ui.custom = async (factory) => {
    await assert.rejects(f.call("grounding_compare", {
      regions: [{ label: "candidate", region: [.1, .1, .8, .8] }],
      reason: "This comparison must wait for the existing review to finish.",
    }), /awaiting human review.*before comparing/);
    return review(factory);
  };
  await f.call("grounding_save_result", f.params, context);
});
