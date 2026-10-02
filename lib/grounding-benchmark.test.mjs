import assert from "node:assert/strict";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { normalizedBboxIou, sanitizeBenchmarkRecord, summarizeBenchmarkResults } from "./grounding-benchmark.mjs";

const full = [0, 0, 1, 1];

test("sanitization preserves only query and modality paths, resolved at the original dataset", () => {
  const source = resolve("fixtures", "dataset", "queries.json");
  const absoluteDepth = resolve("other", "depth.png");
  const record = {
    query: "  the second bird's beak\n", visible: "visible/photo.png",
    infrared: "../thermal.png", depth: absoluteDepth,
    bbox: full, annotations: { expected: full }, expected: full,
    description: "REFERENCE_SENTINEL", metadata: { query: "REFERENCE_SENTINEL" },
  };
  const snapshot = structuredClone(record);
  assert.deepEqual(sanitizeBenchmarkRecord(record, source), {
    query: record.query, visible: join(dirname(source), "visible", "photo.png"),
    infrared: resolve(dirname(source), "../thermal.png"), depth: absoluteDepth,
  });
  assert.deepEqual(record, snapshot);
  assert.doesNotMatch(JSON.stringify(sanitizeBenchmarkRecord(record, source)), /bbox|annotations|REFERENCE_SENTINEL|metadata/);
});

test("sanitization ignores annotation accessors and omits absent optional images", () => {
  const record = { query: "target", visible: "photo.png", infrared: null, depth: " " };
  Object.defineProperty(record, "bbox", { enumerable: true, get() { throw new Error("reference accessed"); } });
  assert.deepEqual(sanitizeBenchmarkRecord(record, "queries.json"), { query: "target", visible: resolve("photo.png") });
});

test("sanitization rejects malformed records without stringifying potentially sensitive objects", () => {
  for (const record of [null, undefined, [], "query", {}, { query: "", visible: "a.png" },
    { query: "ok", visible: [] }, { query: {}, visible: "a.png" }, { query: "ok", visible: " " },
    { query: "ok", visible: "a\0.png" }, { query: "ok", visible: "a.png", depth: { bbox: full } },
    { query: "ok", visible: "a.png", infrared: 1 }]) {
    assert.throws(() => sanitizeBenchmarkRecord(record, "queries.json"), TypeError);
  }
  for (const source of [undefined, null, 3, {}, "", " ", "a\0.json"]) {
    assert.throws(() => sanitizeBenchmarkRecord({ query: "ok", visible: "a.png" }, source), TypeError);
  }
});

test("IoU is exact for identity, containment, disjoint and touching boxes", () => {
  assert.equal(normalizedBboxIou(full, full), 1);
  assert.equal(normalizedBboxIou([.25, .25, .75, .75], full), .25);
  assert.equal(normalizedBboxIou([0, 0, .25, .25], [.75, .75, 1, 1]), 0);
  assert.equal(normalizedBboxIou([0, 0, .5, 1], [.5, 0, 1, 1]), 0);
  assert.equal(normalizedBboxIou([0, 0, .75, 1], [.25, 0, 1, 1]), .5);
  assert.equal(normalizedBboxIou([0, 0, Number.MIN_VALUE, Number.MIN_VALUE], [0, 0, Number.MIN_VALUE, Number.MIN_VALUE]), 1);
});

test("IoU rejects malformed boxes without clamping or reordering", () => {
  const invalid = [null, undefined, "0,0,1,1", {}, { x1: 0, y1: 0, x2: 1, y2: 1 },
    new Float32Array(full), [], [0, 0, 1], [0, 0, 1, 1, 2], [0, 0, 0, 1], [0, 1, 1, 1],
    [1, 0, 0, 1], [0, 1, 1, 0], [-.001, 0, 1, 1], [0, 0, 1.001, 1], [0, 0, 640, 480],
    ["0", 0, 1, 1], [false, 0, 1, 1], [null, 0, 1, 1], [0, 0, NaN, 1],
    [0, 0, Infinity, 1], [0, -Infinity, 1, 1], new Array(4), [0, , 1, 1]];
  for (const box of invalid) {
    assert.equal(normalizedBboxIou(box, full), null);
    assert.equal(normalizedBboxIou(full, box), null);
  }
});

test("all valid references remain in the denominator, including errors and missing/invalid predictions", () => {
  const summary = summarizeBenchmarkResults([
    { status: "proposal", prediction: full, reference: full },
    { status: "timeout", reference: full },
    { status: "error", reference: full },
    { status: "settled_without_proposal", reference: full },
    { status: "proposal", prediction: [0, 0, 640, 480], reference: full },
  ]);
  assert.equal(summary.total, 5);
  assert.equal(summary.proposals, 1);
  assert.equal(summary.validReferences, 5);
  assert.equal(summary.scoredProposals, 1);
  assert.equal(summary.scoredMisses, 4);
  assert.equal(summary.missingPredictions, 3);
  assert.equal(summary.invalidPredictions, 1);
  assert.equal(summary.failures, 1);
  assert.equal(summary.timeouts, 1);
  assert.equal(summary.settledWithoutProposal, 1);
  assert.equal(summary.meanIoU, .2);
  assert.equal(summary.accuracyAt50, .2);
  assert.equal(summary.accuracyAt75, .2);
  assert.equal(summary.proposalOnlyMeanIoU, 1);
});

test("unresolved and low-confidence boxes are geometric predictions, never ground truth or automatic failures", () => {
  const summary = summarizeBenchmarkResults([
    { status: "proposal", prediction: full, reference: full, predictionStatus: "unresolved" },
    { status: "proposal", prediction: [0, 0, .5, 1], reference: full, predictionStatus: "low_confidence" },
    { status: "proposal", prediction: full, reference: null, predictionStatus: "ok" },
    { status: "proposal", prediction: full, reference: [0, 0, 0, 1] },
    { status: "timeout", prediction: full, reference: undefined },
  ]);
  assert.equal(summary.proposals, 5);
  assert.equal(summary.unresolved, 1);
  assert.equal(summary.lowConfidence, 1);
  assert.equal(summary.noGroundTruth, 2);
  assert.equal(summary.invalidReferences, 1);
  assert.equal(summary.validReferences, 2);
  assert.equal(summary.meanIoU, .75);
  assert.equal(summary.accuracyAt50, 1);
  assert.equal(summary.accuracyAt75, .5);
  assert.equal(summary.failures, 0);
  assert.equal(summary.timeouts, 1);
});

test("overlap thresholds include exact .50 and .75 boundaries", () => {
  const summary = summarizeBenchmarkResults([.49, .5, .749, .75].map((width) => ({
    prediction: [0, 0, width, 1], reference: full,
  })));
  assert.equal(summary.accuracyAt50, .75);
  assert.equal(summary.accuracyAt75, .25);
});

test("unscored runs report null accuracy rather than zero or perfect accuracy", () => {
  for (const cases of [[], [{ prediction: full }], [{ reference: [0, 0, 0, 0] }]]) {
    const summary = summarizeBenchmarkResults(cases);
    assert.equal(summary.meanIoU, null);
    assert.equal(summary.accuracyAt50, null);
    assert.equal(summary.accuracyAt75, null);
    assert.equal(summary.proposalOnlyMeanIoU, null);
  }
  const summary = summarizeBenchmarkResults([{ status: "timeout", reference: full }]);
  assert.equal(summary.meanIoU, 0);
  assert.equal(summary.accuracyAt50, 0);
  assert.equal(summary.proposalOnlyMeanIoU, null);
});

test("latency includes failed cases, excludes invalid samples, and uses nearest-rank p95", () => {
  const cases = [40, 10, 30, 20].map((elapsedMs, index) => ({
    status: index === 0 ? "timeout" : "proposal", elapsedMs,
    toolElapsedMs: elapsedMs / 10, providerElapsedMs: elapsedMs / 2,
  }));
  cases.push(...[null, "100", NaN, Infinity, -1, undefined].map((elapsedMs) => ({ elapsedMs })));
  const summary = summarizeBenchmarkResults(cases);
  assert.deepEqual(summary.latency.elapsedMs, { count: 4, mean: 25, median: 25, p95: 40 });
  assert.deepEqual(summary.latency.toolElapsedMs, { count: 4, mean: 2.5, median: 2.5, p95: 4 });
  assert.deepEqual(summary.latency.providerElapsedMs, { count: 4, mean: 12.5, median: 12.5, p95: 20 });
  assert.deepEqual(summarizeBenchmarkResults([]).latency.elapsedMs, { count: 0, mean: null, median: null, p95: null });
  assert.deepEqual(summarizeBenchmarkResults([{ elapsedMs: 0 }]).latency.elapsedMs, { count: 1, mean: 0, median: 0, p95: 0 });
  const twenty = summarizeBenchmarkResults(Array.from({ length: 20 }, (_, i) => ({ elapsedMs: i + 1 })));
  assert.equal(twenty.latency.elapsedMs.p95, 19);
  assert.equal(summarizeBenchmarkResults([1, 3, 2].map((elapsedMs) => ({ elapsedMs }))).latency.elapsedMs.median, 2);
});

test("usage totals include unsuccessful runs and support numeric or structured cost without inventing missing totals", () => {
  const summary = summarizeBenchmarkResults([
    { status: "error", usage: { input: 10, output: 2, cacheRead: 3, cacheWrite: 4, totalTokens: 19, cost: { input: 1, output: 2, cacheRead: 3, cacheWrite: 4, total: 10 } } },
    { status: "proposal", usage: { input: 5, output: 3, cacheRead: 1, cacheWrite: 0, totalTokens: 9, cost: 2 } },
    { usage: { input: NaN, output: -3, cacheRead: "20", cacheWrite: Infinity, totalTokens: null, cost: { total: -1 } } },
  ]);
  assert.deepEqual(summary.usage, { input: 15, output: 5, cacheRead: 4, cacheWrite: 4, totalTokens: 28,
    cost: { input: 1, output: 2, cacheRead: 3, cacheWrite: 4, total: 12 } });
  assert.equal(summarizeBenchmarkResults([{ usage: { input: 10, output: 2 } }]).usage.totalTokens, 0);
});

test("unknown outcomes are visible and summary does not mutate records", () => {
  const cases = [null, [], { status: "__proto__", prediction: full, reference: full }];
  const before = structuredClone(cases);
  const summary = summarizeBenchmarkResults(cases);
  assert.equal(summary.statusCounts.unknown, 2);
  assert.equal(summary.statusCounts.__proto__, 1);
  assert.equal(summary.total, summary.proposals + summary.invalidPredictions + summary.missingPredictions);
  assert.equal(summary.total, summary.validReferences + summary.noGroundTruth + summary.invalidReferences);
  assert.deepEqual(cases, before);
  assert.throws(() => summarizeBenchmarkResults({}), TypeError);
});
