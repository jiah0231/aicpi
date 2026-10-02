import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createJiti } from "jiti";
const jiti = createJiti(import.meta.url);
const { readGroundingLearningSettings, updateGroundingLearningSettings } = await jiti.import("./grounding-learning-settings.ts");
const { tickGroundingLearningWorker, getGroundingLearningJobStatus, groundingLearningWorkerStoreOptions, requestGroundingLearningRetry } = await jiti.import("./grounding-learning-worker.ts");
const { addGenericLearningSource, readLearningStore } = await jiti.import("./grounding-learning-store.ts");
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "generic-worker-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const path = join(root, "methods.jsonl");
  await addGenericLearningSource({ text: "Inspect complete visible outlines before deciding the outer boundary.", sampleIndependent: true }, groundingLearningWorkerStoreOptions(path));
  return path;
}
const enable = path => updateGroundingLearningSettings({ enabled: true, provider: "fake", modelId: "test" }, path);
const tick = (path, call, extra = {}) => tickGroundingLearningWorker({ learningPath: path, debounceMs: 0, call, ...extra });
const empty = async () => '[]';
test("default off, strict bounded settings and optimistic revision", async t => {
  const path = await fixture(t);
  assert.equal((await readGroundingLearningSettings(path)).enabled, false);
  await tick(path, async () => assert.fail("disabled worker called model"));
  for (const patch of [{ dailyAttemptLimit: 5 }, { apiKey: "secret" }, { endpoint: "https://x" }, { enabled: true }, { maxOutputTokens: 1601 }]) await assert.rejects(updateGroundingLearningSettings(patch, path));
  await enable(path);
  await assert.rejects(updateGroundingLearningSettings({ enabled: false }, path, 0));
});
test("successful batch charged before call, projects only generic sources and never repeats", async t => {
  const path = await fixture(t); await enable(path); let calls = 0;
  await tick(path, async request => {
    calls++;
    const saved = JSON.parse(await readFile(`${path}.consolidation-state.json`, "utf8"));
    assert.equal(saved.attempts, 1); assert.equal(saved.running, true);
    assert.equal(/imagePath|sampleId|groundTruth/.test(request.text), false);
    return empty();
  });
  await tick(path, async () => { calls++; return empty(); });
  assert.equal(calls, 1);
  const completed = JSON.parse(await readFile(`${path}.consolidation-state.json`, "utf8"));
  assert.equal(completed.errorCode, undefined); assert.equal(completed.processedSourceIds.length, 1);
  assert.equal((await readLearningStore(groundingLearningWorkerStoreOptions(path))).rules.length, 0);
});
test("failed call counts, hides provider error and does not retry unchanged input", async t => {
  const path = await fixture(t); await enable(path); let calls = 0; const now = Date.now();
  const call = async () => { calls++; throw new Error("PROVIDER_SECRET_BODY"); };
  await tick(path, call, { now }); await tick(path, call, { now: now + 7200000 });
  assert.equal(calls, 1);
  const status = await getGroundingLearningJobStatus(path);
  assert.equal(status.attemptsToday, 1); assert.equal(status.errorCode, "request_failed");
  assert.equal(JSON.stringify(status).includes("PROVIDER_SECRET_BODY"), false);
});
test("concurrent ticks make one call and disable aborts without persisting", async t => {
  const path = await fixture(t); await enable(path); let started;
  const ready = new Promise(resolve => { started = resolve; });
  let aborted = false;
  const run = tick(path, async (_request, options) => { started(); return new Promise((resolve, reject) => {
    options.signal.addEventListener("abort", () => { aborted = true; reject(new Error("stopped")); }, { once: true });
  }); });
  await ready;
  await tick(path, async () => assert.fail("concurrent call"));
  await updateGroundingLearningSettings({ enabled: false }, path);
  await run; assert.equal(aborted, true);
  assert.equal((await readLearningStore(groundingLearningWorkerStoreOptions(path))).proposals.length, 0);
});
test("interrupted reservation survives restart and is not assumed unpaid", async t => {
  const path = await fixture(t); await enable(path);
  await tick(path, empty);
  const file = `${path}.consolidation-state.json`;
  const state = JSON.parse(await readFile(file, "utf8")); state.running = true; state.processedSourceIds = [];
  await writeFile(file, JSON.stringify(state));
  await tick(path, async () => assert.fail("interrupted digest was retried"));
  const next = JSON.parse(await readFile(file, "utf8"));
  assert.equal(next.attempts, 1); assert.equal(next.errorCode, "interrupted");
  assert.ok(Date.parse(next.nextAttemptAt) > Date.now());
});
test("daily limit blocks fresh inputs and malformed state fails closed", async t => {
  const path = await fixture(t); await updateGroundingLearningSettings({ enabled: true, provider: "fake", modelId: "test", dailyAttemptLimit: 1 }, path);
  await tick(path, empty);
  await addGenericLearningSource({ text: "Check occlusion evidence carefully and never invent hidden object parts.", sampleIndependent: true }, groundingLearningWorkerStoreOptions(path));
  await tick(path, async () => assert.fail("daily quota exceeded"));
  assert.equal((await getGroundingLearningJobStatus(path)).status, "budget");
  await writeFile(`${path}.consolidation-state.json`, "corrupt");
  await assert.rejects(tick(path, empty), /quota/);
});
test("proposal output stays pending and cannot activate a rule", async t => {
  const path = await fixture(t); await enable(path);
  await tick(path, async request => JSON.stringify([{
    sourceIds: [JSON.parse(request.text).sources[0].id], operation: "normalize",
    procedure: { category: "boundary", applicability: "Partly occluded object boundaries", error: "High contrast regions may hide the outer silhouette", method: "Inspect the full visible contour before deciding each outer edge", check: "Verify visible protrusions without inventing hidden regions" },
    note: "Generic procedure for human review",
  }]));
  const store = await readLearningStore(groundingLearningWorkerStoreOptions(path));
  assert.equal(store.proposals.length, 1); assert.equal(store.proposals[0].status, "pending"); assert.equal(store.rules.length, 0);
});
test("debounce waits for stable generic input and lock excludes another process", async t => {
  const path = await fixture(t); await enable(path); let calls = 0;
  const call = async () => { calls++; return "[]"; }; const now = Date.now();
  await tick(path, call, { now, debounceMs: 60000 }); assert.equal(calls, 0);
  await tick(path, call, { now: now + 59999, debounceMs: 60000 }); assert.equal(calls, 0);
  const { default: lockfile } = await import("proper-lockfile");
  const release = await lockfile.lock(`${path}.consolidation-state.json`, { realpath: false });
  try { await tick(path, call, { now: now + 60000, debounceMs: 60000 }); assert.equal(calls, 0); }
  finally { await release(); }
  await tick(path, call, { now: now + 60001, debounceMs: 60000 }); assert.equal(calls, 1);
});
test("worker timeout bounds even an adapter that ignores abort", async t => {
  const path = await fixture(t); await updateGroundingLearningSettings({ enabled: true, provider: "fake", modelId: "test", timeoutMs: 1000 }, path);
  const keepalive = setTimeout(() => {}, 2000);
  let settle;
  try { await tick(path, async () => new Promise(resolve => { settle = resolve; })); } finally { clearTimeout(keepalive); }
  await tick(path, async () => assert.fail("unsettled timed-out request overlapped"), { now: Date.now() + 120000 });
  const { default: lockfile } = await import("proper-lockfile");
  assert.equal(await lockfile.check(`${path}.consolidation-state.json`, { realpath: false }), true);
  settle("[]");
  await new Promise(resolve => setTimeout(resolve, 20));
  const state = JSON.parse(await readFile(`${path}.consolidation-state.json`, "utf8"));
  assert.equal(state.running, false); assert.equal(state.attempts, 1); assert.equal(state.errorCode, "request_failed");
});

test("explicit retry retains quota and backoff, excludes successes and rejects stale settings", async t => {
  const path = await fixture(t); const settings = await enable(path); const now = Date.now();
  await tick(path, async () => { throw new Error("fake failure"); }, { now });
  await assert.rejects(requestGroundingLearningRetry(settings.revision - 1, path));
  await requestGroundingLearningRetry(settings.revision, path);
  const state = JSON.parse(await readFile(`${path}.consolidation-state.json`, "utf8"));
  assert.equal(state.attempts, 1); assert.equal(state.attemptedSourceIds.length, 0); assert.ok(state.nextAttemptAt);
  await tick(path, async () => assert.fail("retry bypassed backoff"), { now });
  await tick(path, empty, { now: now + 60001 });
  const complete = JSON.parse(await readFile(`${path}.consolidation-state.json`, "utf8"));
  assert.equal(complete.attempts, 2); assert.equal(complete.processedSourceIds.length, 1);
  await requestGroundingLearningRetry(settings.revision, path);
  await tick(path, async () => assert.fail("retry reprocessed a success"), { now: now + 120000 });
});
test("corrupt or future quota day fails closed without model access", async t => {
  const path = await fixture(t); await enable(path); await tick(path, empty);
  const file = `${path}.consolidation-state.json`; const state = JSON.parse(await readFile(file, "utf8"));
  for (const day of [undefined, "corrupt", "2026-02-31", "2099-01-01"]) {
    await writeFile(file, JSON.stringify({ ...state, day }));
    await assert.rejects(tick(path, async () => assert.fail("bad day bypassed quota")));
  }
});
