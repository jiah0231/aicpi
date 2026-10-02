import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { benchmarkPrompt, consumeEvents } from "../e2e/grounding-benchmark.mjs";

const script = fileURLToPath(new URL("../e2e/grounding-benchmark.mjs", import.meta.url));
const projectRoot = dirname(dirname(script));
// Historical dataset boxes are untrusted metadata, never correctness labels.
const historicalBox = [0, 0, .5, 1];
const prediction = [0, 0, 1, 1];
const sentinel = "PRIVATE_REFERENCE_ANNOTATION_SENTINEL";

function streamOf(text, chunkSize = 1) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({
    start(controller) {
      for (let offset = 0; offset < bytes.length; offset += chunkSize) controller.enqueue(bytes.slice(offset, offset + chunkSize));
      controller.close();
    },
  });
}

test("benchmark prompt preserves review approval and original-target constraints in both variants", () => {
  const baseline = benchmarkPrompt("C:/fixtures/queries.json", "C:/bench/run/prediction-output");
  assert.match(baseline, /C:\/fixtures\/queries\.json/);
  assert.match(baseline, /C:\/bench\/run\/prediction-output/);
  assert.match(baseline, /批准后才能保存/);
  assert.match(baseline, /targetCount=1，limit=1/);
  assert.match(baseline, /原始 query/);
  assert.match(baseline, /unresolved/);
  assert.match(baseline, /\[grounding-benchmark:request-controls\]/);
  const focused = benchmarkPrompt("C:/fixtures/queries.json", "C:/bench/run/prediction-output", "focused");
  assert.ok(focused.startsWith(baseline));
  assert.match(focused, /grounding_evidence/);
  assert.match(focused, /颜色分析不是必做/);
  assert.throws(() => benchmarkPrompt("q", "out", "unknown"), /Unknown prompt variant/);
});

test("SSE consumer handles arbitrary UTF-8 boundaries, CRLF, comments and multiline data", async () => {
  const body = streamOf(': heartbeat\r\n\r\nid: ignored\r\ndata: {"type":"connected"}\r\n\r\n'
    + 'event: ignored\ndata: {\ndata: "type":"message",\ndata: "text":"目标：鸟喙"}\n\n'
    + 'data: {"type":"done"}\n\n');
  const events = [];
  await consumeEvents(body, (event) => events.push(event));
  assert.deepEqual(events, [{ type: "connected" }, { type: "message", text: "目标：鸟喙" }, { type: "done" }]);
  assert.equal(body.locked, false);
});

test("SSE consumer rejects malformed JSON and releases its stream lock", async () => {
  const body = streamOf("data: not-json\n\n");
  await assert.rejects(consumeEvents(body, () => {}), SyntaxError);
  assert.equal(body.locked, false);
});

test("SSE consumer ignores an unfinished final event and releases locks after callback failure", async () => {
  const events = [];
  await consumeEvents(streamOf('data: {"type":"unfinished"}'), (event) => events.push(event));
  assert.deepEqual(events, []);
  const body = streamOf('data: {"type":"connected"}\n\n');
  await assert.rejects(consumeEvents(body, () => { throw new Error("consumer failed"); }), /consumer failed/);
  assert.equal(body.locked, false);
});

async function mockedRun(t, scenario, stopState = { running: false }, legacyBbox = historicalBox) {
  const setupFailure = scenario.startsWith("setup_");
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-benchmark-runner-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, "dataset");
  const fixtureRoot = join(root, "fixtures");
  const out = join(root, "results");
  await mkdir(source);
  const sourcePath = join(source, "queries.json");
  const record = {
    query: "the small red target", visible: "visible.png", bbox: legacyBbox,
    annotations: sentinel, metadata: { expected: sentinel, bbox: historicalBox },
    reference: historicalBox, groundTruth: sentinel, expected: sentinel, iou: .99,
  };
  await writeFile(sourcePath, JSON.stringify({ one: record, ...(setupFailure ? { two: record } : {}) }));
  const sourceBefore = await readFile(sourcePath, "utf8");
  const requests = [];
  const sequence = [];
  const errors = [];
  const timers = new Set();
  let eventsResponse;
  let promptAt;
  let abortAt;
  let promptAcknowledged = false;
  let abortedBeforePromptAcknowledgment = false;
  const sendEvent = (event) => {
    if (eventsResponse && !eventsResponse.destroyed) eventsResponse.write(`data: ${JSON.stringify(event)}\n\n`);
  };
  const review = () => sendEvent({ type: "extension_ui_request", method: "custom", id: "review-one",
    details: { kind: "grounding_review", key: "one", bbox: prediction, status: "ok", confidence: .9,
      reason: "The synthetic target is visibly identified." } });
  const json = (response, value) => {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify(value));
  };
  const server = createServer(async (request, response) => {
    try {
      let body = "";
      for await (const chunk of request) body += chunk;
      const command = body ? JSON.parse(body) : null;
      requests.push({ method: request.method, path: request.url, command });
      if (request.url === "/api/agent/new" && request.method === "POST") {
        sequence.push("created");
        if (scenario === "setup_http_error") {
          response.writeHead(500, { "Content-Type": "application/json" });
          response.end(JSON.stringify({ error: "mock creation failed after an unknown server outcome" }));
          return;
        }
        if (scenario === "setup_timeout") return;
        if (scenario === "setup_missing_id") return json(response, { success: true, model: { provider: "mock", modelId: "vision" } });
        if (scenario === "setup_invalid_id") return json(response, { success: true, sessionId: { anotherSession: "must-not-abort" } });
        if (scenario === "setup_invalid_json") {
          response.writeHead(200, { "Content-Type": "application/json" });
          response.end("not json");
          return;
        }
        return json(response, { success: true, sessionId: "benchmark-owned", model: { provider: "mock", modelId: "vision" }, thinkingLevel: "high" });
      }
      if (request.url === "/api/agent/benchmark-owned/events" && request.method === "GET") {
        eventsResponse = response;
        response.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
        response.flushHeaders();
        const timer = setTimeout(() => { sequence.push("connected"); sendEvent({ type: "connected" }); }, 10);
        timers.add(timer);
        return;
      }
      if (request.url !== "/api/agent/benchmark-owned") throw new Error(`Unexpected API endpoint: ${request.url}`);
      if (request.method === "GET") {
        sequence.push("stop_checked");
        return json(response, stopState);
      }
      if (command?.type === "abort") {
        sequence.push("aborted");
        abortAt = performance.now();
        abortedBeforePromptAcknowledgment = !promptAcknowledged;
        // A late review racing an already-completed timeout must not become a prediction.
        if (scenario === "timeout") review();
        return json(response, { success: true, data: null });
      }
      if (command?.type !== "prompt") throw new Error(`Unsafe/unexpected command: ${command?.type}`);
      sequence.push("prompt");
      promptAt = performance.now();
      assert.ok(sequence.indexOf("connected") >= 0, "prompt must await explicit connected SSE event");
      assert.equal(sequence.indexOf("connected") < sequence.indexOf("prompt"), true);
      const fixture = JSON.parse(await readFile(join(fixtureRoot, "one", "source", "queries.json"), "utf8"));
      assert.deepEqual(fixture, { one: { query: "the small red target", visible: join(source, "visible.png") } });
      assert.doesNotMatch(JSON.stringify(fixture) + command.message, /bbox|annotations|groundTruth|reference|expected|iou|PRIVATE_REFERENCE_ANNOTATION_SENTINEL/);
      assert.equal(command.message.includes(sourcePath), false, "original annotated JSON path must not enter prompt");
      if (scenario === "delayed_ack") {
        timers.add(setTimeout(() => {
          promptAcknowledged = true;
          json(response, { success: true, data: null });
        }, 2200));
        return;
      }
      promptAcknowledged = true;
      json(response, { success: true, data: null });
      if (scenario === "timeout") return;
      if (scenario === "closed_stream") {
        eventsResponse.end();
        return;
      }
      if (scenario === "unexpected_ui") {
        sendEvent({ type: "extension_ui_request", method: "confirm", id: "unrelated", title: "Unexpected approval" });
        return;
      }
      sendEvent({ type: "entry_appended", entry: { type: "custom", customType: "grounding:request-controls", data: {
        version: 1,
        model: { provider: "mock", id: "vision", reasoning: true, thinkingFormat: "deepseek", privateEndpoint: sentinel },
        controls: { thinking: { type: "disabled", privateText: sentinel }, reasoning_effort: "off", privatePayload: sentinel },
        privateRequest: sentinel,
      } } });
      sendEvent({ type: "tool_execution_start", toolCallId: "load-one", toolName: "grounding_next_batch" });
      sendEvent({ type: "tool_execution_end", toolCallId: "load-one", toolName: "grounding_next_batch", result: { details: {} } });
      sendEvent({ type: "message_end", message: { role: "assistant", stopReason: "toolUse",
        usage: { input: 4, output: 3, cacheRead: 2, cacheWrite: 1, totalTokens: 10, cost: { total: .005 } } } });
      sendEvent({ type: "tool_execution_start", toolCallId: "review-one", toolName: "grounding_save_result" });
      review();
    } catch (error) {
      errors.push(error);
      if (!response.headersSent) response.writeHead(500, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ error: error.message }));
    }
  });
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  t.after(async () => {
    for (const timer of timers) clearTimeout(timer);
    eventsResponse?.destroy();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  const child = spawn(process.execPath, [script, "--dataset", sourcePath, "--keys", setupFailure ? "one,two" : "one", "--cwd", root,
    "--fixture-root", fixtureRoot, "--out", out, "--url", `http://127.0.0.1:${server.address().port}`,
    "--timeout-ms", "1000", "--setup-timeout-ms", "1000", "--expected-model", "mock/vision"], {
    cwd: projectRoot, windowsHide: true, stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8").on("data", (data) => { stdout += data; });
  child.stderr.setEncoding("utf8").on("data", (data) => { stderr += data; });
  t.after(() => { if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL"); });
  const guard = setTimeout(() => { errors.push(new Error("mock child exceeded 12-second test safety limit")); child.kill("SIGKILL"); }, 12_000);
  let exitCode;
  try {
    exitCode = await new Promise((resolve, reject) => { child.once("error", reject); child.once("close", resolve); });
  } finally { clearTimeout(guard); }
  assert.deepEqual(errors.map((error) => error.message), [], `${stdout}\n${stderr}`);
  assert.deepEqual(requests.filter((request) => request.method === "POST").map((request) => request.command.type),
    setupFailure ? ["ensure_session"] : ["ensure_session", "prompt", "abort"], "never approve, retry creation, or abort an unknown session");
  assert.deepEqual(sequence, setupFailure ? ["created"] : ["created", "connected", "prompt", "aborted", "stop_checked"]);
  assert.equal(await readFile(sourcePath, "utf8"), sourceBefore);
  const result = JSON.parse(await readFile(join(out, "one", "result.json"), "utf8"));
  const summary = JSON.parse(await readFile(join(out, "summary.json"), "utf8"));
  const persistedResults = await readFile(join(out, "results.jsonl"), "utf8");
  const live = JSON.parse(await readFile(join(out, "one", "live.json"), "utf8"));
  assert.equal(live.phase, "finished");
  assert.equal(live.status, result.status);
  assert.equal(persistedResults.trim().split("\n").length, 1);
  if (setupFailure) {
    assert.equal(requests.length, 1, "without a confirmed id, no session APIs or later cases may run");
    assert.equal(result.status, "error");
    assert.equal(result.errorStage, "session_creation");
    assert.equal(result.sessionCreation, "unknown");
    assert.equal(result.sessionId, null);
    assert.equal(result.stopped, false);
    assert.ok(result.setupMs > 0);
    assert.match(result.cleanupError, /no owned sessionId is known/);
    assert.equal(summary.failures, 1);
    assert.equal(summary.total, 1);
    assert.equal(summary.attempted, 1);
    assert.equal(summary.planned, 2);
    assert.equal(summary.runStatus, "terminated");
    assert.deepEqual(summary.termination, { key: "one", reason: "session_creation_unverified" });
    assert.deepEqual(summary.unattemptedKeys, ["two"]);
    assert.match(stderr, /remaining samples were not started/);
    await assert.rejects(stat(join(out, "two")), { code: "ENOENT" });
    await assert.rejects(stat(join(fixtureRoot, "two")), { code: "ENOENT" });
  }
  assert.equal(Object.hasOwn(result, "reference"), false, "historical boxes must not be collected as reference labels");
  assert.equal(Object.hasOwn(result, "iou"), false, "real benchmark must not score historical boxes");
  assert.doesNotMatch(persistedResults + stdout, /"reference"\s*:|"iou"\s*:|PRIVATE_REFERENCE_ANNOTATION_SENTINEL/);
  assert.equal(result.visualAudit.status, "pending", "a proposal or model confidence is not an independent visual audit");
  assert.match(summary.evaluation, /Independent visual audit required/);
  assert.equal(summary.noGroundTruth, 1);
  assert.equal(summary.validReferences, 0);
  assert.equal(summary.invalidReferences, 0, "legacy box fields must be ignored, not read and classified");
  assert.equal(summary.scoredProposals, 0);
  assert.equal(summary.scoredMisses, 0);
  assert.equal(summary.meanIoU, null);
  assert.equal(summary.accuracyAt50, null);
  assert.equal(summary.accuracyAt75, null);
  assert.equal(summary.proposalOnlyMeanIoU, null);
  for (const file of ["queries.json", "progress.jsonl", "queries.zip"]) {
    await assert.rejects(stat(join(out, "one", "prediction-output", file)), { code: "ENOENT" });
  }
  assert.deepEqual(result.savedArtifacts, []);
  return { exitCode, stdout, stderr, result, summary, abortAfterPromptMs: abortAt - promptAt, abortedBeforePromptAcknowledgment };
}

test("mock benchmark captures a first proposal for independent visual audit without collecting reference boxes or scores", async (t) => {
  const { exitCode, stderr, result, summary } = await mockedRun(t, "proposal");
  assert.equal(exitCode, 0, stderr);
  assert.equal(result.status, "proposal");
  assert.deepEqual(result.prediction, prediction);
  assert.equal(result.stopped, true);
  assert.deepEqual(result.toolCounts, { grounding_next_batch: 1, grounding_save_result: 1 });
  assert.equal(result.assistantResponses, 1);
  assert.equal(result.usage.totalTokens, 10);
  assert.equal(result.usage.cost.total, .005);
  assert.deepEqual(result.requestControls, [{
    version: 1,
    model: { provider: "mock", id: "vision", reasoning: true, thinkingFormat: "deepseek" },
    controls: { thinking: { type: "disabled" }, reasoning_effort: "off" },
  }]);
  assert.doesNotMatch(JSON.stringify(result.requestControls), new RegExp(sentinel));
  assert.equal(summary.proposals, 1);
});

test("mock benchmark reports timeout separately without claiming a scored miss and ignores its late review", async (t) => {
  const { exitCode, stderr, result, summary } = await mockedRun(t, "timeout");
  assert.equal(exitCode, 0, stderr);
  assert.equal(result.status, "timeout");
  assert.equal(result.prediction, null);
  assert.equal(result.stopped, true);
  assert.equal(summary.proposals, 0);
  assert.equal(summary.timeouts, 1);
});

test("mock benchmark refuses unrelated interactive approval and reports error instead", async (t) => {
  const { exitCode, stderr, result, summary } = await mockedRun(t, "unexpected_ui");
  assert.equal(exitCode, 0, stderr);
  assert.equal(result.status, "error");
  assert.match(result.error, /unexpected interactive prompt/);
  assert.equal(result.prediction, null);
  assert.equal(result.stopped, true);
  assert.equal(summary.failures, 1);
  assert.equal(summary.proposals, 0);
});

test("mock benchmark ignores malformed and misleading legacy box fields rather than treating them as ground truth", async (t) => {
  for (const [name, legacyBbox] of [
    ["null", null], ["empty", []], ["string", sentinel], ["object", { expected: sentinel }],
    ["reversed", [1, 1, 0, 0]], ["pixel coordinates", [10, 20, 640, 480]],
  ]) {
    await t.test(name, async (t) => {
      const { exitCode, stderr, result, summary } = await mockedRun(t, "proposal", { running: false }, legacyBbox);
      assert.equal(exitCode, 0, stderr);
      assert.equal(result.status, "proposal");
      assert.equal(summary.proposals, 1);
    });
  }
});

test("mock benchmark fails safety check when nonstreaming wrapper still has a pending prompt", async (t) => {
  const { exitCode, stderr, result } = await mockedRun(t, "proposal", { running: true, state: { isStreaming: false, isPromptRunning: true } });
  assert.equal(exitCode, 1);
  assert.equal(result.stopped, false);
  assert.match(stderr, /Safety check failed/);
});

test("mock benchmark does not treat a malformed stop-state response as verified shutdown", async (t) => {
  const { exitCode, stderr, result } = await mockedRun(t, "proposal", {});
  assert.equal(exitCode, 1);
  assert.equal(result.stopped, false);
  assert.match(stderr, /Safety check failed/);
});

test("mock benchmark aborts on deadline even while the HTTP prompt acknowledgment is pending", async (t) => {
  const { exitCode, stderr, result, abortAfterPromptMs, abortedBeforePromptAcknowledgment } = await mockedRun(t, "delayed_ack");
  assert.equal(exitCode, 0, stderr);
  assert.equal(result.status, "timeout");
  assert.equal(result.prediction, null);
  assert.equal(abortedBeforePromptAcknowledgment, true, `abort waited for ACK (${Math.round(abortAfterPromptMs)}ms) despite 1000ms deadline`);
});

test("mock benchmark reports a closed SSE stream as an error rather than waiting for a model timeout", async (t) => {
  const { exitCode, stderr, result } = await mockedRun(t, "closed_stream");
  assert.equal(exitCode, 0, stderr);
  assert.equal(result.status, "error");
  assert.equal(result.prediction, null);
});

test("mock benchmark records unknown creation failures and explicitly terminates without dropping the failed case", async (t) => {
  for (const scenario of ["setup_http_error", "setup_missing_id", "setup_invalid_id", "setup_invalid_json", "setup_timeout"]) {
    await t.test(scenario, async (t) => {
      const { exitCode, result, summary } = await mockedRun(t, scenario);
      assert.equal(exitCode, 1);
      assert.equal(result.prediction, null);
      assert.equal(summary.proposals, 0);
      if (scenario === "setup_timeout") {
        assert.equal(result.setupTimedOut, true);
        assert.match(result.error, /timeout/i);
      }
    });
  }
});
