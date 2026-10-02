// Live, blind first-proposal benchmark. Never sends an approval or saves a prediction.
import { appendFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { pathToFileURL } from "node:url";
import { sanitizeBenchmarkRecord, summarizeBenchmarkResults } from "../lib/grounding-benchmark.mjs";

// Keep these in sync with lib/grounding-request-diagnostics.ts. The marker is
// intentionally present only in benchmark-owned turns; the extension records
// an allow-listed custom entry without retaining provider request contents.
const REQUEST_DIAGNOSTICS_MARKER = "[grounding-benchmark:request-controls]";
const REQUEST_DIAGNOSTICS_ENTRY = "grounding:request-controls";

export function benchmarkPrompt(queryPath, outputDir, variant = "baseline") {
  const base = `处理这个数据集的 1 条记录，每次只加载 1 条。每个框必须等我在审核界面批准后才能保存；退回后修正当前记录，不得提前保存或加载下一条。\n数据集：${queryPath}\n输出目录：${outputDir}\ntargetCount=1，limit=1。严格保留原始 query 的身份、顺序、颜色、部位和空间关系要求；证据不足时使用 unresolved 和低置信度。\n${REQUEST_DIAGNOSTICS_MARKER}`;
  if (variant === "baseline") return base;
  if (variant !== "focused") throw new Error(`Unknown prompt variant: ${variant}`);
  return base + "\n高效定位：先从总览确定 query 所指对象、所属对象及部位；已足以定位就直接提交审核。只有一个具体可见疑点会改变选择或边界时才补看局部。每次工具返回后决定下一项动作，不重复口头推演已有证据。确定目标后用 grounding_evidence 简短锁定候选，之后只处理它的边界；颜色分析不是必做步骤。歧义确实无法从现有像素解决时保留不确定性并提交审核，不发明隐藏候选。";
}

function diagnosticFromEvent(event) {
  const entry = event?.type === "entry_appended" ? event.entry : undefined;
  const data = entry?.type === "custom" && entry.customType === REQUEST_DIAGNOSTICS_ENTRY ? entry.data : undefined;
  if (!data || typeof data !== "object" || Array.isArray(data) || data.version !== 1) return null;
  const model = {};
  if (typeof data.model?.provider === "string") model.provider = data.model.provider;
  if (typeof data.model?.id === "string") model.id = data.model.id;
  if (typeof data.model?.reasoning === "boolean") model.reasoning = data.model.reasoning;
  if (typeof data.model?.thinkingFormat === "string") model.thinkingFormat = data.model.thinkingFormat;
  const tokenCount = (value, allowZero = false) => typeof value === "number" && Number.isSafeInteger(value)
    && value >= (allowZero ? 0 : 1) && value <= 1_000_000_000;
  for (const key of ["maxTokens", "contextWindow"]) {
    if (tokenCount(data.model?.[key])) model[key] = data.model[key];
  }
  const controls = {};
  for (const key of ["max_tokens", "max_completion_tokens", "max_output_tokens", "thinking_token_budget"]) {
    if (tokenCount(data.controls?.[key], key === "thinking_token_budget")) controls[key] = data.controls[key];
  }
  const thinking = data.controls?.thinking;
  if (typeof thinking === "string") controls.thinking = thinking;
  else if (thinking && typeof thinking === "object" && !Array.isArray(thinking) && typeof thinking.type === "string") {
    controls.thinking = { type: thinking.type };
    if (tokenCount(thinking.budget_tokens, true)) controls.thinking.budget_tokens = thinking.budget_tokens;
  }
  if (typeof data.controls?.reasoning_effort === "string") controls.reasoning_effort = data.controls.reasoning_effort;
  if (typeof data.controls?.enable_thinking === "boolean") controls.enable_thinking = data.controls.enable_thinking;
  return { version: 1, model, controls };
}

async function requestJson(baseUrl, path, body, signal) {
  const response = await fetch(new URL(path, baseUrl), {
    ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(30_000)]) : AbortSignal.timeout(30_000),
  });
  let value = await response.json();
  if (typeof value === "string") value = JSON.parse(value);
  if (!response.ok || value.error) throw new Error(`${path}: ${response.status} ${value.error ?? "request failed"}`);
  return value;
}

export async function consumeEvents(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = /\r?\n\r?\n/u.exec(buffered))) {
        const frame = buffered.slice(0, boundary.index);
        buffered = buffered.slice(boundary.index + boundary[0].length);
        const data = frame.split(/\r?\n/u).filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
        if (data) onEvent(JSON.parse(data));
      }
    }
  } finally {
    reader.releaseLock();
  }
}

async function runCase({ baseUrl, cwd, key, queryPath, outputDir, artifactDir, variant, timeoutMs, setupTimeoutMs, signal, expectedModel }) {
  const setupStarted = performance.now();
  const result = { key, variant, sessionId: null, model: null, thinkingLevel: null, sessionCreation: "not_requested",
    status: "error", prediction: null, elapsedMs: 0, setupMs: 0,
    toolElapsedMs: 0, toolCounts: {}, toolErrors: 0, repeatedViews: 0, assistantResponses: 0, outputLimitStops: 0,
    thinkingBlocks: 0, thinkingCharacters: 0,
    requestControls: [],
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { total: 0 } }, events: [] };
  const streamController = new AbortController();
  const promptController = new AbortController();
  const tools = new Map();
  let started;
  let timer;
  let heartbeat;
  let connectionTimer;
  let streamTask;
  let path;
  let phase = "session_creation";
  let finished = false;
  let reviewImage;
  let settle;
  let ready;
  const completed = new Promise((r) => { settle = r; });
  const connected = new Promise((r) => { ready = r; });
  const finish = (status, error) => {
    if (finished) return;
    finished = true;
    result.status = status;
    result.elapsedMs = started === undefined ? 0 : performance.now() - started;
    if (error) { result.error = error; result.errorStage = phase; }
    settle();
  };
  const interrupt = () => {
    finish("interrupted", "Benchmark interrupted; only its own session will be aborted.");
    streamController.abort();
  };
  signal?.addEventListener("abort", interrupt, { once: true });
  try {
    if (signal?.aborted) { interrupt(); return result; }
    await writeFile(join(artifactDir, "live.json"), JSON.stringify({ key, phase: "creating_session" }, null, 2));
    // A failed response does not prove that the server failed to create a session.
    // Keep that uncertainty explicit until a usable, benchmark-owned id arrives.
    result.sessionCreation = "unknown";
    const setupSignal = AbortSignal.timeout(setupTimeoutMs);
    const created = await requestJson(baseUrl, "/api/agent/new", { cwd, type: "ensure_session" },
      signal ? AbortSignal.any([signal, setupSignal]) : setupSignal);
    if (typeof created.sessionId !== "string" || !created.sessionId.trim()) throw new Error("New session response omitted a valid sessionId");
    const sessionId = created.sessionId;
    path = `/api/agent/${encodeURIComponent(sessionId)}`;
    result.sessionId = sessionId;
    result.sessionCreation = "confirmed";
    result.model = created.model;
    result.thinkingLevel = created.thinkingLevel;
    result.setupMs = performance.now() - setupStarted;
    phase = "setup";
    if (expectedModel && `${created.model?.provider}/${created.model?.modelId}` !== expectedModel) {
      throw new Error(`Expected ${expectedModel}, got ${created.model?.provider}/${created.model?.modelId}; no defaults were changed.`);
    }
    await writeFile(join(artifactDir, "live.json"), JSON.stringify({ key, sessionId, phase: "connecting", url: new URL(`/?session=${sessionId}`, baseUrl).href }, null, 2));
    connectionTimer = setTimeout(() => streamController.abort(new Error("SSE headers timed out")), 20_000);
    const response = await fetch(new URL(`${path}/events`, baseUrl), { signal: streamController.signal });
    clearTimeout(connectionTimer);
    if (!response.ok || !response.body) throw new Error(`SSE connection failed: ${response.status}`);
    streamTask = consumeEvents(response.body, (event) => {
      if (event.type === "connected") ready();
      if (finished) return;
      const atMs = started === undefined ? 0 : performance.now() - started;
      const diagnostic = diagnosticFromEvent(event);
      if (diagnostic) result.requestControls.push(diagnostic);
      if (event.type === "message_end" && event.message?.role === "assistant") {
        result.assistantResponses++;
        for (const block of event.message.content ?? []) {
          if (block.type === "thinking") {
            result.thinkingBlocks++;
            result.thinkingCharacters += block.thinking?.length ?? 0;
          }
        }
        if (event.message.stopReason === "length") result.outputLimitStops++;
        for (const name of ["input", "output", "cacheRead", "cacheWrite", "totalTokens"]) result.usage[name] += event.message.usage?.[name] ?? 0;
        result.usage.cost.total += event.message.usage?.cost?.total ?? 0;
        result.events.push({ type: event.type, atMs, stopReason: event.message.stopReason, outputTokens: event.message.usage?.output });
      }
      if (event.type === "tool_execution_start") {
        tools.set(event.toolCallId, { started: performance.now(), toolName: event.toolName });
        result.toolCounts[event.toolName] = (result.toolCounts[event.toolName] ?? 0) + 1;
        result.events.push({ type: event.type, atMs, toolName: event.toolName });
      }
      if (event.type === "tool_execution_end") {
        const tool = tools.get(event.toolCallId);
        if (tool) { result.toolElapsedMs += performance.now() - tool.started; tools.delete(event.toolCallId); }
        const details = event.result?.details;
        if (details?.sourceReuse?.relation || details?.decisionCheckpoint) result.repeatedViews++;
        const isError = event.isError ?? event.result?.isError ?? false;
        if (isError) result.toolErrors++;
        result.events.push({ type: event.type, atMs, toolName: event.toolName, isError });
      }
      if (event.type === "extension_ui_request" && event.details?.kind === "grounding_review") {
        const { image, ...proposal } = event.details;
        result.proposal = proposal;
        result.prediction = proposal.bbox;
        result.predictionStatus = proposal.status;
        reviewImage = image;
        finish("proposal");
      } else if (event.type === "extension_ui_request" && ["select", "confirm", "input", "editor", "custom"].includes(event.method) && !event.closed) {
        finish("error", "An unexpected interactive prompt requires user input; the benchmark will not approve it.");
      }
      if (["prompt_error", "startup_error"].includes(event.type)) finish("error", event.message ?? event.errorMessage ?? "Session error");
      if (event.type === "prompt_done") finish("settled_without_proposal");
    }).then(() => finish("error", "SSE closed before a terminal result; the benchmark session will be stopped, not restarted."))
      .catch((error) => { if (!streamController.signal.aborted) finish("error", `SSE failed: ${error.message}`); });
    const readinessTimer = setTimeout(() => finish("error", "SSE did not confirm readiness"), 20_000);
    await Promise.race([connected, completed]);
    clearTimeout(readinessTimer);
    if (!finished && !signal?.aborted) {
      phase = "run";
      started = performance.now();
      timer = setTimeout(() => finish("timeout", `No first proposal within ${timeoutMs} ms`), timeoutMs);
      heartbeat = setInterval(() => console.log(JSON.stringify({ key, phase: "running", sessionId, elapsedSeconds: Math.round((performance.now() - started) / 1000), toolCounts: result.toolCounts, assistantResponses: result.assistantResponses })), 20_000);
      await writeFile(join(artifactDir, "live.json"), JSON.stringify({ key, sessionId, phase: "running", startedAt: new Date().toISOString() }, null, 2));
      const admitted = requestJson(baseUrl, path, { type: "prompt", message: benchmarkPrompt(queryPath, outputDir, variant) }, promptController.signal)
        .catch((error) => finish("error", error.message));
      await Promise.race([admitted, completed]);
      await completed;
    } else if (signal?.aborted) interrupt();
  } catch (error) {
    if (phase === "session_creation" && error.name === "TimeoutError") result.setupTimedOut = true;
    finish("error", error.message);
  } finally {
    clearTimeout(timer);
    clearInterval(heartbeat);
    clearTimeout(connectionTimer);
    signal?.removeEventListener("abort", interrupt);
    promptController.abort();
    // Abort stops this benchmark-owned session, including a pending review.
    // No extension_ui_response/confirm is ever submitted.
    if (path) {
      try {
        await requestJson(baseUrl, path, { type: "abort" });
        const state = await requestJson(baseUrl, path);
        result.stopped = state.running === false;
      } catch (error) {
        result.stopped = false;
        result.cleanupError = error.message;
      }
    } else {
      result.setupMs = performance.now() - setupStarted;
      result.stopped = result.sessionCreation === "not_requested";
      if (!result.stopped) result.cleanupError = "Session creation is unverified and no owned sessionId is known. Do not retry or guess an id; inspect the local Web before continuing.";
    }
    streamController.abort();
    await streamTask;
    for (const tool of tools.values()) {
      const end = started === undefined ? tool.started : started + result.elapsedMs;
      result.toolElapsedMs += Math.max(0, end - tool.started);
    }
    // Wall minus tool time is not provider latency: keep the label honest.
    result.nonToolElapsedMs = Math.max(0, result.elapsedMs - result.toolElapsedMs);
    result.usageCoverage = "Completed assistant responses observed before the stop condition only; in-flight/aborted provider usage may be unavailable.";
    if (reviewImage?.data) await writeFile(join(artifactDir, `review.${reviewImage.mimeType === "image/jpeg" ? "jpg" : "png"}`), Buffer.from(reviewImage.data, "base64"));
    result.savedArtifacts = [];
    for (const name of ["progress.jsonl", "queries.json", "queries.zip"]) {
      if (await stat(join(outputDir, name)).then(() => true, (error) => { if (error.code === "ENOENT") return false; throw error; })) result.savedArtifacts.push(name);
    }
    await writeFile(join(artifactDir, "live.json"), JSON.stringify({ key, sessionId: result.sessionId, phase: "finished", status: result.status,
      sessionCreation: result.sessionCreation, stopped: result.stopped }, null, 2));
  }
  return result;
}

async function main() {
  const { values } = parseArgs({ options: {
    dataset: { type: "string" }, keys: { type: "string" }, cwd: { type: "string" }, out: { type: "string" },
    "fixture-root": { type: "string" }, url: { type: "string", default: "http://127.0.0.1:30141" },
    variant: { type: "string", default: "baseline" }, "timeout-ms": { type: "string", default: "180000" },
    "setup-timeout-ms": { type: "string", default: "30000" },
    "expected-model": { type: "string" }, help: { type: "boolean", default: false },
  } });
  if (values.help) {
    console.log("node e2e/grounding-benchmark.mjs --dataset PATH --keys key1,key2 --cwd WORKSPACE --fixture-root NEW_FIXTURE_DIR --out NEW_OUTPUT_DIR [--variant baseline|focused] [--timeout-ms 180000] [--setup-timeout-ms 30000] [--expected-model provider/modelId]\nUses the Web's current default model without changing settings. Live requests incur provider usage. Captures the first review proposal and aborts without approval. Unknown session-creation outcomes terminate the batch with a recorded failure. Requires an already-running local Web.");
    return;
  }
  for (const name of ["dataset", "keys", "cwd", "fixture-root", "out"]) if (!values[name]) throw new Error(`--${name} is required`);
  const url = new URL(values.url);
  if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) throw new Error("Benchmark only connects to a local Web instance");
  const keys = [...new Set(values.keys.split(",").map((key) => key.trim()).filter(Boolean))];
  if (!keys.length || keys.length > 12 || keys.some((key) => !/^[\w-]+$/u.test(key))) throw new Error("Choose 1–12 explicit safe dataset keys; never run an entire official dataset");
  const timeoutMs = Number(values["timeout-ms"]);
  if (!Number.isFinite(timeoutMs) || timeoutMs < 1000 || timeoutMs > 900000) throw new Error("timeout-ms must be between 1000 and 900000");
  const setupTimeoutMs = Number(values["setup-timeout-ms"]);
  if (!Number.isFinite(setupTimeoutMs) || setupTimeoutMs < 1000 || setupTimeoutMs > 30000) throw new Error("setup-timeout-ms must be between 1000 and 30000");
  benchmarkPrompt("", "", values.variant);
  const datasetPath = resolve(values.dataset);
  const dataset = JSON.parse(await readFile(datasetPath, "utf8"));
  const selected = keys.map((key) => {
    if (!Object.hasOwn(dataset, key)) throw new Error(`Unknown key: ${key}`);
    return { key, record: sanitizeBenchmarkRecord(dataset[key], datasetPath) };
  });
  const outputRoot = resolve(values.out);
  const fixtureRoot = resolve(values["fixture-root"]);
  for (const directory of [outputRoot, fixtureRoot]) {
    await mkdir(dirname(directory), { recursive: true });
    await mkdir(directory); // Must be new: never overwrite a previous run.
  }
  const manifest = { schemaVersion: 1, createdAt: new Date().toISOString(), sourceDataset: datasetPath,
    keys, variant: values.variant, timeoutMs, setupTimeoutMs, evaluation: "Independent visual audit required. Dataset bbox fields are ignored: no ground truth is available.",
    groundingSourceSha256: createHash("sha256").update(await readFile(new URL("../lib/grounding-safety-extension.ts", import.meta.url))).digest("hex"),
    mode: "blind-first-proposal-no-approval", fixtures: fixtureRoot, outputRoot };
  await writeFile(join(outputRoot, "manifest.json"), JSON.stringify(manifest, null, 2));
  const controller = new AbortController();
  const interrupt = () => controller.abort();
  process.once("SIGINT", interrupt);
  process.once("SIGTERM", interrupt);
  const results = [];
  for (const { key, record } of selected) {
    if (controller.signal.aborted) break;
    const sourceDir = join(fixtureRoot, key, "source");
    const artifactDir = join(outputRoot, key);
    const outputDir = join(artifactDir, "prediction-output");
    await mkdir(sourceDir, { recursive: true });
    await mkdir(artifactDir);
    const queryPath = join(sourceDir, "queries.json");
    await writeFile(queryPath, JSON.stringify({ [key]: record }, null, 2));
    console.log(JSON.stringify({ key, phase: "starting", variant: values.variant }));
    const result = await runCase({ baseUrl: values.url, cwd: resolve(values.cwd), key, queryPath, outputDir,
      artifactDir, variant: values.variant, timeoutMs, setupTimeoutMs, signal: controller.signal, expectedModel: values["expected-model"] });
    result.visualAudit = { status: "pending", note: "Inspect original images and this proposed box. There is no ground truth; do not score against historical boxes." };
    results.push(result);
    await writeFile(join(artifactDir, "result.json"), JSON.stringify(result, null, 2));
    await appendFile(join(outputRoot, "results.jsonl"), JSON.stringify(result) + "\n");
    const terminationReason = result.sessionCreation === "unknown" ? "session_creation_unverified"
      : !result.stopped ? "session_shutdown_unverified" : result.savedArtifacts.length ? "unexpected_saved_artifacts" : null;
    const unattemptedKeys = keys.slice(results.length);
    const summary = { ...manifest, ...summarizeBenchmarkResults(results), attempted: results.length, planned: keys.length,
      runStatus: terminationReason ? "terminated" : controller.signal.aborted ? "interrupted" : unattemptedKeys.length ? "running" : "completed",
      termination: terminationReason ? { key, reason: terminationReason } : null, unattemptedKeys };
    await writeFile(join(outputRoot, "summary.json"), JSON.stringify(summary, null, 2));
    console.log(JSON.stringify({ key, status: result.status, elapsedMs: Math.round(result.elapsedMs), visualAudit: "pending", stopped: result.stopped, savedArtifacts: result.savedArtifacts }));
    if (terminationReason) throw new Error(`Safety check failed for ${key} (${terminationReason}); remaining samples were not started. Inspect ${artifactDir} before any further run`);
  }
  process.removeListener("SIGINT", interrupt);
  process.removeListener("SIGTERM", interrupt);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
