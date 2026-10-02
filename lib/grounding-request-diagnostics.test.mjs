import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";
import { AgentSession, ExtensionRunner, SessionManager } from "@earendil-works/pi-coding-agent";

const jiti = createJiti(import.meta.url);
const {
  GROUNDING_REQUEST_DIAGNOSTICS_MARKER: marker,
  GROUNDING_REQUEST_DIAGNOSTICS_ENTRY: entryType,
  groundingRequestControls,
} = await jiti.import("./grounding-request-diagnostics.ts");
const { createGroundingSafetyExtension } = await jiti.import("./grounding-safety-extension.ts");
const { toClientAgentEvent } = await jiti.import("./agent-event-wire.ts");
const { normalizeBuildSystemPromptOptions } = await import(
  new URL("./core/system-prompt.js", import.meta.resolve("@earendil-works/pi-coding-agent"))
);

const model = Object.freeze({
  provider: "linuxdo", id: "mimo-v2.5", reasoning: true,
  compat: Object.freeze({ thinkingFormat: "deepseek" }),
});
const batchPrompt = "Process 1 grounding record from queries.json using visible images.";

test("request diagnostic copies only recognized identities and control values", () => {
  const payload = {
    thinking: { type: "disabled", budget_tokens: 999, reasoning: "PRIVATE_THOUGHT" },
    reasoning_effort: "low", enable_thinking: false,
    messages: [{ content: "PRIVATE_CONVERSATION" }],
    headers: { Authorization: "PRIVATE_CREDENTIAL" },
    baseURL: "https://PRIVATE_ENDPOINT.invalid", apiKey: "PRIVATE_KEY",
  };
  const richModel = { ...model, baseUrl: "https://PRIVATE_MODEL_ENDPOINT.invalid", apiKey: "PRIVATE_MODEL_KEY" };
  const original = structuredClone({ richModel, payload });
  const result = groundingRequestControls(richModel, payload);
  assert.deepEqual(result, {
    version: 1,
    model: { provider: "linuxdo", id: "mimo-v2.5", reasoning: true, thinkingFormat: "deepseek" },
    controls: { thinking: { type: "disabled", budget_tokens: 999 }, reasoning_effort: "low", enable_thinking: false },
  });
  assert.doesNotMatch(JSON.stringify(result), /PRIVATE|https|messages|headers|apiKey/);
  assert.deepEqual({ richModel, payload }, original);
  payload.thinking.type = "enabled";
  assert.equal(result.controls.thinking.type, "disabled", "stored metadata must not alias the request");
});

test("unknown controls, URLs, credentials, arrays and accessors are omitted", () => {
  for (const value of [undefined, null, [], "PRIVATE", 3]) {
    assert.deepEqual(groundingRequestControls(value, value), { version: 1, model: {}, controls: {} });
  }
  assert.deepEqual(groundingRequestControls({
    provider: "https://private.invalid", id: "sk-private-key", reasoning: "yes",
    compat: { thinkingFormat: "PRIVATE_THOUGHT" },
  }, {
    thinking: { type: "PRIVATE_THOUGHT" }, reasoning_effort: "PRIVATE_KEY", enable_thinking: "false",
  }), { version: 1, model: {}, controls: {} });
  for (const identity of ["/private/path", "C:\\private", "name?key=private", "Bearer private", "a".repeat(161)]) {
    assert.deepEqual(groundingRequestControls({ provider: identity, id: identity }, {}).model, {});
  }
  const dangerous = Object.create({ enable_thinking: true });
  for (const key of ["thinking", "reasoning_effort", "provider", "id", "reasoning", "compat", "toJSON", "maxTokens", "contextWindow", "max_tokens", "max_completion_tokens", "max_output_tokens", "thinking_token_budget", "budget_tokens"]) {
    Object.defineProperty(dangerous, key, { get() { throw new Error("must not invoke accessors"); } });
  }
  assert.deepEqual(groundingRequestControls(dangerous, dangerous), { version: 1, model: {}, controls: {} });
});

test("known thinking representations retain booleans and omit unrelated nested fields", () => {
  for (const type of ["enabled", "disabled", "adaptive"]) {
    assert.deepEqual(groundingRequestControls({}, { thinking: { type, text: "PRIVATE" } }).controls, { thinking: { type } });
    assert.deepEqual(groundingRequestControls({}, { thinking: type }).controls, { thinking: type });
  }
  for (const effort of ["none", "off", "minimal", "low", "medium", "high", "xhigh", "max"]) {
    assert.equal(groundingRequestControls({}, { reasoning_effort: effort }).controls.reasoning_effort, effort);
  }
  assert.deepEqual(groundingRequestControls({ provider: "openrouter", id: "vendor/model-v1", reasoning: false }, {
    enable_thinking: true, chat_template_kwargs: { enable_thinking: false },
  }), { version: 1, model: { provider: "openrouter", id: "vendor/model-v1", reasoning: false }, controls: { enable_thinking: true } });
});

test("budgets are bounded integers, detached and do not expose request content", () => {
  const limits = Object.freeze({ maxTokens: 32768, contextWindow: 131072, endpoint: "PRIVATE" });
  const payload = Object.freeze({
    max_tokens: 16000, max_completion_tokens: 17000, max_output_tokens: 18000,
    thinking_token_budget: 0,
    thinking: Object.freeze({ type: "enabled", budget_tokens: 2048, text: "PRIVATE" }),
    messages: [{ content: [{ type: "image", data: "PRIVATE" }] }], apiKey: "PRIVATE",
  });
  assert.deepEqual(groundingRequestControls(limits, payload), {
    version: 1, model: { maxTokens: 32768, contextWindow: 131072 },
    controls: { max_tokens: 16000, max_completion_tokens: 17000, max_output_tokens: 18000,
      thinking_token_budget: 0, thinking: { type: "enabled", budget_tokens: 2048 } },
  });
  for (const invalid of [-1, 1.5, NaN, Infinity, -Infinity, 1_000_000_001, Number.MAX_SAFE_INTEGER, "4096", null, {}, []]) {
    const data = Object.fromEntries(["maxTokens", "contextWindow", "max_tokens", "max_completion_tokens", "max_output_tokens", "thinking_token_budget"].map((key) => [key, invalid]));
    data.thinking = { type: "enabled", budget_tokens: invalid };
    assert.deepEqual(groundingRequestControls(data, data), { version: 1, model: {}, controls: { thinking: { type: "enabled" } } });
  }
  assert.deepEqual(groundingRequestControls({ maxTokens: 0, contextWindow: 0 }, { max_tokens: 0, max_completion_tokens: 0, max_output_tokens: 0 }).controls, {});
  const nested = Object.create({ budget_tokens: 300 });
  Object.defineProperty(nested, "type", { value: "enabled" });
  Object.defineProperty(nested, "budget_tokens", { get() { throw new Error("must not read budget accessor"); } });
  assert.deepEqual(groundingRequestControls({}, { thinking: nested }).controls, { thinking: { type: "enabled" } });
});

/** Real SDK hook dispatch and appendEntry binding, with no provider or filesystem. */
function harness({ appendFailure = false, earlier = [], later = [] } = {}) {
  const cwd = process.cwd();
  const sessionManager = SessionManager.inMemory(cwd);
  const events = [];
  const session = Object.assign(Object.create(AgentSession.prototype), {
    sessionManager, _eventListeners: [(event) => events.push(event)],
  });
  let sdkAppend;
  session._bindExtensionCore({ bindCore(actions) { sdkAppend = actions.appendEntry; } });
  const handlers = new Map();
  let activeTools = ["read", "write", "bash"];
  createGroundingSafetyExtension({ cwd, sessionId: "request-diagnostics-test" }).factory({
    on(name, handler) { handlers.set(name, handler); },
    registerTool(tool) { activeTools.push(tool.name); },
    getActiveTools: () => [...activeTools],
    setActiveTools(tools) { activeTools = [...tools]; },
    appendEntry(type, data) {
      if (appendFailure) throw new Error("diagnostic storage unavailable");
      sdkAppend(type, data);
    },
  });
  const runner = new ExtensionRunner([{
    path: "request-diagnostics-integration",
    handlers: new Map([
      ["before_agent_start", [handlers.get("before_agent_start")]],
      ["session_start", [handlers.get("session_start")]],
      ["before_provider_request", [...earlier, handlers.get("before_provider_request"), ...later]],
    ]),
  }], {}, cwd, sessionManager, {});
  runner.getModel = () => model;
  const errors = [];
  runner.emitError = (error) => errors.push(error);
  return {
    runner, sessionManager, events, errors,
    start: (prompt) => runner.emitBeforeAgentStart(prompt, undefined, normalizeBuildSystemPromptOptions({ cwd })),
    request: (payload) => runner.emitBeforeProviderRequest(payload),
    diagnostics: () => sessionManager.getEntries().filter((entry) => entry.type === "custom" && entry.customType === entryType),
  };
}

test("SDK hook is disabled by default and requires exact current marker plus active batch", async (t) => {
  for (const prompt of [
    batchPrompt,
    `${batchPrompt} [grounding-benchmark:request-control]`,
    `${batchPrompt} [Grounding-benchmark:request-controls]`,
    `${marker} Explain a TypeScript type.`,
    `${marker} Process 1 record.`, // marker itself must not activate grounding
    `${marker} Locate the grounding target.`, // grounding, but not a batch
  ]) {
    await t.test(prompt, async () => {
      const h = harness();
      const payload = Object.freeze({ thinking: Object.freeze({ type: "disabled" }) });
      assert.strictEqual(await h.request(payload), payload);
      await h.start(prompt);
      assert.strictEqual(await h.request(payload), payload);
      assert.deepEqual(h.diagnostics(), []);
      assert.deepEqual(h.errors, []);
    });
  }
});

test("SDK marked batch records real controls, emits SSE-safe custom entries, and leaves payload unchanged", async () => {
  const h = harness();
  await h.start(`${batchPrompt}\n${marker}`);
  for (const type of ["disabled", "enabled"]) {
    const payload = Object.freeze({ thinking: Object.freeze({ type }), messages: Object.freeze(["PRIVATE"]) });
    assert.strictEqual(await h.request(payload), payload);
  }
  assert.deepEqual(h.diagnostics().map((entry) => entry.data.controls.thinking), [{ type: "disabled" }, { type: "enabled" }]);
  assert.equal(h.events.length, 2);
  for (const event of h.events) {
    assert.equal(event.type, "entry_appended");
    assert.equal(event.entry.customType, entryType);
    assert.strictEqual(toClientAgentEvent(event), event, "SSE wire filter must preserve the SDK event");
    assert.doesNotMatch(JSON.stringify(event), /PRIVATE|messages/);
  }
  assert.deepEqual(h.sessionManager.buildSessionContext().messages, [], "plain custom entries never become model conversation");
  assert.deepEqual(h.errors, []);
});

test("unmarked next user turn and session restoration disable diagnostics", async () => {
  const h = harness();
  const payload = { thinking: { type: "disabled" } };
  await h.start(`${batchPrompt} ${marker}`);
  await h.request(payload);
  await h.start("Continue the grounding batch.");
  await h.request(payload);
  assert.equal(h.diagnostics().length, 1);
  await h.start(`Continue the grounding batch. ${marker}`);
  await h.request(payload);
  assert.equal(h.diagnostics().length, 2);
  h.sessionManager.appendMessage({ role: "user", content: `${batchPrompt} ${marker}`, timestamp: 1 });
  await h.runner.emit({ type: "session_start" });
  await h.request(payload);
  assert.equal(h.diagnostics().length, 2, "restoring marked history must not restore the opt-in");
  assert.deepEqual(h.errors, []);
});

test("restoring a marker-only prompt cannot activate grounding on the next marked batch-like turn", async () => {
  const h = harness();
  h.sessionManager.appendMessage({ role: "user", content: `${marker} Explain a type.`, timestamp: 1 });
  await h.runner.emit({ type: "session_start" });
  await h.start(`${marker} Process 1 record.`);
  await h.request({ enable_thinking: true });
  assert.deepEqual(h.diagnostics(), []);
  assert.deepEqual(h.errors, []);
});

test("SDK payload pipeline still honors other extensions and diagnostic write failure cannot alter requests", async () => {
  const replacement = Object.freeze({ thinking: Object.freeze({ type: "disabled" }), messages: ["PRIVATE"] });
  let observed;
  const h = harness({ earlier: [() => replacement], later: [(event) => { observed = event.payload; }] });
  await h.start(`${batchPrompt} ${marker}`);
  assert.strictEqual(await h.request({ enable_thinking: true }), replacement);
  assert.strictEqual(observed, replacement);
  assert.deepEqual(h.diagnostics()[0].data.controls, { thinking: { type: "disabled" } });
  assert.deepEqual(h.errors, []);

  const failing = harness({ appendFailure: true });
  await failing.start(`${batchPrompt} ${marker}`);
  assert.strictEqual(await failing.request(replacement), replacement);
  assert.equal(failing.errors.length, 1);
  assert.equal(failing.errors[0].event, "before_provider_request");
  assert.deepEqual(failing.diagnostics(), []);
});
