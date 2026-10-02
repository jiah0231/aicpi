import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";
import { Agent } from "@earendil-works/pi-agent-core";
import { AgentSession } from "@earendil-works/pi-coding-agent";
import { AssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
const { parseGroundingGenerationControl: parse, applyGroundingGenerationLimit: apply, createGroundingGenerationControl: create } = await createJiti(import.meta.url).import("./grounding-generation-control.ts");
const marker = "[grounding-generation:max-output-tokens=8192]";
const lengthMessage = { role: "assistant", stopReason: "length", content: [], provider: "test", model: "test", api: "openai-completions", timestamp: 1, usage: { input: 20, output: 8192, totalTokens: 8212, cacheRead: 0, cacheWrite: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
function harness(active = true) {
  let controller = new AbortController();
  const entries = [], messages = [], order = [];
  const control = create({ appendEntry: (...args) => entries.push(args), sendMessage: (...args) => messages.push(args) }, () => active, () => order.push("blocked"));
  const ctx = { get signal() { return controller.signal; }, abort() { order.push("abort"); controller.abort(); } };
  return { control, ctx, entries, messages, order, newRun() { controller = new AbortController(); } };
}
test("generation budget is explicit, stripped, validated and absent by default", () => {
  assert.deepEqual(parse("Inspect deeply"), { prompt: "Inspect deeply" });
  assert.deepEqual(parse(`${marker} inspect`), { prompt: " inspect", maxOutputTokens: 8192 });
  for (const value of ["0", "-1", "1.5", "NaN", "8192 extra", "1000000001", "9007199254740993"]) assert.ok(parse(`[grounding-generation:max-output-tokens=${value}]`).error);
  assert.ok(parse(`${marker}${marker}`).error);
});
test("supported payload limits preserve reasoning, identity and lower pre-existing caps", () => {
  for (const field of ["max_tokens", "max_output_tokens", "max_completion_tokens"]) {
    const payload = Object.freeze({ [field]: 32768, reasoning_effort: "high", model: "mimo-v2.5", thinking: { type: "enabled" } });
    const result = apply(payload, 8192);
    assert.deepEqual(result.payload, { ...payload, [field]: 8192 });
    assert.equal(payload[field], 32768);
    assert.equal(apply({ [field]: 4096 }, 8192).payload[field], 4096);
  }
  assert.deepEqual(apply({ max_tokens: 100, max_completion_tokens: 200 }, 150).payload, { max_tokens: 100, max_completion_tokens: 150 });
});
test("unknown or conflicting provider controls are blocked rather than silently ignored", () => {
  for (const payload of [null, [], {}, { generationConfig: { maxOutputTokens: 8192 } }, { max_tokens: "10000" }, { max_tokens: 0 }, { max_tokens: 32768, thinking: { budget_tokens: 8192 } }, { max_tokens: 32768, thinking_token_budget: 9000 }]) assert.ok(apply(payload, 8192).error);
  assert.ok(apply({ max_tokens: 32768, thinking: { budget_tokens: 4096 } }, 8192).payload);
});
test("default and non-grounding turns never modify or abort a request", () => {
  for (const active of [false, true]) {
    const h = harness(active);
    h.control.startTurn(active ? "normal" : marker);
    assert.equal(h.control.beforeRequest({ max_tokens: 32768 }, h.ctx), undefined);
    h.control.messageEnd(lengthMessage, h.ctx);
    assert.equal(h.messages.length, 0);
    assert.equal(h.ctx.signal.aborted, false);
  }
});
test("length abort is synchronous, one-shot, blocks auto-compaction and does not trigger a new turn", () => {
  const h = harness();
  h.control.startTurn(marker);
  assert.equal(h.control.beforeRequest({ max_tokens: 32768 }, h.ctx).max_tokens, 8192);
  h.control.messageEnd(lengthMessage, h.ctx);
  h.control.messageEnd(lengthMessage, h.ctx);
  assert.deepEqual(h.order, ["abort", "blocked"]);
  assert.equal(h.messages.length, 2);
  assert.ok(h.messages.every(([, options]) => options.triggerTurn === false));
  assert.match(h.messages[1][0].content, /pending/);
  assert.deepEqual(h.control.beforeCompact("threshold"), { cancel: true });
  assert.deepEqual(h.control.beforeCompact("overflow"), { cancel: true });
  assert.equal(h.control.beforeCompact("manual"), undefined);
  h.newRun();
  h.control.startTurn("Continue with full reasoning");
  assert.equal(h.control.beforeRequest({ max_tokens: 32768 }, h.ctx), undefined);
  h.control.messageEnd(lengthMessage, h.ctx);
  assert.equal(h.ctx.signal.aborted, false);
  assert.equal(h.control.beforeCompact("threshold"), undefined);
});
test("stale request signals and session resets cannot abort another run", () => {
  const h = harness();
  h.control.startTurn(marker);
  h.control.beforeRequest({ max_tokens: 32768 }, h.ctx);
  h.newRun();
  h.control.messageEnd(lengthMessage, h.ctx);
  assert.equal(h.ctx.signal.aborted, false);
  h.control.reset();
  h.control.messageEnd(lengthMessage, h.ctx);
  assert.equal(h.ctx.signal.aborted, false);
});
test("unsupported or invalid requested budget aborts before request without claiming enforcement", () => {
  for (const [prompt, payload] of [[marker, {}], ["[grounding-generation:unknown=1]", { max_tokens: 32768 }]]) {
    const h = harness(); h.control.startTurn(prompt);
    assert.equal(h.control.beforeRequest(payload, h.ctx), undefined);
    assert.equal(h.ctx.signal.aborted, true);
    assert.equal(h.entries[0][1].status, "blocked");
    assert.equal(h.messages.length, 1);
  }
});
test("SDK 0.87.1 rejects every truncated tool and session abort skips post-run retries", async () => {
  const h = harness(); h.control.startTurn(marker);
  let executed = 0, requests = 0, cancelledStreams = 0;
  const agent = new Agent({ initialState: { model: { id: "test", provider: "test", api: "openai-completions" }, tools: [{ name: "save", label: "save", description: "save", parameters: { type: "object", properties: {} }, execute: async () => { executed++; return { content: [] }; } }] },
    streamFn: (_model, _context, options) => {
      const stream = new AssistantMessageEventStream();
      if (options.signal.aborted) { cancelledStreams++; stream.push({ type: "error", reason: "aborted", error: { ...lengthMessage, stopReason: "aborted" } }); return stream; }
      requests++;
      h.control.beforeRequest({ max_tokens: 32768 }, ctx);
      const msg = { ...lengthMessage, content: [{ type: "toolCall", id: "save-1", name: "save", arguments: {} }] };
      stream.push({ type: "done", reason: "length", message: msg }); return stream;
    } });
  const session = Object.assign(Object.create(AgentSession.prototype), { agent, _isAgentRunActive: true, _agentRunAbortRequested: false, abortRetry() {}, abortCompaction() {}, abortBranchSummary() {}, waitForIdle: async () => {} });
  const ctx = { get signal() { return agent.signal; }, abort() { void session.abort(); } };
  agent.subscribe((event) => { if (event.type === "message_end") h.control.messageEnd(event.message, ctx); });
  await agent.prompt(marker);
  assert.equal(executed, 0);
  assert.equal(requests, 1);
  assert.equal(cancelledStreams, 1); // SDK visits another stream, already aborted; no provider request.
  assert.equal(session._agentRunAbortRequested, true);
  session._finishCancelledRetry = () => {};
  session._checkCompaction = () => { throw new Error("must not compact/retry"); };
  assert.equal(await session._handlePostAgentRun(), false);
});

test("installed OpenAI-compatible transport does not fetch after an unsupported budget abort", async () => {
  const { stream } = await import("@earendil-works/pi-ai/api/openai-completions");
  const h = harness(); h.control.startTurn(marker);
  let fetches = 0;
  const result = await stream({ id: "test", name: "Test", provider: "test", api: "openai-completions", baseUrl: "https://test.invalid/v1", reasoning: false, input: ["text"], contextWindow: 32768, maxTokens: 16384, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }, { messages: [{ role: "user", content: "test", timestamp: 1 }] }, {
    apiKey: "test-only", signal: h.ctx.signal,
    onPayload: () => h.control.beforeRequest({}, h.ctx),
    fetch: async () => { fetches++; throw new Error("No network permitted"); },
  }).result();
  assert.equal(fetches, 0);
  assert.equal(result.stopReason, "aborted");
});

test("diagnostic or notice persistence failure cannot remove the requested payload cap", () => {
  const control = create({ appendEntry() { throw new Error("storage failure"); }, sendMessage() { throw new Error("notice failure"); } }, () => true, () => {});
  control.startTurn(marker);
  const controller = new AbortController();
  const ctx = { signal: controller.signal, abort() { controller.abort(); } };
  assert.equal(control.beforeRequest({ max_tokens: 32768 }, ctx).max_tokens, 8192);
  control.messageEnd(lengthMessage, ctx);
  assert.equal(controller.signal.aborted, true);
});

test("installed Agent consumes queued steering/follow-up budgets only at the next user boundary", async () => {
  for (const queue of ["steer", "followUp"]) {
    for (const [initial, queued, expected] of [[marker, "continue", [8192, 32768]], ["inspect", marker, [32768, 8192]]]) {
      const h = harness(); h.control.startTurn(initial);
      const caps = [];
      let agent;
      const ctx = { get signal() { return agent.signal; }, abort() { agent.abort(); } };
      agent = new Agent({ initialState: { model: { id: "test", provider: "test", api: "openai-completions" } }, streamFn: () => {
        const source = { max_tokens: 32768 };
        caps.push((h.control.beforeRequest(source, ctx) ?? source).max_tokens);
        if (caps.length === 1) agent[queue]({ role: "user", content: queued, timestamp: 2 });
        assert.ok(caps.length <= 2);
        const stream = new AssistantMessageEventStream();
        stream.push({ type: "done", reason: "stop", message: { ...lengthMessage, stopReason: "stop", content: [{ type: "text", text: "done" }] } });
        return stream;
      } });
      agent.subscribe((event) => { if (event.type === "message_end") h.control.messageEnd(event.message, ctx); });
      await agent.prompt(initial);
      assert.deepEqual(caps, expected, queue);
    }
  }
});

test("main grounding extension keeps a controlled payload when diagnostics persistence throws", async () => {
  const { createGroundingSafetyExtension } = await createJiti(import.meta.url).import("./grounding-safety-extension.ts");
  const { ExtensionRunner, SessionManager } = await import("@earendil-works/pi-coding-agent");
  const { normalizeBuildSystemPromptOptions } = await import(new URL("./core/system-prompt.js", import.meta.resolve("@earendil-works/pi-coding-agent")));
  const handlers = new Map(); let activeTools = ["read"];
  createGroundingSafetyExtension({ cwd: process.cwd(), sessionId: "generation-integration" }).factory({
    on(name, handler) { handlers.set(name, handler); }, registerTool() {}, getActiveTools: () => activeTools, setActiveTools: (tools) => { activeTools = tools; },
    appendEntry() { throw new Error("diagnostic storage failure"); }, sendMessage() {},
  });
  const runner = new ExtensionRunner([{ path: "generation-test", handlers: new Map([...handlers].map(([name, handler]) => [name, [handler]])) }], {}, process.cwd(), SessionManager.inMemory(process.cwd()), {});
  const errors = []; runner.emitError = (error) => errors.push(error);
  const notices = [];
  runner.setUIContext({ ...runner.uiContext, notify: (text, kind) => notices.push({ text, kind }) });
  await runner.emitBeforeAgentStart(`Process 1 grounding record from queries.json using visible images. ${marker} [grounding-benchmark:request-controls]`, undefined, normalizeBuildSystemPromptOptions({ cwd: process.cwd() }));
  assert.equal((await runner.emitBeforeProviderRequest({ max_tokens: 32768 })).max_tokens, 8192);
  assert.deepEqual(errors, []);
  assert.equal(notices.length, 1);
  assert.equal(notices[0].kind, "warning");
  assert.match(notices[0].text, /diagnostics could not be saved/);
  assert.match(notices[0].text, /budget is still applied/);
  runner.setUIContext({ ...runner.uiContext, notify() { throw new Error("UI unavailable"); } });
  assert.equal((await runner.emitBeforeProviderRequest({ max_tokens: 32768 })).max_tokens, 8192);
  assert.deepEqual(errors, []);
});
