import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import test from "node:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createJiti } from "jiti";
import { AgentSession, ExtensionRunner, SessionManager } from "@earendil-works/pi-coding-agent";
import JSZip from "jszip";
import sharp from "sharp";

const { normalizeBuildSystemPromptOptions, buildSystemPrompt, buildSystemPromptSections, buildSystemPromptState } =
  await import(new URL("./core/system-prompt.js", import.meta.resolve("@earendil-works/pi-coding-agent")));

const {
  compactCompletedGroundingContext,
  createGroundingSafetyExtension,
  describeGroundingCandidateChange,
  expectedGroundingOrdinal,
  validateGroundingReviewResponse,
  redactGroundingToolText,
  sanitizeGroundingQueryJson,
} = await createJiti(import.meta.url).import("./grounding-safety-extension.ts");

function assistantMessage(content) {
  return {
    role: "assistant",
    content,
    api: "openai-completions",
    provider: "test",
    model: "test",
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "toolUse",
    timestamp: 1,
  };
}

function toolResult(toolName, content, timestamp = 1, toolCallId = `${toolName}-${timestamp}`) {
  return {
    role: "toolResult",
    toolCallId,
    toolName,
    content,
    isError: false,
    timestamp,
  };
}

test("completed grounding images and reasoning are omitted from later provider context", () => {
  const oldImage = { type: "image", data: "old", mimeType: "image/png" };
  const currentImage = { type: "image", data: "current", mimeType: "image/png" };
  const messages = [
    { role: "user", content: "process three records", timestamp: 0 },
    assistantMessage([
      { type: "thinking", thinking: "long old reasoning" },
      { type: "toolCall", id: "next-1", name: "grounding_next_batch", arguments: {} },
    ]),
    toolResult("grounding_next_batch", [{ type: "text", text: "old record" }, oldImage], 2, "next-1"),
    assistantMessage([
      { type: "thinking", thinking: "more old reasoning" },
      { type: "toolCall", id: "save-next", name: "grounding_save_and_next", arguments: {} },
    ]),
    toolResult("grounding_save_and_next", [{ type: "text", text: "current record" }, currentImage], 3, "save-next"),
    assistantMessage([{ type: "thinking", thinking: "current reasoning" }]),
  ];

  const compacted = compactCompletedGroundingContext(messages, true);
  assert.equal(compacted.length, 4);
  assert.equal(compacted[0].role, "user");
  assert.deepEqual(compacted[1].content.map((block) => block.type), ["toolCall"]);
  assert.equal(compacted[1].content[0].id, "save-next");
  assert.equal(compacted[2].content.some((block) => block.type === "image" && block.data === "current"), true);
  assert.equal(compacted[3].content[0].thinking, "current reasoning");

  const noActiveRecord = compactCompletedGroundingContext(messages, false);
  assert.deepEqual(noActiveRecord.map((message) => message.role), ["user", "assistant", "toolResult"]);
  assert.deepEqual(noActiveRecord[2].content, [{ type: "text", text: "current record" }]);
});

for (const siblingFirst of [true, false]) {
  test(`context keeps paired sibling tool calls when sibling result is ${siblingFirst ? "before" : "after"} the record`, () => {
    const load = toolResult("grounding_next_batch", [{ type: "text", text: "record" }, { type: "image", data: "image", mimeType: "image/png" }], 2, "load");
    const sibling = toolResult("grounding_status", [{ type: "text", text: "status" }], 2, "status");
    const messages = [
      { role: "user", content: "continue", timestamp: 0 },
      assistantMessage([{ type: "toolCall", id: "load", name: "grounding_next_batch", arguments: {} },
        { type: "toolCall", id: "status", name: "grounding_status", arguments: {} }]),
      ...(siblingFirst ? [sibling, load] : [load, sibling]),
    ];
    for (const active of [true, false]) {
      const compacted = compactCompletedGroundingContext(messages, active);
      const ids = compacted.filter((m) => m.role === "assistant").flatMap((m) => m.content.filter((b) => b.type === "toolCall").map((b) => b.id));
      const results = compacted.filter((m) => m.role === "toolResult").map((m) => m.toolCallId);
      assert.deepEqual([...ids].sort(), [...results].sort());
      assert.equal(ids.length, 2);
    }
  });
}

test("candidate drift reports material movement without changing the proposed box", () => {
  const previous = [0.594, 0.507, 0.616, 0.549];
  const next = [0.5955, 0.482, 0.6075, 0.524];
  const change = describeGroundingCandidateChange(previous, next, 1920, 1080);
  assert.deepEqual(change.previousBbox, previous);
  assert.ok(Math.abs(change.centerDeltaPixels[0] + 6.72) < .01);
  assert.ok(Math.abs(change.centerDeltaPixels[1] + 27) < .01);
  assert.ok(change.iou < .2);
  assert.equal(change.materialChange, true);
  assert.match(change.note, /object and requested part/);
});

function installExtension(extension, handlers, tools = new Map(), runtime = {}) {
  let activeTools = ["read", "write", "bash"];
  extension.factory({
    on(name, handler) {
      handlers.set(name, handler);
      return () => {};
    },
    registerTool(tool) {
      tools.set(tool.name, tool);
      activeTools.push(tool.name);
    },
    getActiveTools: () => [...activeTools],
    setActiveTools(nextTools) {
      activeTools = [...nextTools];
    },
    setModel: runtime.setModel ?? (async () => true),
    sendUserMessage: runtime.sendUserMessage ?? (() => {}),
    sendMessage: runtime.sendMessage ?? (() => {}),
    appendEntry: runtime.appendEntry ?? (() => {}),
  });
  return tools;
}

test("grounding tool schemas avoid tuple-style items arrays rejected by OpenAI-compatible providers", () => {
  const tools = installExtension(
    createGroundingSafetyExtension({ cwd: process.cwd(), sessionId: "schema-compatibility" }),
    new Map(),
  );
  const visit = (schema, path) => {
    if (!schema || typeof schema !== "object") return;
    assert.equal(Array.isArray(schema.items), false, `${path} uses tuple-style JSON Schema items`);
    for (const [key, value] of Object.entries(schema)) visit(value, `${path}.${key}`);
  };
  for (const [name, tool] of tools) visit(tool.parameters, name);

  const color = tools.get("grounding_color_region").parameters.properties;
  assert.deepEqual(
    [color.region.minItems, color.region.maxItems, color.point.minItems, color.point.maxItems],
    [4, 4, 2, 2],
  );
  const save = tools.get("grounding_save_result").parameters.properties.bbox;
  assert.deepEqual([save.minItems, save.maxItems], [4, 4]);
});

function confirmedReviewResponse(details) {
  const proposedStatus = details.modelProposal?.status ?? details.status;
  const proposedConfidence = details.modelProposal?.confidence ?? details.confidence;
  const targetFound = proposedStatus !== "unresolved";
  const candidateCount = targetFound ? Math.max(1, details.candidateCount, details.expectedOrdinal ?? 0) : 0;
  return {
    type: "grounding_review_response",
    action: "confirm",
    constraintsResolved: true, // Mock the human's explicit resolution, never model approval.
    bbox: details.bbox,
    status: targetFound ? proposedStatus : "unresolved",
    confidence: targetFound ? proposedConfidence : Math.min(proposedConfidence, 0.4),
    targetFound,
    candidateCount,
    ...(targetFound ? { candidateRank: details.expectedOrdinal ?? details.candidateRank ?? 1 } : {}),
    reason: targetFound ? "Browser verified the visible target and candidate rank." : "Browser found no supported target in the image.",
  };
}

function createReviewContext(responseFactory = confirmedReviewResponse) {
  return {
    ui: {
      custom: async (factory) => {
        const component = await factory({}, {}, {}, () => {});
        const details = component?.groundingReview;
        if (!details) throw new Error("Test review factory did not expose grounding details.");
        return responseFactory(details);
      },
    },
    isIdle: () => true,
    abort: () => {},
  };
}

function createDeferredReviewContext() {
  const pending = [];
  const waiting = [];
  const opened = [];
  const context = {
    ui: {
      custom: (factory, options) => new Promise((resolve) => {
        let component;
        let completed = false;
        const done = (response) => {
          if (completed) return;
          completed = true;
          component?.dispose();
          resolve(response);
        };
        component = factory({}, {}, {}, done);
        if (completed) component.dispose();
        const panel = {
          details: component.groundingReview,
          options,
          get completed() { return completed; },
          input: (response) => component.handleInput(typeof response === "string" ? response : JSON.stringify(response)),
        };
        opened.push(panel);
        const notify = waiting.shift();
        if (notify) notify(panel);
        else pending.push(panel);
      }),
    },
    isIdle: () => true,
    abort: () => {},
  };
  return {
    context,
    opened,
    nextReview: () => pending.length ? Promise.resolve(pending.shift()) : new Promise((resolve) => waiting.push(resolve)),
  };
}

async function createReviewFixture(t, count = 2) {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-review-gate-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sourceDir = join(root, "source");
  const outputDir = join(root, "output");
  await mkdir(sourceDir, { recursive: true });
  const sourceImage = await sharp({
    create: { width: 64, height: 48, channels: 3, background: { r: 20, g: 30, b: 40 } },
  }).png().toBuffer();
  await writeFile(join(sourceDir, "fixture.png"), sourceImage);
  const queryPath = join(sourceDir, "queries.json");
  const queries = Object.fromEntries(["one", "two", "three"].slice(0, count).map((key) => [
    key, { visible: "fixture.png", query: key === "two" ? "the second synthetic object" : `the synthetic object ${key}` },
  ]));
  await writeFile(queryPath, JSON.stringify(queries), "utf8");
  const handlers = new Map();
  const entries = [];
  const tools = installExtension(createGroundingSafetyExtension({ cwd: root, sessionId: "review-gate" }), handlers, new Map(), {
    appendEntry(customType, data) { entries.push({ type: "custom", customType, data: structuredClone(data) }); },
  });
  await handlers.get("before_agent_start")({
    prompt: `批处理 ${count} 条 ${queryPath}`,
    systemPromptOptions: normalizeBuildSystemPromptOptions({ cwd: root, selectedTools: [...tools.keys()] }),
  });
  await tools.get("grounding_next_batch").execute("next-one", { queryPath, outputDir });
  return {
    root, outputDir, handlers, tools, entries,
    params: { queryPath, outputDir, key: "one", bbox: [0.2, 0.3, 0.21, 0.32], status: "ok", confidence: 0.9,
      reason: "The compact dark target is visibly enclosed by these exact edges." },
  };
}

async function assertNoSavedArtifacts(outputDir) {
  for (const name of ["progress.jsonl", "queries.json", "queries.zip"]) {
    await assert.rejects(readFile(join(outputDir, name)), { code: "ENOENT" });
  }
}

// Exercise the installed SDK's boundary preview, draft commit, extension runner,
// and post-commit eligibility check. Only the low-level provider/queue is faked;
// no credentials, model requests, or on-disk SDK session are involved.
function createSdkSettleHarness(handler, { stopReason = "stop", outcome = "completed", earlierHandlers = [], userPrompt = "Process the requested grounding record." } = {}) {
  const cwd = process.cwd();
  const sessionManager = SessionManager.inMemory(cwd);
  sessionManager.appendMessage({ role: "user", content: userPrompt, timestamp: 1 });
  const queue = [];
  const errors = [];
  const previews = [];
  const runner = new ExtensionRunner([{
    path: "grounding-settle-test",
    handlers: new Map([["agent_before_settle", [...earlierHandlers, (event, ctx) => {
      previews.push(event.context);
      return handler(event, ctx);
    }]]]),
  }], {}, cwd, sessionManager, {});
  runner.emitError = (error) => errors.push(error);
  const session = Object.assign(Object.create(AgentSession.prototype), {
    sessionManager,
    agent: { state: { messages: [] }, peekQueuedMessages: () => [...queue], hasQueuedMessages: () => queue.length > 0 },
    _cwd: cwd,
    _extensionRunner: runner,
    _pendingCustomMessages: [],
    _entryIdsByMessage: new Map(),
    _eventListeners: [],
    _lastActivityOutcome: outcome,
  });
  const appendAssistant = (reason = stopReason) => {
    sessionManager.appendMessage({ ...assistantMessage([{ type: "text", text: "Inspecting the target." }]), stopReason: reason });
    session._refreshFinalizedContext();
  };
  appendAssistant();
  return {
    session, queue, errors, previews, appendAssistant,
    settle: () => session._runBeforeSettleBoundary(),
    customMessages: () => sessionManager.getEntries().filter((entry) => entry.type === "custom_message"),
  };
}

function createSdkPromptRunner(handlers, cwd = process.cwd()) {
  const errors = [];
  const runner = new ExtensionRunner([{
    path: "grounding-prompt-integration",
    handlers: new Map([["before_agent_start", [handlers.get("before_agent_start")]]]),
  }], {}, cwd, SessionManager.inMemory(cwd), {});
  runner.emitError = (error) => errors.push(error);
  return { runner, errors };
}

test("SDK active-batch rendering uses the compact specialist prompt and fresh job state without mutating base options", async (t) => {
  const { root, outputDir, handlers, tools, params } = await createReviewFixture(t, 2);
  const { runner, errors } = createSdkPromptRunner(handlers, root);
  const base = normalizeBuildSystemPromptOptions({
    cwd: root,
    selectedTools: [...tools.keys()],
    toolSnippets: Object.fromEntries([...tools].map(([name, tool]) => [name, tool.promptSnippet ?? ""])),
    toolGuidelines: Object.fromEntries([...tools].map(([name, tool]) => [name, tool.promptGuidelines ?? []])),
    appendSystemPrompt: "  USER ADDENDUM\r\n保留目标所属对象\t ",
    sections: { user_section: " USER CUSTOM SECTION\r\n " },
  });
  const original = structuredClone(base);
  const originalSections = buildSystemPromptSections(base);
  const first = await runner.emitBeforeAgentStart("继续当前记录", undefined, base);
  const options = first.systemPromptOptions;
  const state = buildSystemPromptState(options);
  assert.match(state.sections.preamble, /visual grounding specialist/);
  assert.equal(options.forceSystemPrompt, undefined);
  for (const section of ["tools", "rules", "docs"]) assert.equal(state.sections[section], undefined);
  assert.equal(state.sections.addendum, originalSections.addendum);
  assert.equal(state.sections.user_section, originalSections.user_section);
  const contract = state.sections.grounding_runtime_safety;
  assert.match(contract, /no obligatory crop/);
  assert.match(contract, /Every result, including unresolved, needs human approval/);
  assert.match(contract, /Never approve on the user's behalf/);
  assert.ok(contract.includes(`Current dataset: ${params.queryPath}. Current output directory: ${outputDir}.`));
  assert.match(contract, /"targetCount":2/);
  assert.match(contract, /"approvedInJob":0/);
  assert.match(contract, /"currentKey":"one"/);
  assert.deepEqual(base, original, "SDK normalization must clone app-owned prompt mutations per invocation");
  assert.deepEqual(options.selectedTools, base.selectedTools, "prompt specialization cannot change available tools");

  await tools.get("grounding_save_result").execute("prompt-approved", params, undefined, undefined, createReviewContext());
  const second = await runner.emitBeforeAgentStart("继续当前批次", undefined, base);
  assert.match(second.systemPromptOptions.customPrompt, /visual grounding specialist/);
  assert.match(second.systemPromptOptions.sections.grounding_runtime_safety, /"approvedInJob":1/);
  assert.match(second.systemPromptOptions.sections.grounding_runtime_safety, /"currentKey":null/);
  assert.doesNotMatch(second.systemPromptOptions.sections.grounding_runtime_safety, /"currentKey":"one"/);
  assert.deepEqual(base, original);
  assert.deepEqual(errors, []);
});

test("SDK normal nonbatch prompts keep their existing role and grounding-only nonbatch uses legacy safety", async (t) => {
  for (const [prompt, grounding] of [["Explain this TypeScript type", false], ["请进行图像定位", true]]) {
    await t.test(prompt, async () => {
      const handlers = new Map();
      installExtension(createGroundingSafetyExtension({ cwd: process.cwd(), sessionId: "nonbatch-prompt" }), handlers);
      const { runner, errors } = createSdkPromptRunner(handlers);
      const base = normalizeBuildSystemPromptOptions({ cwd: process.cwd(), selectedTools: ["read"], appendSystemPrompt: "KEEP USER ADDENDUM" });
      const original = structuredClone(base);
      const before = buildSystemPromptSections(base);
      const result = await runner.emitBeforeAgentStart(prompt, undefined, base);
      const after = buildSystemPromptSections(result.systemPromptOptions);
      assert.equal(result.systemPromptOptions.customPrompt, undefined);
      assert.equal(after.preamble, before.preamble);
      assert.equal(after.addendum, before.addendum);
      if (grounding) {
        assert.match(after.grounding_runtime_safety, /Runtime grounding safety is active/);
        assert.match(after.grounding_runtime_safety, /Human review is mandatory for every record/);
      } else {
        assert.equal(buildSystemPrompt(result.systemPromptOptions), buildSystemPrompt(base));
        assert.equal(after.grounding_runtime_safety, undefined);
      }
      assert.deepEqual(base, original);
      assert.deepEqual(errors, []);
    });
  }
});

test("SDK user custom and forced prompts retain the existing grounding safety fallback", async (t) => {
  for (const choice of [{ customPrompt: "USER SPECIALIZED ROLE" }, { customPrompt: "" }, { forceSystemPrompt: "USER EXACT PROMPT" }]) {
    await t.test(JSON.stringify(choice), async () => {
      const handlers = new Map();
      installExtension(createGroundingSafetyExtension({ cwd: process.cwd(), sessionId: "custom-batch-prompt" }), handlers);
      const { runner, errors } = createSdkPromptRunner(handlers);
      const base = normalizeBuildSystemPromptOptions({ cwd: process.cwd(), ...choice, appendSystemPrompt: "KEEP CUSTOM ADDENDUM" });
      const original = structuredClone(base);
      const result = await runner.emitBeforeAgentStart("批处理 2 条 queries.json", undefined, base);
      const options = result.systemPromptOptions;
      assert.equal(options.customPrompt, choice.customPrompt);
      assert.equal(options.forceSystemPrompt, choice.forceSystemPrompt);
      assert.match(options.sections.grounding_runtime_safety, /Human review is mandatory for every record/);
      assert.match(options.sections.grounding_runtime_safety, /at most one record/);
      assert.match(options.sections.grounding_runtime_safety, /"targetCount":2/);
      const rendered = buildSystemPrompt(options);
      if (choice.forceSystemPrompt !== undefined) {
        assert.equal(rendered, choice.forceSystemPrompt, "an explicit exact prompt must never be overridden");
      } else {
        assert.match(rendered, /KEEP CUSTOM ADDENDUM/);
        assert.match(rendered, /Human review is mandatory for every record/);
        if (choice.customPrompt) assert.ok(rendered.includes(choice.customPrompt));
      }
      assert.doesNotMatch(rendered, /visual grounding specialist/);
      assert.deepEqual(base, original);
      assert.deepEqual(errors, []);
    });
  }
});
for (const failure of ["missing", "corrupt"]) {
  test(`a ${failure} image can be repaired and retried without skipping its record`, async (t) => {
    const root = await mkdtemp(join(tmpdir(), "pi-grounding-load-retry-"));
    t.after(() => rm(root, { recursive: true, force: true }));
    const sourceDir = join(root, "source");
    await mkdir(sourceDir);
    const queryPath = join(sourceDir, "queries.json");
    const outputDir = join(root, "output");
    const imagePath = join(sourceDir, "fixture.png");
    await writeFile(queryPath, JSON.stringify({
      one: { visible: "fixture.png", query: "the synthetic object" },
      two: { visible: "fixture.png", query: "another synthetic object" },
    }));
    if (failure === "corrupt") await writeFile(imagePath, "not an image");
    const handlers = new Map();
    const tools = installExtension(createGroundingSafetyExtension({
      cwd: root, sessionId: `retry-${failure}`,
      imageArchives: { visible: join(root, "missing.zip") },
    }), handlers);
    await handlers.get("before_agent_start")({
      prompt: `批处理 2 条 ${queryPath}`, systemPromptOptions: { sections: {} },
    });
    await assert.rejects(
      tools.get("grounding_next_batch").execute("failed", { queryPath, outputDir }),
      /image|input|ENOENT|unsupported/i,
    );
    await assertNoSavedArtifacts(outputDir);
    await writeFile(imagePath, await sharp({
      create: { width: 64, height: 48, channels: 3, background: "navy" },
    }).png().toBuffer());
    const retry = await tools.get("grounding_next_batch").execute("retry", { queryPath, outputDir });
    assert.equal(retry.details.records[0].key, "one");
    assert.equal(retry.content.some((block) => block.type === "image"), true);
    await assert.rejects(
      tools.get("grounding_next_batch").execute("skip", { queryPath, outputDir }),
      /until the user approves/,
    );
    await assertNoSavedArtifacts(outputDir);
  });
}

test("sanitized query records keep only the three views and query", () => {
  const result = sanitizeGroundingQueryJson({
    one: {
      visible: "Images/visible/one.png",
      infrared: "Images/infrared/one.png",
      depth: "Images/depth/one.png",
      query: "the red object",
      bbox: [0.1, 0.2, 0.3, 0.4],
      confidence: 0.99,
    },
    invalid: { bbox: [0, 0, 1, 1] },
  });

  assert.deepEqual(result, {
    one: {
      visible: "Images/visible/one.png",
      infrared: "Images/infrared/one.png",
      depth: "Images/depth/one.png",
      query: "the red object",
    },
  });
});

test("grounding tool text redacts annotations and prior artifacts", () => {
  assert.equal(
    redactGroundingToolText('{"bbox":[0.1,0.2,0.3,0.4]}'),
    "[grounding safety] Annotation or evaluation content was redacted.",
  );
  assert.equal(
    redactGroundingToolText("_annotated\nImages\n"),
    "[grounding safety] Annotation or evaluation content was redacted.",
  );
  assert.equal(redactGroundingToolText("Images/visible/one.png\n"), "Images/visible/one.png\n");
});

test("runtime extension redirects query reads and blocks reference paths", async () => {
  const handlers = new Map();
  const extension = createGroundingSafetyExtension({
    cwd: "D:/grounding-test",
    sessionId: "test-session",
  });
  installExtension(extension, handlers);

  await handlers.get("before_agent_start")({
    prompt: "测试图像定位",
    systemPromptOptions: { sections: {} },
  });

  const blocked = await handlers.get("tool_call")({
    toolName: "read",
    input: { path: "D:/grounding-test/expected.json" },
  });
  assert.equal(blocked.block, true);

  const writeBlocked = await handlers.get("tool_call")({
    toolName: "write",
    input: { path: "D:/grounding-test/_annotated/out.jpg" },
  });
  assert.equal(writeBlocked.block, true);

  const shellBlocked = await handlers.get("tool_call")({
    toolName: "powershell",
    input: { command: "Get-Content 'queries.json'" },
  });
  assert.equal(shellBlocked.block, true);

  const annotationFileBlocked = await handlers.get("tool_call")({
    toolName: "read",
    input: { path: "D:/grounding-test/annotations.json" },
  });
  assert.equal(annotationFileBlocked.block, true);

  const trustedGroundingResult = await handlers.get("tool_result")({
    toolName: "grounding_view",
    content: [{ type: "text", text: "Final bbox remains source-normalized." }],
  });
  assert.equal(trustedGroundingResult, undefined);
  const untrustedAnnotationResult = await handlers.get("tool_result")({
    toolName: "read",
    content: [{ type: "text", text: '{"bbox":[0,0,1,1]}' }],
  });
  assert.match(untrustedAnnotationResult.content[0].text, /redacted/);
});

test("dataset-only first prompts activate runtime grounding safety", async () => {
  const handlers = new Map();
  const extension = createGroundingSafetyExtension({
    cwd: "D:/grounding-test",
    sessionId: "dataset-prompt",
  });
  installExtension(extension, handlers);

  const systemPromptOptions = normalizeBuildSystemPromptOptions({ cwd: "D:/grounding-test" });
  await handlers.get("before_agent_start")({
    prompt: "D:\\evidence\\queries.json，处理这些 visible/infrared/depth 图像",
    systemPromptOptions,
  });
  assert.match(systemPromptOptions.sections.grounding_runtime_safety, /Runtime grounding safety/);
  assert.match(systemPromptOptions.customPrompt, /visual grounding specialist/);
  assert.match(systemPromptOptions.sections.grounding_runtime_safety, /loads one record: limit=1/);
  assert.match(systemPromptOptions.sections.grounding_runtime_safety, /records:\[\] with remaining:0/);
});

test("only sanitized-query image paths remain readable", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-test-"));
  try {
    const queryPath = join(root, "queries.json");
    await writeFile(queryPath, JSON.stringify({
      one: {
        visible: "Images/visible/one.png",
        infrared: "Images/infrared/one.png",
        depth: "Images/depth/one.png",
        query: "the object",
        bbox: [0, 0, 1, 1],
      },
    }), "utf8");

    const handlers = new Map();
    const extension = createGroundingSafetyExtension({ cwd: root, sessionId: "allow-list" });
    installExtension(extension, handlers);
    await handlers.get("before_agent_start")({
      prompt: queryPath,
      systemPromptOptions: { sections: {} },
    });

    const queryInput = { path: queryPath };
    assert.equal(await handlers.get("tool_call")({ toolName: "read", input: queryInput }), undefined);
    assert.match(queryInput.path, /queries-sanitized\.json$/);

    const allowed = await handlers.get("tool_call")({
      toolName: "read",
      input: { path: join(root, "Images", "visible", "one.png") },
    });
    assert.equal(allowed, undefined);

    const unknown = await handlers.get("tool_call")({
      toolName: "read",
      input: { path: join(root, "notes.txt") },
    });
    assert.equal(unknown.block, true);

    const listing = await handlers.get("tool_call")({
      toolName: "ls",
      input: { path: root },
    });
    assert.equal(listing.block, true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("grounding tools stream missing source images directly from configured archives", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-archive-test-"));
  try {
    const sourceDir = join(root, "source");
    const outputDir = join(root, "output");
    await mkdir(sourceDir, { recursive: true });
    const image = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64",
    );
    const archive = new JSZip();
    archive.file("visible/one.png", image);
    const archivePath = join(root, "images.zip");
    await writeFile(archivePath, await archive.generateAsync({ type: "nodebuffer" }));
    const queryPath = join(sourceDir, "queries.json");
    await writeFile(queryPath, JSON.stringify({
      one: {
        visible: "Images/visible/one.png",
        query: "the object",
      },
    }), "utf8");

    const handlers = new Map();
    const tools = installExtension(createGroundingSafetyExtension({
      cwd: root,
      sessionId: "archive-reader",
      imageArchives: { visible: archivePath },
    }), handlers);
    await handlers.get("before_agent_start")({
      prompt: `${sourceDir} 测试一下这些图像定位的问题`,
      systemPromptOptions: { sections: {} },
    });
    const result = await tools.get("grounding_next_batch").execute("next-archive", {
      queryPath,
      outputDir,
      limit: 1,
    });
    assert.equal(result.content.some((item) => item.type === "image"), true);
    assert.match(result.content.find((item) => item.type === "text" && item.text.includes("dimensions")).text, /images\.zip::visible\/one\.png/);
    const view = await tools.get("grounding_view").execute("view-archive", {
      queryPath,
      key: "one",
      modality: "visible",
      bbox: [0.2, 0.2, 0.4, 0.4],
      zoom: 12,
      region: [0, 0, 1, 1],
      reason: "The target is tiny and needs one full-frame magnified review.",
    });
    assert.equal(view.content.some((item) => item.type === "image"), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("grounding batch tools page, resume, validate, and build official JSON", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-batch-test-"));
  try {
    const sourceDir = join(root, "source");
    const outputDir = join(root, "output");
    await mkdir(sourceDir, { recursive: true });
    const onePixelPng = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64",
    );
    for (const modality of ["visible", "infrared", "depth"]) {
      const directory = join(sourceDir, modality);
      await mkdir(directory, { recursive: true });
      await writeFile(join(directory, "one.png"), onePixelPng);
      await writeFile(join(directory, "two.png"), onePixelPng);
    }
    const queryPath = join(sourceDir, "queries.json");
    await writeFile(queryPath, JSON.stringify({
      one: {
        visible: "visible/one.png",
        infrared: "infrared/one.png",
        depth: "depth/one.png",
        query: "the first object",
        bbox: [0, 0, 1, 1],
      },
      two: {
        visible: "visible/two.png",
        infrared: "infrared/two.png",
        depth: "depth/two.png",
        query: "the second object",
        bbox: [0, 0, 1, 1],
      },
    }), "utf8");

    const handlers = new Map();
    const tools = installExtension(createGroundingSafetyExtension({
      cwd: root,
      sessionId: "batch-tools",
    }), handlers);
    await handlers.get("before_agent_start")({
      prompt: `批处理 2 条 ${queryPath}`,
      systemPromptOptions: { sections: {} },
    });

    const directReadBlocked = await handlers.get("tool_call")({
      toolName: "read",
      input: { path: queryPath },
    });
    assert.equal(directReadBlocked.block, true);

    const nextTool = tools.get("grounding_next_batch");
    const viewTool = tools.get("grounding_view");
    const evidenceTool = tools.get("grounding_evidence");
    const saveTool = tools.get("grounding_save_result");
    const saveAndNextTool = tools.get("grounding_save_and_next");
    const reviewContext = createReviewContext();
    const firstPage = await nextTool.execute("next-1", { queryPath, outputDir, limit: 1 });
    assert.deepEqual(JSON.parse(firstPage.content[0].text).records, [{
      key: "one",
      visible: "visible/one.png",
      infrared: "infrared/one.png",
      depth: "depth/one.png",
      query: "the first object",
    }]);
    assert.equal(firstPage.content.some((item) => item.type === "image"), true);
    await assert.rejects(
      nextTool.execute("next-duplicate", { queryPath, outputDir, limit: 1 }),
      /until the user approves the current one/,
    );

    const nudge = await handlers.get("agent_before_settle")({
      outcome: "completed",
      context: { canContinue: false },
    }, { isIdle: () => true });
    assert.equal(nudge.continue, true);
    assert.match(nudge.entries[0].content, /still unsaved/);
    assert.equal(await handlers.get("agent_before_settle")({
      outcome: "completed",
      context: { canContinue: false },
    }, { isIdle: () => true }), undefined);

    const visibleAllowed = await handlers.get("tool_call")({
      toolName: "read",
      input: { path: join(sourceDir, "visible", "one.png") },
    });
    assert.equal(visibleAllowed, undefined);
    const infraredBlocked = await handlers.get("tool_call")({
      toolName: "read",
      input: { path: join(sourceDir, "infrared", "one.png") },
    });
    assert.equal(infraredBlocked.block, true);

    const infrared = await viewTool.execute("infrared-1", {
      queryPath,
      key: "one",
      modality: "infrared",
      bbox: [0.2, 0.2, 0.4, 0.4],
      reason: "visible objects overlap",
    });
    assert.equal(infrared.content.some((item) => item.type === "image"), true);
    assert.equal(await handlers.get("tool_call")({
      toolName: "read",
      input: { path: join(sourceDir, "infrared", "one.png") },
    }), undefined);
    const depthBlocked = await handlers.get("tool_call")({
      toolName: "read",
      input: { path: join(sourceDir, "depth", "one.png") },
    });
    assert.equal(depthBlocked.block, true);

    const crop = await viewTool.execute("crop-1", {
      queryPath,
      key: "one",
      modality: "visible",
      bbox: [0.3, 0.3, 0.7, 0.7],
      zoom: 12,
      region: [0.2, 0.1, 0.8, 0.9],
      reason: "target is only one pixel",
    });
    assert.equal(crop.content.some((item) => item.type === "image"), true);

    const cropPayload = JSON.parse(crop.content[1].text);
    assert.deepEqual(cropPayload.cropPixels, [0, 0, 1, 1]);
    assert.deepEqual(cropPayload.displayedNormalized, [0, 0, 1, 1]);
    assert.deepEqual(cropPayload.displayedSizePixels, [12, 12]);
    assert.equal(cropPayload.magnification, 12);
    assert.deepEqual(cropPayload.gridStep, { x: 0.2, y: 0.2 });
    assert.equal(cropPayload.touchesSourceEdge, true);
    assert.match(cropPayload.coordinateNote, /full-image normalized coordinates/);
    const cropImage = crop.content.find((item) => item.type === "image");
    assert.equal(cropImage.mimeType, "image/png");
    const cropPng = Buffer.from(cropImage.data, "base64");
    assert.equal(cropPng.subarray(1, 4).toString("latin1"), "PNG");
    const magnified = await sharp(cropPng).raw().toBuffer({ resolveWithObject: true });
    assert.deepEqual([magnified.info.width, magnified.info.height], [12, 12]);
    // The 1x1 source fixture is pure black, so cyan pixels can only come from the
    // measurement grid and its labels.
    let gridPixels = 0;
    for (let index = 0; index < magnified.data.length; index += magnified.info.channels) {
      const red = magnified.data[index];
      const green = magnified.data[index + 1];
      const blue = magnified.data[index + 2];
      if (blue - red > 60 && green - red > 50) gridPixels += 1;
    }
    assert.equal(gridPixels > 0, true);
    await evidenceTool.execute("evidence-1", {
      state: {
        target: "the first object",
        facts: ["The one-pixel fixture still needs one lower-scale rendering for coordinate regression coverage."],
        openQuestions: ["Does a lower requested zoom preserve the exact source mapping?"],
      },
    });
    const secondCrop = await viewTool.execute("crop-2", {
      queryPath,
      key: "one",
      modality: "visible",
      bbox: [0.3, 0.3, 0.7, 0.7],
      zoom: 4,
      region: [0, 0, 1, 1],
      reason: "checking another model-selected magnification",
    });
    const secondPayload = JSON.parse(secondCrop.content[1].text);
    assert.equal(secondPayload.requestedZoom, 4);
    assert.equal(secondPayload.zoomSource, "model");
    assert.deepEqual(secondPayload.displayedSizePixels, [4, 4]);

    await assert.rejects(
      saveTool.execute("bad-box", {
        queryPath,
        outputDir,
        key: "one",
        bbox: [0.8, 0.2, 0.3, 0.5],
        status: "ok",
        confidence: 0.8,
      }),
      /x1 < x2/,
    );

    const savedAndNext = await saveAndNextTool.execute("save-next-1", {
      queryPath,
      outputDir,
      key: "one",
      bbox: [0.1, 0.2, 0.3, 0.5],
      status: "low_confidence",
      confidence: 0.4,
      reason: "Visible edges bound the selected synthetic target in this crop.",
      coordinateSpace: "last_crop",
    }, undefined, undefined, reviewContext);
    assert.equal(JSON.parse(savedAndNext.content[0].text).next.records[0].key, "two");
    assert.equal(savedAndNext.content.some((item) => item.type === "image"), true);
    assert.equal(JSON.parse(savedAndNext.content[0].text).saved.archivePath, undefined);

    await viewTool.execute("focus-2", {
      queryPath,
      key: "two",
      modality: "visible",
      bbox: [0.2, 0.2, 0.4, 0.5],
      zoom: 12,
      region: [0, 0, 1, 1],
      reason: "Inspecting the next candidate at readable zoom before saving.",
    });
    const finalSave = await saveAndNextTool.execute("save-2", {
      queryPath,
      outputDir,
      key: "two",
      bbox: [0.2, 0.2, 0.4, 0.5],
      status: "ok",
      confidence: 0.8,
      reason: "Visible edges bound the second synthetic target.",
    }, undefined, undefined, reviewContext);
    assert.equal(finalSave.terminate, true);
    assert.equal(JSON.parse(finalSave.content[0].text).requestedLimitReached, true);
    assert.equal(JSON.parse(finalSave.content[0].text).next, null);
    const finalImages = finalSave.content.filter((item) => item.type === "image");
    assert.equal(finalImages.length, 1);
    assert.equal(finalImages[0].mimeType, "image/png");
    const finalOverlay = await sharp(Buffer.from(finalImages[0].data, "base64")).raw().toBuffer({ resolveWithObject: true });
    assert.deepEqual([finalOverlay.info.width, finalOverlay.info.height], [12, 12]);
    let finalBoxPixels = 0;
    for (let index = 0; index < finalOverlay.data.length; index += finalOverlay.info.channels) {
      const red = finalOverlay.data[index];
      const green = finalOverlay.data[index + 1];
      const blue = finalOverlay.data[index + 2];
      if (red - green > 60 && red - blue > 40) finalBoxPixels += 1;
    }
    assert.equal(finalBoxPixels > 0, true);

    const progressLines = (await readFile(join(outputDir, "progress.jsonl"), "utf8")).trim().split(/\r?\n/u);
    assert.equal(progressLines.length, 2);
    assert.equal(JSON.parse(progressLines[0]).status, "low_confidence");

    const submission = JSON.parse(await readFile(join(outputDir, "queries.json"), "utf8"));
    assert.deepEqual(Object.keys(submission), ["one", "two"]);
    assert.deepEqual(Object.keys(submission.one), ["visible", "infrared", "depth", "query", "bbox"]);
    assert.deepEqual(submission.one.bbox, [0.1, 0.2, 0.3, 0.5]);

    const archive = await JSZip.loadAsync(await readFile(join(outputDir, "queries.zip")));
    assert.deepEqual(Object.keys(archive.files), ["queries.json"]);
    assert.equal(
      await archive.file("queries.json").async("string"),
      await readFile(join(outputDir, "queries.json"), "utf8"),
    );
    assert.equal(JSON.parse(finalSave.content[0].text).saved.archivePath, join(outputDir, "queries.zip"));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("model may request human review directly when no focus crop is needed", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-optional-crop-test-"));
  const previousMode = process.env.PI_WEB_GROUNDING_REVIEW_MODE;
  try {
    delete process.env.PI_WEB_GROUNDING_REVIEW_MODE;
    const sourceDir = join(root, "source");
    const outputDir = join(root, "output");
    await mkdir(sourceDir, { recursive: true });
    const image = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64",
    );
    await writeFile(join(sourceDir, "visible.png"), image);
    const queryPath = join(sourceDir, "queries.json");
    await writeFile(queryPath, JSON.stringify({
      one: { visible: "visible.png", query: "the clearly visible object" },
    }), "utf8");

    const handlers = new Map();
    const tools = installExtension(createGroundingSafetyExtension({ cwd: root, sessionId: "optional-crop" }), handlers);
    await handlers.get("before_agent_start")({
      prompt: `Locate the object in ${queryPath}`,
      systemPromptOptions: { sections: {} },
    });
    await tools.get("grounding_next_batch").execute("next-one", { queryPath, outputDir });
    const saved = await tools.get("grounding_save_result").execute("save-one", {
      queryPath,
      outputDir,
      key: "one",
      bbox: [0.2, 0.2, 0.8, 0.8],
      status: "ok",
      confidence: 0.9,
      reason: "Visible edges bound the selected synthetic target in the full image.",
    }, undefined, undefined, createReviewContext());

    assert.equal(saved.details.processed, 1);
    const progress = JSON.parse((await readFile(join(outputDir, "progress.jsonl"), "utf8")).trim());
    assert.equal(progress.reviewSource, "human");
    assert.match(progress.reason, /Browser verified/);
  } finally {
    if (previousMode === undefined) delete process.env.PI_WEB_GROUNDING_REVIEW_MODE;
    else process.env.PI_WEB_GROUNDING_REVIEW_MODE = previousMode;
    await rm(root, { recursive: true, force: true });
  }
});

for (const stopReason of ["stop", "length"]) {
  test(`SDK assistant-tail ${stopReason} recovers once and still requires human approval`, async (t) => {
    const { outputDir, tools, handlers, params } = await createReviewFixture(t, 3);
    await handlers.get("before_agent_start")({ prompt: "只处理 1 条", systemPromptOptions: { sections: {} } });
    const sdk = createSdkSettleHarness(handlers.get("agent_before_settle"), { stopReason });
    assert.equal(sdk.session._buildBoundaryContext([], "agent_before_settle").canContinue, false);
    assert.equal(await sdk.settle(), true);
    assert.equal(sdk.previews[0].canContinue, false, "recover from the actual SDK assistant-tail boundary");
    assert.equal(sdk.session._buildBoundaryContext([], "agent_before_settle").canContinue, true);
    const messages = sdk.customMessages();
    assert.equal(messages.length, 1);
    assert.equal(messages[0].customType, "grounding-required-save");
    assert.match(messages[0].content, /Original query: "the synthetic object one"/);
    assert.match(messages[0].content, /human review.*unresolved.*do not invent certainty/);
    if (stopReason === "length") assert.match(messages[0].content, /output limit.*Keep this recovery brief/);
    else assert.doesNotMatch(messages[0].content, /output limit/);
    await assertNoSavedArtifacts(outputDir);

    sdk.appendAssistant();
    assert.equal(await sdk.settle(), false, "a second premature completion must not loop");
    assert.equal(sdk.customMessages().length, 1);
    const review = createDeferredReviewContext();
    const saving = tools.get("grounding_save_and_next").execute("recovered-save", params, undefined, undefined, review.context);
    const panel = await review.nextReview();
    assert.equal(await sdk.settle(), false, "a pending human review must remain a wait");
    await assertNoSavedArtifacts(outputDir);
    panel.input(confirmedReviewResponse(panel.details));
    const saved = await saving;
    assert.equal(saved.terminate, true);
    assert.equal(saved.details.requestedLimitReached, true);
    assert.equal(saved.details.saved.total, 3, "unprocessed records remain outside the requested count");
    assert.equal(saved.details.next, null);
    assert.equal(saved.details.job.targetCount, 1);
    sdk.appendAssistant();
    assert.equal(await sdk.settle(), false, "the requested count is a hard stop after approval");
    const progress = JSON.parse((await readFile(join(outputDir, "progress.jsonl"), "utf8")).trim());
    assert.equal(progress.reviewSource, "human");
    assert.deepEqual(sdk.errors, []);
  });
}

test("SDK settlement defers to queued user input without spending the recovery allowance", async (t) => {
  const { outputDir, handlers } = await createReviewFixture(t);
  const sdk = createSdkSettleHarness(handlers.get("agent_before_settle"));
  sdk.queue.push({ role: "user", content: "Stop and check the target identity first.", timestamp: 2 });
  assert.equal(await sdk.settle(), true, "the SDK handles its queued continuation");
  assert.equal(sdk.customMessages().length, 0);
  sdk.queue.length = 0;
  assert.equal(await sdk.settle(), true, "the skipped nudge remains available after user input is drained");
  assert.equal(sdk.customMessages().length, 1);
  await assertNoSavedArtifacts(outputDir);
  assert.deepEqual(sdk.errors, []);
});

test("SDK settlement preserves earlier extension drafts and does not duplicate their continuation", async (t) => {
  for (const continuing of [false, true]) {
    await t.test(`prior continuation ${continuing}`, async (subtest) => {
      const { handlers } = await createReviewFixture(subtest);
      const earlierDraft = { type: "custom_message", customType: "earlier-evidence", display: false, content: "Retain the user's target constraints." };
      const sdk = createSdkSettleHarness(handlers.get("agent_before_settle"), {
        earlierHandlers: [() => ({ entries: [earlierDraft], continue: continuing })],
      });
      assert.equal(await sdk.settle(), true);
      assert.deepEqual(sdk.customMessages().map((entry) => entry.customType), continuing
        ? ["earlier-evidence"] : ["earlier-evidence", "grounding-required-save"]);
      assert.deepEqual(sdk.errors, []);
    });
  }
});

test("SDK abort and provider-error boundaries never trigger grounding recovery", async (t) => {
  for (const outcome of ["aborted", "error"]) {
    await t.test(outcome, async (subtest) => {
      const { outputDir, handlers } = await createReviewFixture(subtest);
      const sdk = createSdkSettleHarness(handlers.get("agent_before_settle"), { outcome, stopReason: outcome });
      assert.equal(await sdk.settle(), false);
      assert.equal(sdk.customMessages().length, 0);
      // Even an inconsistent completed outcome must not retry an error/abort tail.
      sdk.session._lastActivityOutcome = "completed";
      assert.equal(await sdk.settle(), false);
      await assertNoSavedArtifacts(outputDir);
      assert.deepEqual(sdk.errors, []);
    });
  }
});

test("SDK startup recovery runs only once with assistant-tail context", async () => {
  const handlers = new Map();
  installExtension(createGroundingSafetyExtension({ cwd: process.cwd(), sessionId: "startup-recovery" }), handlers);
  await handlers.get("before_agent_start")({ prompt: "批处理 2 条 queries.json", systemPromptOptions: { sections: {} } });
  const sdk = createSdkSettleHarness(handlers.get("agent_before_settle"), { stopReason: "length" });
  assert.equal(await sdk.settle(), true);
  assert.equal(sdk.customMessages()[0].customType, "grounding-required-start");
  assert.match(sdk.customMessages()[0].content, /output limit.*grounding_next_batch exactly once/);
  sdk.appendAssistant();
  assert.equal(await sdk.settle(), false);
  assert.equal(sdk.customMessages().length, 1);
  assert.deepEqual(sdk.errors, []);
});

test("restored pending jobs never recover annotation on a new explanation, status, or stop turn", async (t) => {
  const f = await createReviewFixture(t, 2);
  assert.equal(f.entries.at(-1).data.pending.key, "one");
  assert.equal(f.entries.at(-1).data.targetCount, 2);
  const prompts = [
    "Only explain the prior grounding mistake. Do not annotate or resume the dataset.",
    "Explain why the last box was wrong.",
    "Continue explaining the prior grounding mistake.",
    "Show grounding_status and current progress only.",
    "Explain the previous 1 record without annotating.",
    "Only inspect the previous 1 record.",
    "Continue editing the TypeScript file.",
    "Fix a typo in the README.",
    "Pause the dataset run.",
    "Stop.",
    "只分析之前框错的原因，不要标注或继续数据集。",
    "继续解释上次定位的问题。",
    "查看当前进度，不要恢复标注。",
    "先暂停标注。",
    "停止",
    "先不要继续。",
    "这里只讨论 queries.json 的内容。",
    "不做标注，只查看以前的结果。",
  ];
  for (const prompt of prompts) {
    await t.test(prompt, async () => {
      const handlers = new Map();
      const tools = installExtension(createGroundingSafetyExtension({ cwd: f.root, sessionId: "restored-readonly" }), handlers);
      await handlers.get("session_start")({}, { sessionManager: { getEntries: () => f.entries } });
      const { runner, errors } = createSdkPromptRunner(handlers, f.root);
      // A prior annotation request in this same wrapper must not leak permission
      // into the new turn either; the persisted pending job remains unchanged.
      const base = normalizeBuildSystemPromptOptions({ cwd: f.root });
      await runner.emitBeforeAgentStart("Resume annotating the dataset.", undefined, base);
      await runner.emitBeforeAgentStart(prompt, undefined, base);
      const status = await tools.get("grounding_status").execute("inspect-pending", {});
      assert.equal(status.details.job.currentKey, "one");
      assert.equal(status.details.job.targetCount, 2, "an informational count must not change the requested job scope");
      for (const stopReason of ["stop", "length"]) {
        const sdk = createSdkSettleHarness(handlers.get("agent_before_settle"), { stopReason, userPrompt: prompt });
        assert.equal(sdk.session._buildBoundaryContext([], "agent_before_settle").canContinue, false);
        assert.equal(await sdk.settle(), false, `${stopReason} on an informational/stopped turn must stay settled`);
        assert.deepEqual(sdk.customMessages(), []);
        assert.deepEqual(sdk.errors, []);
      }
      await assertNoSavedArtifacts(f.outputDir);
      assert.deepEqual(errors, []);
    });
  }
});

test("an explicitly resumed pending job retains bounded output-limit recovery and its requested count", async (t) => {
  const f = await createReviewFixture(t, 2);
  for (const prompt of [
    "Resume the dataset annotation.",
    "继续处理当前记录。",
    "Next record please.",
    "这个框不对。请重新核对目标身份和所属对象。",
    "先查状态，然后继续标注。",
    "Check the status, then continue annotating.",
    "Stop overthinking and continue annotation.",
    "Please don't stop until 2 records are done.",
    "请根据图像定位目标，逐条审核。",
    "继续",
    "Continue",
  ]) {
    await t.test(prompt, async () => {
      const handlers = new Map();
      const tools = installExtension(createGroundingSafetyExtension({ cwd: f.root, sessionId: "restored-resume" }), handlers);
      await handlers.get("session_start")({}, { sessionManager: { getEntries: () => f.entries } });
      const { runner, errors } = createSdkPromptRunner(handlers, f.root);
      const base = normalizeBuildSystemPromptOptions({ cwd: f.root });
      await runner.emitBeforeAgentStart("只查看当前进度，不要继续标注。", undefined, base);
      await runner.emitBeforeAgentStart(prompt, undefined, base);
      const sdk = createSdkSettleHarness(handlers.get("agent_before_settle"), { stopReason: "length", userPrompt: prompt });
      assert.equal(await sdk.settle(), true, "the latest explicit work request authorizes startup recovery");
      assert.ok(["grounding-required-start", "grounding-required-next", "grounding-required-restore"].includes(sdk.customMessages()[0].customType));
      sdk.appendAssistant();
      assert.equal(await sdk.settle(), false, "startup recovery remains one-shot");
      const loaded = await tools.get("grounding_next_batch").execute("resume-pending", { queryPath: f.params.queryPath, outputDir: f.outputDir });
      assert.equal(loaded.details.records[0].key, "one", "restore the pending key, never skip to another record");
      assert.equal(loaded.details.job.targetCount, 2);
      sdk.appendAssistant();
      assert.equal(await sdk.settle(), true);
      assert.equal(sdk.customMessages().at(-1).customType, "grounding-required-save");
      sdk.appendAssistant();
      assert.equal(await sdk.settle(), false, "loaded-record recovery remains one-shot");
      assert.equal(sdk.customMessages().length, 2);
      await assertNoSavedArtifacts(f.outputDir);
      assert.deepEqual(errors, []);
      assert.deepEqual(sdk.errors, []);
    });
  }
});

test("a loaded record's unused recovery allowance does not override a later status or cancellation turn", async (t) => {
  const f = await createReviewFixture(t, 2);
  const { runner } = createSdkPromptRunner(f.handlers, f.root);
  const base = normalizeBuildSystemPromptOptions({ cwd: f.root });
  for (const prompt of ["Only show the current progress.", "暂停标注。", "Continue explaining this box, not annotating it."]) {
    await runner.emitBeforeAgentStart(prompt, undefined, base);
    const sdk = createSdkSettleHarness(f.handlers.get("agent_before_settle"), { stopReason: "length", userPrompt: prompt });
    assert.equal(await sdk.settle(), false);
    assert.deepEqual(sdk.customMessages(), []);
  }
  await runner.emitBeforeAgentStart("继续标注当前目标。", undefined, base);
  const resumed = createSdkSettleHarness(f.handlers.get("agent_before_settle"), { stopReason: "length" });
  assert.equal(await resumed.settle(), true, "read-only turns do not spend the per-record allowance");
  assert.equal(resumed.customMessages()[0].customType, "grounding-required-save");
  await assertNoSavedArtifacts(f.outputDir);
});

test("an unfinished requested batch keeps the turn open until the record is saved", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-settle-test-"));
  try {
    const sourceDir = join(root, "source");
    const outputDir = join(root, "output");
    await mkdir(join(sourceDir, "visible"), { recursive: true });
    const image = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64",
    );
    await writeFile(join(sourceDir, "visible", "one.png"), image);
    await writeFile(join(sourceDir, "visible", "two.png"), image);
    const queryPath = join(sourceDir, "queries.json");
    await writeFile(queryPath, JSON.stringify({
      one: {
        visible: "visible/one.png",
        query: "the object",
        bbox: [0, 0, 1, 1],
      },
      two: {
        visible: "visible/two.png",
        query: "the other object",
        bbox: [0, 0, 1, 1],
      },
    }), "utf8");

    const handlers = new Map();
    const tools = new Map();
    installExtension(createGroundingSafetyExtension({
      cwd: root,
      sessionId: "settle-guard",
    }), handlers, tools);
    await handlers.get("before_agent_start")({
      prompt: `批处理 2 条 ${queryPath}`,
      systemPromptOptions: { sections: {} },
    });

    const ctx = { ui: createReviewContext().ui, isIdle: () => true };
    await tools.get("grounding_next_batch").execute(
      "next-one",
      { queryPath, outputDir, limit: 1 },
      undefined,
      undefined,
      ctx,
    );
    await tools.get("grounding_view").execute("focus-one", {
      queryPath,
      key: "one",
      modality: "visible",
      bbox: [0.1, 0.1, 0.9, 0.9],
      zoom: 12,
      region: [0, 0, 1, 1],
      reason: "Inspecting the full frame at readable zoom before saving.",
    });
    const unsaved = await handlers.get("agent_before_settle")({
      outcome: "completed",
      context: { canContinue: false },
    });
    assert.equal(unsaved.continue, true);
    assert.match(unsaved.entries[0].content, /record one is still unsaved/);

    await tools.get("grounding_save_result").execute("save-one", {
      queryPath,
      outputDir,
      key: "one",
      bbox: [0.1, 0.1, 0.9, 0.9],
      status: "low_confidence",
      confidence: 0.4,
      reason: "Visible edges weakly support the selected synthetic target.",
    }, undefined, undefined, ctx);
    const continueBatch = await handlers.get("agent_before_settle")({
      outcome: "completed",
      context: { canContinue: false },
    });
    assert.equal(continueBatch.continue, true);
    assert.match(continueBatch.entries[0].content, /Only 1 of the requested 2/);
    assert.equal(await handlers.get("agent_before_settle")({
      outcome: "completed", context: { canContinue: false },
    }), undefined, "the next-record recovery is also bounded");

    const next = await tools.get("grounding_next_batch").execute(
      "next-two",
      { queryPath, outputDir, limit: 1 },
    );
    assert.equal(JSON.parse(next.content[0].text).records[0].key, "two");
    assert.equal(next.terminate, false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("an already-exhausted batch does not inject a startup retry", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-exhausted-start-test-"));
  try {
    const sourceDir = join(root, "source");
    const outputDir = join(root, "output");
    await mkdir(sourceDir, { recursive: true });
    const image = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64",
    );
    await writeFile(join(sourceDir, "visible.png"), image);
    const queryPath = join(sourceDir, "queries.json");
    await writeFile(queryPath, JSON.stringify({
      one: { visible: "visible.png", query: "the object" },
    }), "utf8");
    await mkdir(outputDir, { recursive: true });
    await writeFile(join(outputDir, "progress.jsonl"), JSON.stringify({
      key: "one",
      status: "ok",
      confidence: 0.9,
      bbox: [0.2, 0.2, 0.8, 0.8],
      reviewed: true,
      targetFound: true,
      candidateCount: 1,
      reason: "synthetic completed record",
      reviewSource: "human",
    }) + "\n", "utf8");

    const handlers = new Map();
    const tools = new Map();
    installExtension(createGroundingSafetyExtension({ cwd: root, sessionId: "exhausted-start" }), handlers, tools);
    await handlers.get("before_agent_start")({
      prompt: `处理 1 条 ${queryPath}`,
      systemPromptOptions: { sections: {} },
    });
    const exhausted = await tools.get("grounding_next_batch").execute(
      "next-exhausted-start",
      { queryPath, outputDir, limit: 1 },
    );
    assert.equal(exhausted.terminate, true);
    assert.equal(await handlers.get("agent_before_settle")({
      outcome: "completed",
      context: { canContinue: false },
    }), undefined);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});


test("medium focus crops are enlarged instead of rounded down to 1x", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-medium-crop-test-"));
  try {
    const sourceDir = join(root, "source");
    const outputDir = join(root, "output");
    await mkdir(sourceDir, { recursive: true });
    await mkdir(outputDir, { recursive: true });
    const sourceImage = await sharp({
      create: { width: 1000, height: 500, channels: 4, background: { r: 20, g: 30, b: 40, alpha: 1 } },
    }).png().toBuffer();
    await writeFile(join(sourceDir, "one.png"), sourceImage);
    const queryPath = join(sourceDir, "queries.json");
    await writeFile(queryPath, JSON.stringify({ one: { visible: "one.png", query: "the object" } }), "utf8");

    const handlers = new Map();
    const tools = installExtension(createGroundingSafetyExtension({
      cwd: root,
      sessionId: "medium-focus-crop",
    }), handlers);
    await handlers.get("before_agent_start")({
      prompt: `批处理 1 条 ${queryPath}`,
      systemPromptOptions: { sections: {} },
    });
    const first = await tools.get("grounding_next_batch").execute("next-1", { queryPath, outputDir, limit: 1 });
    const firstPayload = JSON.parse(first.content.find((item) => item.type === "text" && item.text.includes('"dimensions"')).text);
    assert.equal(firstPayload.currentBbox, null);
    assert.equal(firstPayload.overlay, "current_hypothesis_none");
    const outside = await tools.get("grounding_view").execute("outside-crop", {
        queryPath, key: "one", modality: "visible",
        bbox: [0.8, 0.2, 0.9, 0.3],
        zoom: 2,
        region: [0, 0, 0.577, 0.476],
        reason: "checking the wrong crop boundary",
      });
    const outsidePayload = JSON.parse(outside.content[1].text);
    assert.equal(outsidePayload.containsCurrentBbox, false);
    assert.equal(outsidePayload.currentBboxInCrop, null);
    assert.equal(outsidePayload.overlay, "current_hypothesis_outside_crop");
    const crop = await tools.get("grounding_view").execute("crop-1", {
      queryPath,
      key: "one",
      modality: "visible",
      bbox: [0.2, 0.2, 0.3, 0.3],
      zoom: 2,
      region: [0, 0, 0.577, 0.476],
      reason: "the target needs a readable focus crop",
    });
    const payload = JSON.parse(crop.content[1].text);
    assert.deepEqual(payload.cropPixels, [0, 0, 577, 238]);
    assert.deepEqual(payload.cropSizePixels, [577, 238]);
    assert.deepEqual(payload.displayedSizePixels, [1154, 476]);
    assert.equal(payload.magnification, 2);
    assert.deepEqual(payload.currentBbox, [0.2, 0.2, 0.3, 0.3]);
    assert.deepEqual(payload.currentBboxInCrop.map((value) => Number(value.toFixed(3))), [0.347, 0.42, 0.52, 0.63]);
    const image = crop.content.find((item) => item.type === "image");
    const decoded = await sharp(Buffer.from(image.data, "base64")).raw().toBuffer({ resolveWithObject: true });
    assert.deepEqual([decoded.info.width, decoded.info.height], [1154, 476]);
    const orangeNear = (centerX, centerY) => {
      for (let y = centerY - 4; y <= centerY + 4; y += 1) {
        for (let x = centerX - 4; x <= centerX + 4; x += 1) {
          const offset = (y * decoded.info.width + x) * decoded.info.channels;
          const [red, green, blue] = decoded.data.subarray(offset, offset + 3);
          if (red > 180 && green > 90 && green < 210 && blue < 70) return true;
        }
      }
      return false;
    };
    assert.equal(orangeNear(400, 200), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("small modality frames are magnified and unavailable views report a usable error", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-magnify-test-"));
  try {
    const sourceDir = join(root, "source");
    const outputDir = join(root, "output");
    await mkdir(join(sourceDir, "visible"), { recursive: true });
    await mkdir(outputDir, { recursive: true });
    const onePixelPng = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64",
    );
    await writeFile(join(sourceDir, "visible", "one.png"), onePixelPng);

    // The infrared archive exists but does not contain the requested entry.
    const archive = new JSZip();
    archive.file("visible/one.png", onePixelPng);
    const archivePath = join(root, "infrared.zip");
    await writeFile(archivePath, await archive.generateAsync({ type: "nodebuffer" }));

    const queryPath = join(sourceDir, "queries.json");
    await writeFile(queryPath, JSON.stringify({
      one: {
        visible: "visible/one.png",
        infrared: "infrared/one.png",
        query: "the small object",
      },
    }), "utf8");

    const handlers = new Map();
    const tools = installExtension(createGroundingSafetyExtension({
      cwd: root,
      sessionId: "magnify-views",
      imageArchives: { infrared: archivePath },
    }), handlers);
    await handlers.get("before_agent_start")({
      prompt: `批处理 1 条 ${queryPath}`,
      systemPromptOptions: { sections: {} },
    });

    const first = await tools.get("grounding_next_batch").execute("next-1", { queryPath, outputDir, limit: 1 });
    const visiblePayload = JSON.parse(
      first.content.find((item) => item.type === "text" && item.text.includes("dimensions")).text,
    );
    assert.equal(visiblePayload.magnification, 12);
    assert.deepEqual(visiblePayload.displayedSizePixels, [12, 12]);
    assert.match(visiblePayload.dimensions, /1x1 source pixels; displayed as 12x12/);
    const visibleImage = first.content.find((item) => item.type === "image");
    assert.equal(visibleImage.mimeType, "image/png");
    const magnified = await sharp(Buffer.from(visibleImage.data, "base64")).raw().toBuffer({ resolveWithObject: true });
    assert.deepEqual([magnified.info.width, magnified.info.height], [12, 12]);

    await assert.rejects(
      tools.get("grounding_view").execute("view-depth", {
        queryPath,
        key: "one",
        modality: "depth",
        bbox: [0.2, 0.2, 0.4, 0.4],
        reason: "checking whether a depth frame exists",
      }),
      /has no depth image[\s\S]*not evidence that the target is absent/,
    );

    await assert.rejects(
      tools.get("grounding_view").execute("view-infrared", {
        queryPath,
        key: "one",
        modality: "infrared",
        bbox: [0.2, 0.2, 0.4, 0.4],
        reason: "checking whether a thermal frame exists",
      }),
      /The infrared image could not be loaded as grounding evidence[\s\S]*not evidence that the target is absent/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
test("save-and-next waits for valid approval, preserves corrections, and reviews every record", async (t) => {
  const { outputDir, tools, handlers, params } = await createReviewFixture(t);
  const review = createDeferredReviewContext();
  const saveTool = tools.get("grounding_save_and_next");
  let finished = false;
  const saving = saveTool.execute("review-one", params, undefined, undefined, review.context).then((result) => {
    finished = true;
    return result;
  });
  const panel = await review.nextReview();
  assert.equal(panel.details.key, "one");
  assert.equal(panel.details.canContinue, true);
  assert.equal(panel.options.timeout, undefined);
  assert.equal(panel.details.rawBbox, undefined);
  assert.deepEqual(panel.details.bbox, params.bbox, "review must use the model's exact edges without automatic padding");
  const cleanImage = await sharp(Buffer.from(panel.details.image.data, "base64")).raw().toBuffer({ resolveWithObject: true });
  for (let index = 0; index < cleanImage.data.length; index += cleanImage.info.channels) {
    assert.deepEqual([...cleanImage.data.subarray(index, index + 3)], [20, 30, 40]);
  }

  const validResponse = confirmedReviewResponse(panel.details);
  for (const invalid of [
    "yes", "{}", { ...validResponse, type: "custom_ui_response" },
    { ...validResponse, bbox: [0.8, 0.2, 0.3, 0.7] },
    { ...validResponse, candidateCount: 0 },
    { ...validResponse, reason: "short" },
  ]) {
    panel.input(invalid);
    assert.equal(panel.completed, false);
  }
  await assertNoSavedArtifacts(outputDir);
  assert.equal(finished, false);
  assert.equal(review.opened.length, 1);
  await assert.rejects(
    tools.get("grounding_next_batch").execute("too-early", params),
    /until the user approves the current one/,
  );
  await assert.rejects(
    saveTool.execute("concurrent-save", params, undefined, undefined, review.context),
    /already awaiting human review or being saved/,
  );
  assert.equal(await handlers.get("agent_before_settle")({
    outcome: "completed", context: { canContinue: false },
  }), undefined);

  // This is still a tiny box: approval must not apply padding or rounding again.
  const approvedBbox = [0.401234567, 0.502345678, 0.411234567, 0.522345678];
  panel.input({ ...validResponse, bbox: approvedBbox, reason: "User corrected the exact synthetic target bounds." });
  const result = await saving;
  assert.equal(finished, true);
  assert.deepEqual(result.details.next.records.map((record) => record.key), ["two"]);
  const progress = (await readFile(join(outputDir, "progress.jsonl"), "utf8")).trim().split(/\r?\n/u).map(JSON.parse);
  assert.equal(progress.length, 1);
  assert.deepEqual(progress[0].bbox, approvedBbox);
  assert.equal(progress[0].reviewSource, "human");
  const submission = JSON.parse(await readFile(join(outputDir, "queries.json"), "utf8"));
  assert.deepEqual(Object.keys(submission), ["one"]);
  assert.deepEqual(submission.one.bbox, approvedBbox);
  await assert.rejects(readFile(join(outputDir, "queries.zip")), { code: "ENOENT" });

  let secondFinished = false;
  const savingSecond = saveTool.execute("review-two", { ...params, key: "two" }, undefined, undefined, review.context).then((value) => {
    secondFinished = true;
    return value;
  });
  const secondPanel = await review.nextReview();
  assert.equal(secondPanel.details.key, "two");
  assert.equal(secondPanel.details.canContinue, false);
  assert.equal(secondPanel.details.expectedOrdinal, 2);
  secondPanel.input({ ...confirmedReviewResponse(secondPanel.details), candidateRank: 1 });
  assert.equal(secondPanel.completed, false);
  assert.equal(secondFinished, false);
  assert.equal(review.opened.length, 2);
  assert.equal((await readFile(join(outputDir, "progress.jsonl"), "utf8")).trim().split(/\r?\n/u).length, 1);
  secondPanel.input(confirmedReviewResponse(secondPanel.details));
  const final = await savingSecond;
  assert.equal(final.details.next, null);
  assert.equal(final.terminate, true);
  const completed = (await readFile(join(outputDir, "progress.jsonl"), "utf8")).trim().split(/\r?\n/u).map(JSON.parse);
  assert.deepEqual(completed.map((entry) => entry.key), ["one", "two"]);
});

test("rejection keeps the current key unsaved until a fresh corrected review", async (t) => {
  const { outputDir, tools, handlers, params } = await createReviewFixture(t);
  const review = createDeferredReviewContext();
  const saveTool = tools.get("grounding_save_and_next");
  const rejected = assert.rejects(
    saveTool.execute("reject-one", params, undefined, undefined, review.context),
    /review rejected candidate: The selected synthetic object is incorrect.*same record/,
  );
  const panel = await review.nextReview();
  panel.input({
    type: "grounding_review_response", action: "reject",
    reason: "The selected synthetic object is incorrect",
  });
  await rejected;
  await assertNoSavedArtifacts(outputDir);
  const sdk = createSdkSettleHarness(handlers.get("agent_before_settle"), { stopReason: "length" });
  assert.equal(await sdk.settle(), false, "rejection must not cause an automatic resubmission");
  assert.equal(sdk.customMessages().length, 0);
  await assert.rejects(
    tools.get("grounding_next_batch").execute("skip-rejected", params),
    /until the user approves the current one/,
  );
  const saving = saveTool.execute("revise-one", { ...params, bbox: [0.3, 0.3, 0.7, 0.7] }, undefined, undefined, review.context);
  const revised = await review.nextReview();
  assert.equal(revised.details.key, "one");
  assert.deepEqual(revised.details.bbox, [0.3, 0.3, 0.7, 0.7]);
  revised.input(confirmedReviewResponse(revised.details));
  assert.deepEqual((await saving).details.next.records.map((record) => record.key), ["two"]);
  sdk.appendAssistant();
  assert.equal(await sdk.settle(), true, "a subsequent record may recover after the correction is explicitly approved");
  assert.match(sdk.customMessages()[0].content, /record two is still unsaved/);
  const progress = (await readFile(join(outputDir, "progress.jsonl"), "utf8")).trim().split(/\r?\n/u);
  assert.equal(progress.length, 1);
  assert.equal(JSON.parse(progress[0]).key, "one");
});

test("cancelling pending review writes nothing and suppresses settle retries", async (t) => {
  const { outputDir, tools, handlers, params } = await createReviewFixture(t);
  const review = createDeferredReviewContext();
  const controller = new AbortController();
  const cancelled = assert.rejects(
    tools.get("grounding_save_and_next").execute("cancel-one", params, controller.signal, undefined, review.context),
    { name: "AbortError" },
  );
  const panel = await review.nextReview();
  controller.abort();
  await cancelled;
  assert.equal(panel.completed, true);
  // A late browser response from the closed review cannot resurrect the save.
  panel.input(confirmedReviewResponse(panel.details));
  await assertNoSavedArtifacts(outputDir);
  assert.equal(review.opened.length, 1);
  assert.equal(await handlers.get("agent_before_settle")({
    outcome: "completed", context: { canContinue: false },
  }), undefined);
  await assert.rejects(
    tools.get("grounding_next_batch").execute("skip-cancelled", params),
    /until the user approves the current one/,
  );
});

test("aborting before review does not open a panel or schedule another save", async (t) => {
  const { outputDir, tools, handlers, params } = await createReviewFixture(t);
  const review = createDeferredReviewContext();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    tools.get("grounding_save_result").execute("pre-cancelled", params, controller.signal, undefined, review.context),
    { name: "AbortError" },
  );
  assert.equal(review.opened.length, 0);
  await assertNoSavedArtifacts(outputDir);
  assert.equal(await handlers.get("agent_before_settle")({
    outcome: "completed", context: { canContinue: false },
  }), undefined);
});

test("legacy auto and none settings cannot bypass human review", async (t) => {
  const previousMode = process.env.PI_WEB_GROUNDING_REVIEW_MODE;
  t.after(() => {
    if (previousMode === undefined) delete process.env.PI_WEB_GROUNDING_REVIEW_MODE;
    else process.env.PI_WEB_GROUNDING_REVIEW_MODE = previousMode;
  });
  for (const mode of ["auto", "none"]) {
    await t.test(mode, async (subtest) => {
      process.env.PI_WEB_GROUNDING_REVIEW_MODE = mode;
      const { outputDir, tools, params } = await createReviewFixture(subtest, 1);
      const review = createDeferredReviewContext();
      const saving = tools.get("grounding_save_result").execute("legacy-mode", params, undefined, undefined, review.context);
      const panel = await review.nextReview();
      assert.equal(panel.completed, false);
      assert.equal(panel.details.canContinue, false);
      await assertNoSavedArtifacts(outputDir);
      panel.input(confirmedReviewResponse(panel.details));
      await saving;
      const progress = JSON.parse((await readFile(join(outputDir, "progress.jsonl"), "utf8")).trim());
      assert.equal(progress.reviewSource, "human");
      assert.equal(progress.status, "ok");
      assert.equal(progress.confidence, 0.9);
    });
  }
});

test("saving cannot switch the output directory of the reviewed record", async (t) => {
  const { root, outputDir, tools, params } = await createReviewFixture(t);
  const otherOutputDir = join(root, "other-output");
  const review = createDeferredReviewContext();
  await assert.rejects(
    tools.get("grounding_save_result").execute(
      "change-output", { ...params, outputDir: otherOutputDir }, undefined, undefined, review.context,
    ),
    /same output directory that loaded this record/,
  );
  assert.equal(review.opened.length, 0);
  await assertNoSavedArtifacts(outputDir);
  await assertNoSavedArtifacts(otherOutputDir);
});

test("small saved boxes get an automatic magnified verification crop", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-verify-crop-test-"));
  try {
    const sourceDir = join(root, "source");
    const outputDir = join(root, "output");
    await mkdir(sourceDir, { recursive: true });
    await mkdir(outputDir, { recursive: true });
    const onePixelPng = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64",
    );
    await writeFile(join(sourceDir, "one.png"), onePixelPng);
    const queryPath = join(sourceDir, "queries.json");
    await writeFile(queryPath, JSON.stringify({
      one: { visible: "one.png", query: "the only object" },
    }), "utf8");

    const handlers = new Map();
    const tools = installExtension(createGroundingSafetyExtension({
      cwd: root,
      sessionId: "verify-crop",
    }), handlers);
    await handlers.get("before_agent_start")({
      prompt: `批处理 1 条 ${queryPath}`,
      systemPromptOptions: { sections: {} },
    });
    await tools.get("grounding_next_batch").execute("next-1", { queryPath, outputDir, limit: 1 });
    await tools.get("grounding_view").execute("focus-1", {
      queryPath,
      key: "one",
      modality: "visible",
      bbox: [0.4, 0.4, 0.5, 0.5],
      zoom: 12,
      region: [0, 0, 1, 1],
      reason: "Inspecting the tiny target at readable zoom before saving.",
    });

    // A box covering 1% of the frame is too small to trust from a full frame, so
    // the runtime must add its own magnified verification crop.
    const saved = await tools.get("grounding_save_result").execute("save-1", {
      queryPath,
      outputDir,
      key: "one",
      bbox: [0.4, 0.4, 0.5, 0.5],
      status: "ok",
      confidence: 0.9,
      reason: "Visible edges tightly bound the selected small synthetic target.",
    }, undefined, undefined, createReviewContext());

    const images = saved.content.filter((item) => item.type === "image");
    assert.equal(images.length, 2);
    const overlay = await sharp(Buffer.from(images[0].data, "base64")).raw().toBuffer({ resolveWithObject: true });
    const verification = await sharp(Buffer.from(images[1].data, "base64")).raw().toBuffer({ resolveWithObject: true });
    assert.deepEqual([overlay.info.width, overlay.info.height], [12, 12]);
    assert.deepEqual([verification.info.width, verification.info.height], [12, 12]);
    const countPixels = (image, predicate) => {
      let hits = 0;
      for (let index = 0; index < image.data.length; index += image.info.channels) {
        if (predicate(image.data, index)) hits += 1;
      }
      return hits;
    };
    const overlayRed = countPixels(overlay, (data, index) => data[index] - data[index + 1] > 60 && data[index] - data[index + 2] > 40);
    const verificationCyan = countPixels(verification, (data, index) => data[index + 2] - data[index] > 60 && data[index + 1] - data[index] > 50);
    assert.equal(overlayRed > 0, true);
    assert.equal(verificationCyan > 0, true);

    const verificationBlock = saved.content.find(
      (item) => item.type === "text" && item.text.includes("automaticMagnifiedVerification"),
    );
    const payload = JSON.parse(verificationBlock.text).automaticMagnifiedVerification;
    assert.deepEqual(payload.cropPixels, [0, 0, 1, 1]);
    assert.equal(payload.magnification, 12);
    assert.deepEqual(payload.gridStep, { x: 0.2, y: 0.2 });
    assert.match(payload.coordinateNote, /full-image normalized coordinates/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("ordinal hints never silently choose first from contradictory second-first wording", () => {
  assert.equal(expectedGroundingOrdinal("The second first alpaca from left to right"), undefined);
  assert.equal(expectedGroundingOrdinal("third drone from the left"), 3);
  assert.equal(expectedGroundingOrdinal("第2个，从右往左"), 2);
});

test("unresolved constraint review needs explicit human resolution before upgrading status or confidence", () => {
  const details = { kind: "grounding_review", key: "one", query: "unreadable character", bbox: [.2, .2, .4, .4],
    status: "unresolved", confidence: .49, targetFound: false, candidateCount: 0,
    reason: "Identity is unsupported in the original pixels.",
    constraintAssessment: { status: "unresolved", canLock: false, requiresHumanReview: true,
      issues: [{ code: "candidate_identity", message: "Character identity is unresolved." }], orders: [], limitations: [] } };
  const upgraded = { type: "grounding_review_response", action: "confirm", bbox: details.bbox,
    status: "ok", confidence: .82, targetFound: true, candidateCount: 1, candidateRank: 1,
    reason: "The human reviewer identifies and verifies the visible character." };
  assert.throws(() => validateGroundingReviewResponse(upgraded, details), /explicitly confirms/);
  assert.equal(validateGroundingReviewResponse({ ...upgraded, constraintsResolved: true }, details).status, "ok");
  assert.equal(validateGroundingReviewResponse({ ...upgraded, status: "unresolved", confidence: .3,
    targetFound: false, candidateCount: 0, candidateRank: undefined }, details).status, "unresolved");
});

test("browser review preserves validated long-term learning on confirmation and rejection", () => {
  const details = { kind: "grounding_review", key: "one", query: "the object", bbox: [.2, .2, .4, .4],
    status: "ok", confidence: .8, targetFound: true, candidateCount: 1,
    reason: "The object is visible.", image: { data: "", mimeType: "image/png", originalWidth: 10, originalHeight: 10 },
    availableModalities: ["visible"] };
  const learning = { category: "boundary", scope: "global",
    advice: "Check the complete outer silhouette before submitting a tight box." };
  const confirmed = validateGroundingReviewResponse({ type: "grounding_review_response", action: "confirm",
    bbox: details.bbox, status: "ok", confidence: .8, targetFound: true, candidateCount: 1, candidateRank: 1,
    reason: "The browser confirms the visible object.", learning }, details);
  assert.deepEqual(confirmed.learning, learning);
  const rejected = validateGroundingReviewResponse({ type: "grounding_review_response", action: "reject",
    reason: "The box misses the outer edge.", learning }, details);
  assert.deepEqual(rejected.learning, learning);
  assert.throws(() => validateGroundingReviewResponse({ type: "grounding_review_response", action: "reject",
    reason: "Wrong.", learning: { ...learning, scope: "unsafe" } }, details), /invalid scope/);
});
