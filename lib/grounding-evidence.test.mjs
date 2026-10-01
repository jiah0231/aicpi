import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";
const { compactGroundingEvidence, validateGroundingWorkingState } = await createJiti(import.meta.url).import("./grounding-evidence.ts");

function result(id, details, extra = {}) {
  return {
    role: "toolResult", toolName: "grounding_view", toolCallId: id,
    content: [{ type: "text", text: `Source coordinates for ${id}` }, { type: "image", mimeType: "image/png", data: id }],
    details, isError: false, timestamp: 1, ...extra,
  };
}
const options = { active: true, archivedViewIds: ["old"], pinnedViewIds: [] };

test("only explicitly archived images are removed, without modifying the transcript", () => {
  const messages = [result("old", { evidenceViewIds: ["old"] }), result("new", { evidenceViewIds: ["new"] })];
  const original = structuredClone(messages);
  const compacted = compactGroundingEvidence(messages, options);
  assert.deepEqual(compacted[0].content, [messages[0].content[0]]);
  assert.equal(compacted[0].details, messages[0].details);
  assert.equal(compacted[1], messages[1]);
  assert.deepEqual(messages, original);
});

test("pinning overrides archival even when a composite result also contains archived views", () => {
  const messages = [result("comparison", { evidenceViewIds: ["old", "pinned"] })];
  const compacted = compactGroundingEvidence(messages, {
    active: true, archivedViewIds: ["old", "pinned"], pinnedViewIds: ["pinned"],
  });
  assert.equal(compacted[0], messages[0]);
});

test("all views in a composite result must be explicitly archived", () => {
  const messages = [result("comparison", { evidenceViewIds: ["old", "new"] })];
  assert.equal(compactGroundingEvidence(messages, options)[0], messages[0]);
  const compacted = compactGroundingEvidence(messages, { ...options, archivedViewIds: ["old", "new"] });
  assert.equal(compacted[0].content.length, 1);
});

test("missing, malformed, empty, and unknown evidence metadata preserve image evidence", () => {
  const details = [undefined, null, [], {}, { evidenceViewIds: [] }, { evidenceViewIds: "old" },
    { evidenceViewIds: ["old", null] }, { evidenceViewIds: [""] }, { evidenceViewIds: ["unknown"] }];
  const messages = details.map((metadata, i) => result(String(i), metadata));
  compactGroundingEvidence(messages, options).forEach((message, i) => assert.equal(message, messages[i]));
});

test("unmarked record overview, user corrections, custom messages and signed thinking survive", () => {
  const messages = [
    result("overview", undefined, { toolName: "grounding_next_batch" }),
    { role: "user", content: [{ type: "text", text: "Wrong sculpture, choose the one on the left." }, { type: "image", data: "correction", mimeType: "image/png" }], timestamp: 2 },
    { role: "custom", customType: "grounding", content: "Human rejected the old identity.", display: true, timestamp: 3 },
    { role: "assistant", content: [{ type: "thinking", thinking: "prior reasoning", thinkingSignature: "provider-signature" }], timestamp: 4 },
    result("old", { evidenceViewIds: ["old"] }),
  ];
  const compacted = compactGroundingEvidence(messages, options);
  for (let i = 0; i < 4; i++) assert.equal(compacted[i], messages[i]);
});

for (const reverse of [false, true]) {
  test(`sibling calls and results retain identity/order (reversed results: ${reverse})`, () => {
    const calls = { role: "assistant", content: [
      { type: "toolCall", id: "old-call", name: "grounding_view", arguments: {} },
      { type: "toolCall", id: "other-call", name: "grounding_status", arguments: {} },
    ], timestamp: 0 };
    const results = [result("old-call", { evidenceViewIds: ["old"] }), result("other-call", undefined, { toolName: "grounding_status" })];
    const messages = [calls, ...(reverse ? results.reverse() : results)];
    const compacted = compactGroundingEvidence(messages, options);
    assert.equal(compacted.length, messages.length);
    assert.equal(compacted[0], calls);
    assert.deepEqual(compacted.slice(1).map((message) => message.toolCallId), messages.slice(1).map((message) => message.toolCallId));
  });
}

test("inactive contexts and unrelated tools are left intact", () => {
  const messages = [result("old", { evidenceViewIds: ["old"] }, { toolName: "image_reader" })];
  assert.equal(compactGroundingEvidence(messages, options)[0], messages[0]);
  assert.equal(compactGroundingEvidence(messages, { ...options, active: false }), messages);
  assert.equal(compactGroundingEvidence(messages, { ...options, archivedViewIds: [] }), messages);
});

test("image-only results remain nonempty and paired after explicit archival", () => {
  const messages = [result("old", { evidenceViewIds: ["old"] }, { content: [{ type: "image", data: "old", mimeType: "image/png" }] })];
  const compacted = compactGroundingEvidence(messages, options);
  assert.equal(compacted[0].toolCallId, "old");
  assert.equal(compacted[0].content[0].type, "text");
  assert.match(compacted[0].content[0].text, /archived.*old/);
});

test("omits older unpinned grounding images when the provider request budget is exceeded", () => {
  const messages = [
    result("overview", {}, { content: [{ type: "image", data: "o".repeat(300), mimeType: "image/png" }] }),
    result("old", { evidenceViewIds: ["old"] }, { content: [{ type: "image", data: "a".repeat(300), mimeType: "image/png" }] }),
    result("new", { evidenceViewIds: ["new"] }, { content: [{ type: "image", data: "b".repeat(300), mimeType: "image/png" }] }),
  ];

  const compacted = compactGroundingEvidence(messages, {
    active: true,
    pinnedViewIds: [],
    archivedViewIds: [],
    maxImageBase64Characters: 700,
  });

  assert.equal(compacted[0].content.some((block) => block.type === "image"), true, "unmarked overview remains");
  assert.equal(compacted[1].content.some((block) => block.type === "image"), false, "older view is omitted");
  assert.match(compacted[1].content.at(-1).text, /request-size limit/);
  assert.equal(compacted[2].content.some((block) => block.type === "image"), true, "latest view remains");
});

test("working state validates concise evidence without modifying input", () => {
  const input = { target: "  the central sculpture's beak ", facts: [" Largest black region is background. "],
    hypotheses: ["The lower dark wedge may be the beak."], openQuestions: ["Which sculpture is largest?"], ruledOut: [],
    selection: { status: "locked", bbox: [.4, .2, .5, .3], evidence: "The visible structure and requested relation identify this candidate." } };
  assert.deepEqual(validateGroundingWorkingState(input), {
    target: "the central sculpture's beak", facts: ["Largest black region is background."],
    hypotheses: ["The lower dark wedge may be the beak."], openQuestions: ["Which sculpture is largest?"], ruledOut: [],
    selection: { status: "locked", bbox: [.4, .2, .5, .3], evidence: "The visible structure and requested relation identify this candidate." },
  });
  assert.equal(input.facts[0], " Largest black region is background. ");
  assert.deepEqual(validateGroundingWorkingState({}), {});
  assert.deepEqual(validateGroundingWorkingState({ target: "", facts: [] }), { target: "", facts: [] });
  assert.equal(validateGroundingWorkingState({
    selection: { status: "selected", bbox: [.1, .1, .2, .2], evidence: "Legacy selected state remains compatible." },
  }).selection.status, "locked");
});

test("working state rejects invalid or unbounded payloads instead of silently truncating corrections", () => {
  for (const input of [null, [], "state", { privateReasoning: "unbounded" }, { target: 123 }, { target: "x".repeat(601) },
    { facts: "fact" }, { facts: Array(9).fill("fact") }, { facts: [null] }, { hypotheses: ["x".repeat(401)] },
    { facts: ["x".repeat(401)] }, { target: undefined }, { selection: null },
    { selection: { status: "selected", evidence: "Candidate is visible." } },
    { selection: { status: "unknown", evidence: "Candidate is visible." } },
    { selection: { status: "selected", bbox: [0, 0, 2, 1], evidence: "Candidate is visible." } },
    { selection: { status: "reconsidering", evidence: "short" } }]) {
    assert.throws(() => validateGroundingWorkingState(input), /Grounding working state|grounding working state/);
  }
});

test("working state carries exact original-query contracts without promoting legacy selections", () => {
  const query = "  The second first alpaca from left to right  ";
  const contract = {
    originalQuery: query,
    queryCoverage: { status: "unresolved", evidence: "The two ordinal words conflict and both remain in the original query." },
    interpretations: [], candidates: [],
  };
  const input = { target: "a tentative alpaca", contract, selection: { status: "reconsidering", evidence: "The original ordinal interpretation remains unresolved." } };
  assert.deepEqual(validateGroundingWorkingState(input, query), input);
  assert.equal(validateGroundingWorkingState(input).contract.originalQuery, query);
  assert.throws(() => validateGroundingWorkingState(input, "The second alpaca from left to right"), /exactly match/);
  assert.throws(() => validateGroundingWorkingState({ contract: null }), /[Gg]rounding contract/);
  assert.throws(() => validateGroundingWorkingState({ contract: { ...contract, queryCoverage: { status: "supported", evidence: "" } } }), /[Gg]rounding contract/);
  assert.deepEqual(validateGroundingWorkingState({ target: "Legacy notebook" }, query), { target: "Legacy notebook" });
});
