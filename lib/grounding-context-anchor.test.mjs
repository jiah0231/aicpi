import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";
const { withGroundingContextAnchor, isGroundingContextAnchor } = await createJiti(import.meta.url).import("./grounding-context-anchor.ts");
const snapshot = (messages) => JSON.parse(messages.find(isGroundingContextAnchor).content[0].text.split("\n").slice(1).join("\n"));

test("canonical ephemeral anchor preserves exact query, signed thinking and tool pairing without mutation", () => {
  const messages = [
    { role: "assistant", content: [{ type: "thinking", thinking: "opaque", thinkingSignature: "signed" }, { type: "toolCall", id: "a", name: "grounding_view", arguments: {} }] },
    { role: "toolResult", toolCallId: "a", toolName: "grounding_view", content: [{ type: "text", text: "pixels" }] },
    { role: "custom", customType: "other", content: "retain other extension", display: false, timestamp: 1 },
  ];
  const original = structuredClone(messages);
  const query = '  Third girl?\n“yellow”, both hands\nbehind the back!  ';
  const first = withGroundingContextAnchor(messages, { key: "one", originalQuery: query, state: { target: "wrong shortened target" } });
  const second = withGroundingContextAnchor(first, { key: "one", originalQuery: query });
  assert.equal(second.filter(isGroundingContextAnchor).length, 1);
  assert.equal(snapshot(second).originalQuery, query);
  assert.deepEqual(messages, original);
  assert.equal(second[0], messages[0]);
  assert.equal(second[1], messages[1]);
  assert.ok(isGroundingContextAnchor(second.at(-1)));
  const changed = withGroundingContextAnchor(second, { key: "two", originalQuery: "new query" });
  assert.equal(snapshot(changed).recordKey, "two");
  assert.equal(snapshot(changed).originalQuery, "new query");
  assert.deepEqual(withGroundingContextAnchor(changed), messages);
});

test("anchor preserves every declared candidate and condition status without repeating evidence prose", () => {
  const evidence = "UNIQUE_LONG_EVIDENCE_SENTINEL";
  const record = { key: "one", originalQuery: "exact question", awaitingClarification: "Which one?", state: {
    selection: { status: "reconsidering", evidence }, openQuestions: ["Is the posture actually visible?"],
    contract: { originalQuery: "exact question", queryCoverage: { status: "unresolved", evidence }, selectedCandidateId: "c",
      candidates: [{ id: "c", bbox: [.1,.2,.3,.4], identity: { label: "girl", status: "unresolved", basis: "unknown", evidence } }],
      interpretations: [{ id: "reading", status: "unresolved", reading: "long interpretation", evidence,
        requirements: [{ id: "hands", queryText: "exact question", status: "unresolved", description: "posture", evidence }],
        spatialOrder: { axis: "x", direction: "ascending", ordinal: 3, candidateIds: ["c"], selectedCandidateId: "c", candidateSet: { status: "unresolved", evidence } } }] },
  } };
  const original = structuredClone(record);
  const anchored = withGroundingContextAnchor([], record);
  const data = snapshot(anchored);
  assert.equal(data.pausedForClarification, true);
  assert.equal(data.declaredState.candidates[0].identityStatus, "unresolved");
  assert.equal(data.declaredState.interpretations[0].conditions[0].status, "unresolved");
  assert.equal(data.declaredState.interpretations[0].spatialOrder.ordinal, 3);
  assert.equal(JSON.stringify(anchored).includes(evidence), false);
  assert.deepEqual(record, original);
});


test("tail anchor survives SDK conversion without changing signed thinking or call/result adjacency", async () => {
  const { convertToLlm } = await import("@earendil-works/pi-coding-agent");
  const assistant = { role: "assistant", content: [{ type: "thinking", thinking: "opaque", thinkingSignature: "signed" },
    { type: "toolCall", id: "a", name: "grounding_view", arguments: {} }, { type: "toolCall", id: "b", name: "grounding_view", arguments: {} }] };
  const a = { role: "toolResult", toolCallId: "a", toolName: "grounding_view", content: [{ type: "text", text: "a" }] };
  const b = { ...a, toolCallId: "b" };
  const history = [assistant, a, b];
  const projected = withGroundingContextAnchor(history, { key: "one", originalQuery: "original question" });
  const converted = convertToLlm(projected);
  assert.deepEqual(converted.slice(0, 3), convertToLlm(history));
  assert.equal(converted[0], assistant);
  assert.equal(converted[1], a);
  assert.equal(converted[2], b);
  assert.equal(converted[3].role, "user");
  assert.match(converted[3].content[0].text, /not a new user request/);
  const incomplete = withGroundingContextAnchor([assistant, a], { key: "one", originalQuery: "original question" });
  assert.ok(isGroundingContextAnchor(incomplete[0]));
  assert.equal(incomplete[1], assistant);
  assert.equal(incomplete[2], a);
});

test("unloaded runtime replaces stale query anchors without changing tool pairing", () => {
  const assistant = { role: "assistant", content: [{ type: "toolCall", id: "a", name: "grounding_view", arguments: {} }] };
  const result = { role: "toolResult", toolCallId: "a", toolName: "grounding_view", content: [{ type: "text", text: "historical" }] };
  const prior = withGroundingContextAnchor([assistant, result], { key: "old", originalQuery: "STALE_QUERY_SENTINEL" });
  const unloaded = { queryPath: "/source/queries.json", outputDir: "/out", pendingKey: null, lastApprovedKey: "old", pausedForClarification: false };
  const projected = withGroundingContextAnchor(prior, undefined, unloaded);
  assert.equal(projected.filter(isGroundingContextAnchor).length, 1);
  assert.equal(snapshot(projected).runtimeState, "record_not_loaded");
  assert.equal(snapshot(projected).lastApprovedKey, "old");
  assert.equal(JSON.stringify(projected).includes("STALE_QUERY_SENTINEL"), false);
  assert.equal(projected[0], assistant);
  assert.equal(projected[1], result);
  const pending = withGroundingContextAnchor([assistant], undefined, unloaded);
  assert.ok(isGroundingContextAnchor(pending[0]));
  assert.equal(pending[1], assistant);
  assert.deepEqual(withGroundingContextAnchor(projected), [assistant, result]);
});
