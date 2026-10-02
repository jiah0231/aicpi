import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { getStreamFailureDiagnostic, isMissingFinishReason } from "./stream-failure.ts";

test("classifies the exact missing-marker guard, not generic provider failures or length", () => {
  assert.equal(isMissingFinishReason("Stream ended without finish_reason"), true);
  assert.equal(isMissingFinishReason(" Error: Stream ended without finish_reason\n"), true);
  assert.equal(isMissingFinishReason("Provider finish_reason: network_error"), false);
  assert.equal(isMissingFinishReason("unrelated error with Stream ended without finish_reason inside"), false);
  assert.equal(isMissingFinishReason(undefined), false);
  assert.equal(getStreamFailureDiagnostic({ stopReason: "length", errorMessage: "Stream ended without finish_reason" }), null);
  assert.equal(getStreamFailureDiagnostic({ stopReason: "stop" }), null);
});

test("diagnostics contain only bounded categories and counts, not provider payloads", () => {
  const diagnostic = getStreamFailureDiagnostic({
    stopReason: "error", errorMessage: "Stream ended without finish_reason",
    provider: "secret-provider", responseId: "secret-id", headers: { authorization: "secret-key" },
    content: [{ type: "text", text: "secret-prompt" }, { type: "toolCall", arguments: { token: "secret-tool" } }],
  });
  assert.deepEqual(diagnostic, {
    errorClass: "missing_finish_reason", terminationKind: "finish_marker_not_observed",
    receivedBlocks: 2, pendingToolCalls: 1,
  });
  assert.doesNotMatch(JSON.stringify(diagnostic), /secret/);
});

test("retry explanation and historical diagnostic are passive UI without retry actions", () => {
  const notice = readFileSync(new URL("../components/StreamFailureNotice.tsx", import.meta.url), "utf8");
  const input = readFileSync(new URL("../components/ChatInput.tsx", import.meta.url), "utf8");
  assert.match(notice, /chat\.streamFailureHistory/);
  assert.match(notice, /chat\.streamFailureSafety/);
  assert.doesNotMatch(notice, /fetch\(|onClick|set_auto_retry/);
  assert.match(input, /isMissingFinishReason\(retryInfo\.errorMessage\)/);
  assert.match(input, /chat\.streamFailureRetry/);
});
