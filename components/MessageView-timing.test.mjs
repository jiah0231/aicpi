import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, {
  jsx: { runtime: "automatic" },
  tsconfigPaths: true,
});
const React = await jiti.import("react");
const { renderToStaticMarkup } = await jiti.import("react-dom/server");
const { MessageView, ThinkingBlock } = await jiti.import("./MessageView.tsx");
const { I18nProvider } = await jiti.import("@/hooks/useI18n");
const { getLocalePlugin } = await jiti.import("@/lib/i18n/registry");

function renderWithI18n(element) {
  return renderToStaticMarkup(React.createElement(I18nProvider, null, element));
}

function renderMessage(message, props = {}) {
  return renderWithI18n(React.createElement(MessageView, { message, ...props }));
}

function toolCall(id) {
  return { type: "toolCall", toolCallId: id, toolName: "write", input: { path: "result.txt", content: "Saved" } };
}

function toolResult(id, timestamp) {
  return { role: "toolResult", toolCallId: id, timestamp, content: [{ type: "text", text: "Saved" }] };
}

test("labels historical tool timing as model + tool and explains possible confirmation waits", () => {
  const html = renderMessage({
    role: "assistant",
    timestamp: 1_000,
    content: [toolCall("save-1"), toolCall("save-2")],
  }, {
    toolResults: new Map([
      ["save-1", toolResult("save-1", 11_000)],
      ["save-2", toolResult("save-2", 81_000)],
    ]),
  });

  assert.match(html, />Model \+ tool 10s<\/span>/);
  assert.match(html, />Model \+ tool 80s<\/span>/);
  assert.equal((html.match(/data-tool-timing="model-tool-roundtrip"/g) ?? []).length, 2);
  assert.match(html, /title="[^"]*model generation[^"]*confirmation wait; not pure tool execution time\./);
});

test("does not show tool timings from unrelated results, missing timestamps, or reversed clocks", () => {
  for (const toolResults of [
    undefined,
    new Map([["other-call", toolResult("other-call", 10_000)]]),
    new Map([["save-1", toolResult("save-1", undefined)]]),
    new Map([["save-1", toolResult("save-1", NaN)]]),
    new Map([["save-1", toolResult("save-1", 500)]]),
  ]) {
    const html = renderMessage({
      role: "assistant",
      timestamp: 1_000,
      content: [toolCall("save-1")],
    }, { toolResults });
    assert.doesNotMatch(html, /data-tool-timing=/);
  }
});

test("does not infer historical thinking duration from the gap before generation started", () => {
  const html = renderMessage({
    role: "assistant",
    timestamp: 121_000,
    content: [{ type: "thinking", thinking: "Historical reasoning" }],
  }, { prevTimestamp: 1_000 });

  assert.match(html, /Historical reasoning/);
  assert.doesNotMatch(html, /120s/);
  assert.doesNotMatch(html, /font-variant-numeric:tabular-nums/);
});

test("preserves explicitly observed live thinking durations", () => {
  const html = renderWithI18n(React.createElement(ThinkingBlock, {
    block: { type: "thinking", thinking: "Observed reasoning" },
    blockIndex: 0,
    duration: 3,
  }));
  assert.match(html, />3s<\/span>/);
});

test("all built-in locales label the round trip and disclose confirmation waits", () => {
  for (const locale of ["en", "zh-CN", "zh-TW"]) {
    const messages = getLocalePlugin(locale).messages;
    assert.match(messages["tools.timing.roundTrip"], /\{seconds\}/);
    assert.match(messages["tools.timing.roundTripHelp"], /confirmation wait|人工确认等待|人工確認等待/);
  }
});
